import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createApp } from '../src/app.js';
import { HfClient, TtlCache, buildSearchParams, validateId } from '../src/hf.js';

let upstream, upstreamUrl, server, base, app;
const calls = [];
const goodModel = {
  id: 'Qwen/Qwen3.8-27B', author: 'Qwen', pipeline_tag: 'image-text-to-text', downloads: 123, likes: 5, lastModified: '2026-08-14T15:00:01.000Z',
  tags: ['transformers', 'license:apache-2.0', 'region:us'], gated: false, sha: 'abc',
  safetensors: { total: 27781427952, parameters: { BF16: 27781427952 } },
  siblings: [{ rfilename: 'model-00001.safetensors', size: 1000 }, { rfilename: 'secret.txt', size: 5 }],
  cardData: { license: 'apache-2.0' }, secretField: 'must-not-leak', token: 'must-not-leak'
};
const ggufModel = { id: 'bartowski/Qwen3.8-27B-GGUF', tags: [], siblings: [{ rfilename: 'Qwen3.8-27B-Q4_K_M.gguf', size: 17440000000 }, { rfilename: 'mmproj-Qwen3.8-27B-f16.gguf', size: 900000000 }] };

before(async () => {
  upstream = http.createServer((req, res) => {
    calls.push({ url: req.url, auth: req.headers.authorization || null, ua: req.headers['user-agent'] });
    const u = new URL(req.url, 'http://x');
    const send = (code, body, headers = {}) => { res.writeHead(code, { 'content-type': 'application/json', ...headers }); res.end(typeof body === 'string' ? body : JSON.stringify(body)); };
    if (u.pathname === '/api/models') return send(200, [goodModel, { id: 'a/b', downloads: 'x', tags: ['t'] }]);
    if (u.pathname === '/api/models/Qwen/Qwen3.8-27B') return send(200, goodModel);
    if (u.pathname === '/api/models/bartowski/Qwen3.8-27B-GGUF') return send(200, ggufModel);
    if (u.pathname === '/api/models/limited/model') return send(429, { error: 'slow down' }, { 'retry-after': '30' });
    if (u.pathname === '/api/models/boom/model') return send(500, 'oops');
    if (u.pathname === '/api/models/junk/model') return send(200, 'not json at all');
    if (u.pathname === '/api/models/missing/model') return send(404, { error: 'nope' });
    if (u.pathname === '/api/models/slow/model') return; // never answers
    if (u.pathname === '/api/models/big/model') return send(200, JSON.stringify({ pad: 'x'.repeat(3 * 1024 * 1024) }));
    if (u.pathname === '/api/models/redir/model') { res.writeHead(302, { location: 'http://evil.example/steal' }); return res.end(); }
    if (u.pathname === '/api/models/redir2/model') { res.writeHead(302, { location: '/api/models/Qwen/Qwen3.8-27B' }); return res.end(); }
    send(404, { error: 'unknown' });
  });
  await new Promise(r => upstream.listen(0, '127.0.0.1', r));
  upstreamUrl = `http://127.0.0.1:${upstream.address().port}`;
  app = createApp({ hf: { baseUrl: upstreamUrl, token: 'hf_test_secret', timeoutMs: 400, cache: new TtlCache({ ttlMs: 60_000 }) }, rateLimits: { api: 1000, hf: 1000, recommend: 1000 } });
  server = app.listen(0, '127.0.0.1');
  await new Promise(r => server.once('listening', r));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(async () => { app.locals.stop(); server.close(); upstream.close(); upstream.closeAllConnections?.(); });

const get = async (p, init) => { const r = await fetch(base + p, init); const t = await r.text(); let j = null; try { j = JSON.parse(t); } catch {} return { r, t, j }; };
const post = (p, body, headers = { 'content-type': 'application/json' }) => get(p, { method: 'POST', headers, body: typeof body === 'string' ? body : JSON.stringify(body) });

test('health endpoint', async () => {
  const { r, j } = await get('/api/health');
  assert.equal(r.status, 200);
  assert.equal(j.ok, true);
  assert.equal(j.service, 'local-ai-atlas');
  assert.ok(j.catalog.models >= 30);
  assert.ok(!JSON.stringify(j).includes('hf_test_secret'));
});

test('static frontend is served with security headers', async () => {
  const { r, t } = await get('/');
  assert.equal(r.status, 200);
  assert.match(r.headers.get('content-type'), /text\/html/);
  assert.match(t, /Local AI Atlas/);
  assert.match(r.headers.get('content-security-policy'), /script-src 'self'/);
  assert.equal(r.headers.get('x-content-type-options'), 'nosniff');
  assert.equal(r.headers.get('x-powered-by'), null);
});

test('catalogue endpoints', async () => {
  const { j } = await get('/api/catalog');
  assert.ok(j.models.length >= 30 && j.runtimes.length >= 6 && j.harnesses.length >= 6);
  assert.match(j.meta.policy, /NOT automatically current/);
  const one = await get('/api/catalog/models/qwen3.8-27b');
  assert.equal(one.j.name, 'Qwen3.8 27B');
  assert.equal((await get('/api/catalog/models/nope')).r.status, 404);
  const s = await get('/api/catalog/search?q=qwen&kind=harness');
  assert.equal(s.j.count, 1);
  assert.equal((await get('/api/catalog/search?kind=bogus')).r.status, 400);
  assert.equal((await get('/api/catalog/search?q=' + 'a'.repeat(101))).r.status, 400);
});

test('recommend endpoint: happy path, validation, content type, malformed JSON', async () => {
  const ok = await post('/api/recommend', { gpuVram: 12, systemRam: 32, task: 'coding', contextTokens: 32768 });
  assert.equal(ok.r.status, 200);
  assert.ok(ok.j.candidates.length > 0 && ok.j.slots.start);
  assert.match(ok.j.assumptions.note, /never summed/);
  assert.equal((await post('/api/recommend', { gpuVram: 5000, systemRam: 32 })).r.status, 400);
  assert.equal((await post('/api/recommend', '{bad json')).r.status, 400);
  assert.equal((await post('/api/recommend', 'gpuVram=12', { 'content-type': 'text/plain' })).r.status, 415);
  const big = await post('/api/recommend', { gpuVram: 12, systemRam: 32, pad: 'x'.repeat(20000) });
  assert.equal(big.r.status, 413);
});

test('unknown API routes return JSON 404, not the SPA', async () => {
  const { r, j } = await get('/api/does-not-exist');
  assert.equal(r.status, 404);
  assert.equal(j.error, 'not_found');
  assert.equal((await get('/missing.js')).r.status, 404);
});

test('HF search: validated, slimmed, fixed upstream, token stays server-side', async () => {
  calls.length = 0;
  const { r, j } = await get('/api/hf/models?search=qwen&limit=5');
  assert.equal(r.status, 200);
  assert.equal(j.live, true);
  assert.ok(j.retrievedAt);
  assert.equal(j.results[0].id, 'Qwen/Qwen3.8-27B');
  assert.equal(j.results[0].license, 'apache-2.0');
  assert.equal(j.results[1].downloads, null); // malformed upstream value is dropped, not passed through
  assert.ok(!JSON.stringify(j).includes('must-not-leak'));
  assert.ok(!JSON.stringify(j).includes('hf_test_secret'));
  assert.equal(calls[0].auth, 'Bearer hf_test_secret');
  assert.match(calls[0].url, /^\/api\/models\?/);
  const again = await get('/api/hf/models?search=qwen&limit=5');
  assert.equal(again.j.cached, true);
  assert.equal(calls.length, 1);
});

test('HF search rejects bad or unsupported parameters', async () => {
  for (const q of ['', '?search=' + encodeURIComponent('a'.repeat(101)), '?search=x&limit=999', '?search=x&sort=evil', '?search=x&author=..', '?search=x&pipeline_tag=nope', '?search=%3Cscript%3E', '?search=x&filter=a,b,c,d', '?limit=5']) {
    const { r } = await get('/api/hf/models' + q);
    assert.equal(r.status, 400, q);
  }
  // unknown params are ignored, never forwarded
  calls.length = 0;
  await get('/api/hf/models?search=zzz&url=http://evil.example&full=true');
  assert.ok(!calls[0].url.includes('evil') && !calls[0].url.includes('full'));
});

test('HF model lookup: id validation blocks traversal / SSRF-style input', async () => {
  for (const p of ['/api/hf/model/../etc', '/api/hf/model/a%2F..%2Fb/c', '/api/hf/model/a/b%3Fx=1', '/api/hf/model/http:/evil/x', '/api/hf/model/-bad/x']) {
    const { r } = await get(p);
    assert.ok([400, 404].includes(r.status), `${p} -> ${r.status}`);
  }
  assert.throws(() => validateId('a', '..'), /Invalid/);
  assert.throws(() => validateId('a/b', 'c'), /Invalid/);
});

test('HF model lookup returns only whitelisted fields and file sizes', async () => {
  const { r, j } = await get('/api/hf/model/Qwen/Qwen3.8-27B');
  assert.equal(r.status, 200);
  assert.equal(j.model.safetensors.totalParams, 27781427952);
  assert.deepEqual(j.model.files, [{ name: 'model-00001.safetensors', sizeBytes: 1000 }]);
  assert.equal(j.model.secretField, undefined);
});

test('HF failure modes map to clean errors and never crash the server', async () => {
  const cases = [['boom/model', 502, 'upstream_error'], ['junk/model', 502, 'upstream_bad_json'], ['missing/model', 404, 'not_found'], ['slow/model', 502, 'upstream_timeout'], ['big/model', 502, 'upstream_too_large'], ['redir/model', 502, 'blocked_redirect']];
  for (const [id, status, code] of cases) {
    const { r, j } = await get('/api/hf/model/' + id);
    assert.equal(r.status, status, id);
    assert.equal(j.error, code, id);
  }
  assert.equal((await get('/api/health')).r.status, 200);
});

test('upstream 429 becomes a 503 with Retry-After and pauses further upstream calls (isolated app)', async () => {
  const iso = createApp({ hf: { baseUrl: upstreamUrl, cache: new TtlCache() }, rateLimits: { api: 1000, hf: 1000 } });
  const srv = iso.listen(0, '127.0.0.1');
  await new Promise(r => srv.once('listening', r));
  const b = `http://127.0.0.1:${srv.address().port}`;
  const first = await fetch(b + '/api/hf/model/limited/model');
  assert.equal(first.status, 503);
  assert.equal((await first.json()).error, 'upstream_rate_limited');
  assert.equal(first.headers.get('retry-after'), '30');
  calls.length = 0;
  const second = await fetch(b + '/api/hf/model/Qwen/Qwen3.8-27B');
  assert.equal(second.status, 503);
  assert.equal(calls.length, 0); // cooldown: upstream not contacted
  assert.equal((await fetch(b + '/api/health')).status, 200);
  iso.locals.stop(); srv.close();
});

test('same-origin API redirects are followed once; foreign ones are refused', async () => {
  const { r, j } = await get('/api/hf/model/redir2/model');
  assert.equal(r.status, 200);
  assert.equal(j.model.id, 'Qwen/Qwen3.8-27B');
});

test('stale cache is served when Hugging Face fails after a previous success', async () => {
  const cache = new TtlCache({ ttlMs: 1, staleMs: 60_000 });
  let fail = false;
  const client = new HfClient({ baseUrl: 'http://hf.invalid', cache, fetchImpl: async () => {
    if (fail) throw new Error('offline');
    return new Response(JSON.stringify(goodModel), { status: 200, headers: { 'content-type': 'application/json' } });
  } });
  const first = await client.model('Qwen', 'Qwen3.8-27B');
  assert.equal(first.stale, false);
  await new Promise(r => setTimeout(r, 5));
  fail = true;
  const second = await client.model('Qwen', 'Qwen3.8-27B');
  assert.equal(second.stale, true);
  assert.match(second.note, /unavailable/);
  await assert.rejects(client.model('Other', 'Model'), e => e.code === 'upstream_unreachable');
});

test('concurrent identical lookups share one upstream request', async () => {
  let n = 0;
  const client = new HfClient({ baseUrl: 'http://hf.invalid', cache: new TtlCache(), fetchImpl: async () => { n++; await new Promise(r => setTimeout(r, 30)); return new Response(JSON.stringify(goodModel), { status: 200 }); } });
  await Promise.all([client.model('Qwen', 'Qwen3.8-27B'), client.model('Qwen', 'Qwen3.8-27B'), client.model('Qwen', 'Qwen3.8-27B')]);
  assert.equal(n, 1);
});

test('rate-limit cooldown is honoured without hammering upstream', async () => {
  let n = 0;
  const client = new HfClient({ baseUrl: 'http://hf.invalid', cache: new TtlCache(), fetchImpl: async () => { n++; return new Response('{}', { status: 429, headers: { 'retry-after': '120' } }); } });
  await assert.rejects(client.model('a', 'b'), e => e.code === 'upstream_rate_limited' && e.extra.retryAfterSeconds === 120);
  await assert.rejects(client.model('c', 'd'), e => e.code === 'upstream_rate_limited');
  assert.equal(n, 1);
});

test('live comparison endpoint only accepts catalogue ids and compares against Hugging Face', async () => {
  const { r, j } = await get('/api/catalog/models/qwen3.8-27b/live');
  assert.equal(r.status, 200);
  assert.ok(j.retrievedAt && j.curatedAt);
  const lic = j.checks.find(c => c.field === 'license');
  assert.equal(lic.status, 'match');
  const q4 = j.checks.find(c => c.field.startsWith('GGUF Q4_K_M'));
  assert.equal(q4.live, 17.44);
  assert.equal(q4.status, 'match');
  assert.equal((await get('/api/catalog/models/gpt-5.5/live')).r.status, 404);
  assert.equal((await get('/api/catalog/models/evil%2Fid/live')).r.status, 404);
});

test('search-parameter builder', () => {
  assert.equal(buildSearchParams({ search: 'qwen 3.8' }).get('direction'), '-1');
  assert.throws(() => buildSearchParams({}), /at least one/);
});

test('API rate limiting kicks in per client', async () => {
  const limited = createApp({ hf: { baseUrl: upstreamUrl }, rateLimits: { api: 3, hf: 3, recommend: 3 } });
  const srv = limited.listen(0, '127.0.0.1');
  await new Promise(r => srv.once('listening', r));
  const b = `http://127.0.0.1:${srv.address().port}`;
  const codes = [];
  for (let i = 0; i < 5; i++) codes.push((await fetch(b + '/api/catalog/search?q=a')).status);
  assert.deepEqual(codes, [200, 200, 200, 429, 429]);
  assert.equal((await fetch(b + '/api/health')).status, 200); // health is exempt
  limited.locals.stop(); srv.close();
});
