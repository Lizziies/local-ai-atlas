import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadCatalog, searchCatalog } from './catalog.js';
import { recommend, TASKS, CONTEXT_OPTIONS, RUNTIME_CHOICES, SPEED_PREFS, KV_CACHE_TYPES } from './engine.js';
import { HfClient, HfError, validateId } from './hf.js';
import { rateLimit } from './ratelimit.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(here, '..', 'public');
const APP_VERSION = '2.0.0';

const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
  "frame-ancestors 'none'"
].join('; ');

function securityHeaders(_req, res, next) {
  res.set({
    'Content-Security-Policy': CSP,
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    'Cross-Origin-Opener-Policy': 'same-origin'
  });
  next();
}

function sendError(res, status, code, message, extra = {}) {
  res.status(status).set('Cache-Control', 'no-store').json({ error: code, message, ...extra });
}

export function createApp({ catalog = loadCatalog(), hf = {}, rateLimits = {}, trustProxy = process.env.TRUST_PROXY ?? '1' } = {}) {
  const app = express();
  app.disable('x-powered-by');
  const tp = String(trustProxy);
  app.set('trust proxy', tp === 'true' ? true : tp === 'false' || tp === '0' ? false : Number.isInteger(Number(tp)) ? Number(tp) : tp);
  app.use(securityHeaders);

  const hfClient = hf instanceof HfClient ? hf : new HfClient(hf);
  const startedAt = Date.now();
  const modelsById = new Map(catalog.models.map(m => [m.id, m]));

  const apiLimit = rateLimit({ windowMs: 60_000, max: rateLimits.api ?? 120, name: 'the API' });
  const hfLimit = rateLimit({ windowMs: 60_000, max: rateLimits.hf ?? 30, name: 'the Hugging Face endpoints' });
  const recLimit = rateLimit({ windowMs: 60_000, max: rateLimits.recommend ?? 60, name: 'the recommendation endpoint' });

  // Health check stays outside the rate limiter so Render's probes are never throttled.
  app.get('/api/health', (_req, res) => {
    res.set('Cache-Control', 'no-store').json({
      ok: true,
      service: 'local-ai-atlas',
      version: APP_VERSION,
      time: new Date().toISOString(),
      uptimeSeconds: Math.round((Date.now() - startedAt) / 1000),
      catalog: { curatedAt: catalog.meta.curatedAt, models: catalog.models.length, recommendable: catalog.models.filter(m => m.recommendable).length, runtimes: catalog.runtimes.length, harnesses: catalog.harnesses.length },
      huggingFace: hfClient.status()
    });
  });

  app.use('/api', apiLimit);
  app.use(express.json({ limit: '8kb' }));

  app.get('/api/catalog', (_req, res) => {
    res.set('Cache-Control', 'public, max-age=300').json({
      ...catalog,
      notice: 'Curated snapshot, not live data. See /api/catalog/models/:id/live to compare a model against Hugging Face right now.'
    });
  });

  app.get('/api/catalog/search', (req, res) => {
    const q = String(req.query.q ?? '');
    const kind = String(req.query.kind ?? 'all');
    const task = String(req.query.task ?? '');
    if (q.length > 100) return sendError(res, 400, 'invalid_query', 'q must be at most 100 characters.');
    if (!['all', 'model', 'runtime', 'harness', 'concept'].includes(kind)) return sendError(res, 400, 'invalid_kind', 'kind must be all, model, runtime, harness or concept.');
    if (task && !TASKS.includes(task)) return sendError(res, 400, 'invalid_task', `task must be one of ${TASKS.join(', ')}.`);
    const results = searchCatalog(catalog, { q, kind, task });
    res.set('Cache-Control', 'no-store').json({ count: results.length, results });
  });

  app.get('/api/catalog/models/:id', (req, res) => {
    const m = modelsById.get(req.params.id);
    if (!m) return sendError(res, 404, 'not_found', 'Unknown model id.');
    res.set('Cache-Control', 'public, max-age=300').json(m);
  });

  app.get('/api/options', (_req, res) => {
    res.set('Cache-Control', 'public, max-age=3600').json({ tasks: TASKS, contexts: CONTEXT_OPTIONS, runtimes: RUNTIME_CHOICES, speedPrefs: SPEED_PREFS, kvCache: Object.keys(KV_CACHE_TYPES) });
  });

  app.post('/api/recommend', recLimit, (req, res) => {
    if (!req.is('application/json')) return sendError(res, 415, 'unsupported_media_type', 'Send the request as application/json.');
    const result = recommend(catalog, req.body);
    if (!result.ok) return sendError(res, 400, 'invalid_input', 'The request could not be understood.', { details: result.errors });
    res.set('Cache-Control', 'no-store').json({ ...result, catalogCuratedAt: catalog.meta.curatedAt });
  });

  /* ---- Hugging Face (fixed upstream, validated, cached) ---- */
  const hfFail = (res, e) => {
    if (e instanceof HfError) {
      if (e.extra.retryAfterSeconds) res.set('Retry-After', String(e.extra.retryAfterSeconds));
      const status = e.status === 429 ? 503 : e.status;
      return sendError(res, status, e.code, e.message, e.extra);
    }
    console.error('Unexpected Hugging Face handler error:', e?.message);
    return sendError(res, 500, 'internal_error', 'Unexpected server error.');
  };

  app.get('/api/hf/models', hfLimit, async (req, res) => {
    try {
      const r = await hfClient.search(req.query);
      res.set('Cache-Control', 'no-store').json({ source: 'huggingface.co/api/models', live: true, retrievedAt: r.retrievedAt, cached: r.cached, stale: r.stale, note: r.note, count: r.data.length, results: r.data });
    } catch (e) { hfFail(res, e); }
  });

  app.get('/api/hf/model/:owner/:repo', hfLimit, async (req, res) => {
    try {
      const r = await hfClient.model(req.params.owner, req.params.repo);
      res.set('Cache-Control', 'no-store').json({ source: `huggingface.co/api/models/${req.params.owner}/${req.params.repo}`, live: true, retrievedAt: r.retrievedAt, cached: r.cached, stale: r.stale, note: r.note, model: r.data });
    } catch (e) { hfFail(res, e); }
  });

  // Compare one curated catalogue entry with today's Hugging Face metadata. Only catalogue ids are accepted.
  app.get('/api/catalog/models/:id/live', hfLimit, async (req, res) => {
    const m = modelsById.get(req.params.id);
    if (!m) return sendError(res, 404, 'not_found', 'Unknown model id.');
    if (!m.links?.hf) return sendError(res, 404, 'no_hf_repo', 'This catalogue entry has no Hugging Face repository (for example, cloud models).');
    try {
      const [owner, repo] = m.links.hf.split('/');
      validateId(owner, repo);
      const main = await hfClient.model(owner, repo);
      let gguf = null;
      if (m.links.gguf && m.links.gguf !== m.links.hf) {
        const [go, gr] = m.links.gguf.split('/');
        try { gguf = await hfClient.model(go, gr); } catch { gguf = null; }
      }
      res.set('Cache-Control', 'no-store').json(compareLive(m, main, gguf, catalog.meta.curatedAt));
    } catch (e) { hfFail(res, e); }
  });

  app.all('/api/*splat', (_req, res) => sendError(res, 404, 'not_found', 'Unknown API route.'));

  // JSON body errors etc.
  // eslint-disable-next-line no-unused-vars
  app.use('/api', (err, _req, res, _next) => {
    if (err?.type === 'entity.parse.failed') return sendError(res, 400, 'invalid_json', 'Request body is not valid JSON.');
    if (err?.type === 'entity.too.large') return sendError(res, 413, 'payload_too_large', 'Request body too large.');
    console.error('API error:', err?.message);
    return sendError(res, 500, 'internal_error', 'Unexpected server error.');
  });

  app.use(express.static(PUBLIC_DIR, {
    index: 'index.html',
    extensions: ['html'],
    maxAge: '5m',
    setHeaders(res, file) { if (file.endsWith('.html')) res.set('Cache-Control', 'no-cache'); }
  }));

  // Single-page fallback only for browser navigations, never for missing assets or API paths.
  app.get('*splat', (req, res, next) => {
    if (path.extname(req.path) || !(req.headers.accept || '').includes('text/html')) return next();
    res.set('Cache-Control', 'no-cache').sendFile(path.join(PUBLIC_DIR, 'index.html'));
  });

  app.use((_req, res) => res.status(404).type('text/plain').send('Not found'));
  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => { console.error('Server error:', err?.message); res.status(500).type('text/plain').send('Server error'); });

  app.locals.stop = () => { apiLimit.stop(); hfLimit.stop(); recLimit.stop(); };
  app.locals.hf = hfClient;
  return app;
}

const normLicense = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');

export function compareLive(model, main, gguf, curatedAt) {
  const live = main.data;
  const checks = [];
  const curatedLic = normLicense(model.license);
  const liveLic = normLicense(live.license);
  checks.push({
    field: 'license',
    curated: model.license,
    live: live.license,
    status: !live.license ? 'unknown' : curatedLic.includes(liveLic) || liveLic.includes(curatedLic.slice(0, 8)) ? 'match' : 'differs'
  });
  if (live.safetensors?.totalParams && model.params?.totalB) {
    const liveB = live.safetensors.totalParams / 1e9;
    const ratio = liveB / model.params.totalB;
    checks.push({ field: 'total parameters (B)', curated: model.params.totalB, live: Math.round(liveB * 100) / 100, status: ratio > 0.9 && ratio < 1.1 ? 'match' : 'differs', note: 'Hugging Face safetensors totals can include vision towers or embeddings.' });
  }
  if (gguf) {
    for (const v of model.variants.filter(x => x.fmt === 'gguf' && x.repo === model.links.gguf)) {
      const token = v.q.split(' ')[0].toLowerCase();
      const files = gguf.data.files.filter(f => f.name.toLowerCase().includes(token) && !/mmproj/i.test(f.name));
      if (!files.length) { checks.push({ field: `GGUF ${v.q} size (GB)`, curated: v.gb, live: null, status: 'unknown', note: 'No matching file found in the live listing.' }); continue; }
      const liveGB = Math.round(files.reduce((s, f) => s + f.sizeBytes, 0) / 1e7) / 100;
      checks.push({ field: `GGUF ${v.q} size (GB)`, curated: v.gb, live: liveGB, status: Math.abs(liveGB / v.gb - 1) < 0.03 ? 'match' : 'differs' });
    }
  }
  return {
    model: { id: model.id, name: model.name },
    curatedAt,
    retrievedAt: main.retrievedAt,
    cached: main.cached,
    stale: main.stale,
    note: main.note,
    live: { id: live.id, lastModified: live.lastModified, downloads: live.downloads, license: live.license, gated: live.gated, url: live.url },
    checks
  };
}
