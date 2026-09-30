// Hugging Face client for the Atlas backend.
//
// Not an open proxy: the upstream host is fixed, only two endpoint shapes are reachable, every
// parameter is validated against an allow-list, and responses are reduced to a small set of
// fields before they are returned. Failures never throw into the request handler.

const ID_PART = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const SEARCH_TEXT = /^[\p{L}\p{N} .+\-_/:()]{1,100}$/u;
const TOKEN = /^[a-z0-9_:.-]{1,40}$/i;
const SORTS = new Set(['downloads', 'likes', 'lastModified', 'createdAt', 'trendingScore']);
const PIPELINES = new Set(['text-generation', 'image-text-to-text', 'any-to-any', 'text2text-generation', 'automatic-speech-recognition', 'feature-extraction', 'image-to-text', 'visual-question-answering', 'text-to-speech']);

export class HfError extends Error {
  constructor(status, code, message, extra = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.extra = extra;
  }
}

export function validateId(owner, repo) {
  if (!ID_PART.test(owner || '') || !ID_PART.test(repo || '') || owner.includes('..') || repo.includes('..')) {
    throw new HfError(400, 'invalid_model_id', 'Invalid model identifier. Expected owner/name using letters, digits, ".", "_" and "-".');
  }
  return `${owner}/${repo}`;
}

/** Build a validated upstream query string for the model search endpoint. */
export function buildSearchParams(query) {
  const p = new URLSearchParams();
  const q = query || {};
  const one = v => (Array.isArray(v) ? v[0] : v);

  if (q.search != null) {
    const s = String(one(q.search)).trim();
    if (!SEARCH_TEXT.test(s)) throw new HfError(400, 'invalid_search', 'search must be 1-100 characters of letters, digits and simple punctuation.');
    p.set('search', s);
  }
  if (q.author != null) {
    const a = String(one(q.author)).trim();
    if (!ID_PART.test(a)) throw new HfError(400, 'invalid_author', 'author is not a valid Hugging Face user or organisation name.');
    p.set('author', a);
  }
  if (q.pipeline_tag != null) {
    const t = String(one(q.pipeline_tag));
    if (!PIPELINES.has(t)) throw new HfError(400, 'invalid_pipeline_tag', `pipeline_tag must be one of: ${[...PIPELINES].join(', ')}`);
    p.set('pipeline_tag', t);
  }
  if (q.filter != null) {
    const list = (Array.isArray(q.filter) ? q.filter : String(q.filter).split(',')).map(x => String(x).trim()).filter(Boolean);
    if (list.length > 3 || list.some(x => !TOKEN.test(x))) throw new HfError(400, 'invalid_filter', 'filter accepts up to 3 simple tags such as "gguf" or "safetensors".');
    for (const f of list) p.append('filter', f);
  }
  const sort = q.sort == null ? 'downloads' : String(one(q.sort));
  if (!SORTS.has(sort)) throw new HfError(400, 'invalid_sort', `sort must be one of: ${[...SORTS].join(', ')}`);
  p.set('sort', sort);
  p.set('direction', '-1');
  let limit = q.limit == null ? 12 : Number(one(q.limit));
  if (!Number.isInteger(limit) || limit < 1 || limit > 30) throw new HfError(400, 'invalid_limit', 'limit must be an integer between 1 and 30.');
  p.set('limit', String(limit));
  if (!p.has('search') && !p.has('author') && !p.has('pipeline_tag') && !p.has('filter')) {
    throw new HfError(400, 'missing_query', 'Provide at least one of search, author, pipeline_tag or filter.');
  }
  return p;
}

const pickTags = tags => (Array.isArray(tags) ? tags.filter(t => typeof t === 'string' && t.length <= 60).slice(0, 24) : []);

function licenseFrom(m) {
  const fromTag = (m.tags || []).find(t => typeof t === 'string' && t.startsWith('license:'));
  const c = m.cardData && typeof m.cardData === 'object' ? m.cardData.license : null;
  return (fromTag ? fromTag.slice(8) : null) || (typeof c === 'string' ? c : null);
}

export function slimListItem(m) {
  return {
    id: String(m.id || m.modelId || ''),
    author: m.author ?? null,
    pipelineTag: m.pipeline_tag ?? null,
    downloads: Number.isFinite(m.downloads) ? m.downloads : null,
    likes: Number.isFinite(m.likes) ? m.likes : null,
    lastModified: m.lastModified ?? null,
    createdAt: m.createdAt ?? null,
    gated: m.gated ?? false,
    license: licenseFrom(m),
    tags: pickTags(m.tags),
    url: `https://huggingface.co/${String(m.id || m.modelId || '')}`
  };
}

export function slimModel(m) {
  const st = m.safetensors && typeof m.safetensors === 'object' ? m.safetensors : null;
  const files = Array.isArray(m.siblings)
    ? m.siblings
        .filter(s => s && typeof s.rfilename === 'string' && /\.(gguf|safetensors)$/i.test(s.rfilename) && Number.isFinite(s.size))
        .slice(0, 80)
        .map(s => ({ name: s.rfilename.slice(0, 200), sizeBytes: s.size }))
    : [];
  return {
    ...slimListItem(m),
    sha: typeof m.sha === 'string' ? m.sha.slice(0, 40) : null,
    libraryName: m.library_name ?? null,
    safetensors: st ? { totalParams: Number.isFinite(st.total) ? st.total : null, dtypes: st.parameters && typeof st.parameters === 'object' ? st.parameters : null } : null,
    gguf: m.gguf && typeof m.gguf === 'object' ? { total: m.gguf.total ?? null, architecture: m.gguf.architecture ?? null, contextLength: m.gguf.context_length ?? null } : null,
    files
  };
}

/** Small bounded cache with stale-if-error support. */
export class TtlCache {
  constructor({ max = 200, ttlMs = 10 * 60 * 1000, staleMs = 24 * 60 * 60 * 1000 } = {}) {
    this.max = max; this.ttlMs = ttlMs; this.staleMs = staleMs; this.map = new Map();
  }
  get(key) {
    const hit = this.map.get(key);
    if (!hit) return null;
    const age = Date.now() - hit.time;
    if (age > this.staleMs) { this.map.delete(key); return null; }
    this.map.delete(key); this.map.set(key, hit); // refresh LRU position
    return { value: hit.value, fresh: age <= this.ttlMs, time: hit.time };
  }
  set(key, value) {
    this.map.delete(key);
    this.map.set(key, { value, time: Date.now() });
    while (this.map.size > this.max) this.map.delete(this.map.keys().next().value);
  }
  get size() { return this.map.size; }
}

export class HfClient {
  constructor({ baseUrl = 'https://huggingface.co', token = process.env.HF_TOKEN, fetchImpl = globalThis.fetch, timeoutMs = 8000, maxBytes = 2 * 1024 * 1024, cache = new TtlCache(), userAgent = 'Local-AI-Atlas/2.0 (+https://github.com/Lizziies/local-ai-atlas)' } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.token = token || null;
    this.fetch = fetchImpl;
    this.timeoutMs = timeoutMs;
    this.maxBytes = maxBytes;
    this.cache = cache;
    this.userAgent = userAgent;
    this.cooldownUntil = 0;
    this.inflight = new Map();
    this.stats = { upstreamCalls: 0, cacheHits: 0, staleServed: 0, failures: 0 };
  }

  status() {
    return { tokenConfigured: !!this.token, cooldownUntil: this.cooldownUntil > Date.now() ? new Date(this.cooldownUntil).toISOString() : null, cacheEntries: this.cache.size, ...this.stats };
  }

  async #upstream(path) {
    const url = new URL(this.baseUrl + path);
    if (!(url.pathname === '/api/models' || url.pathname.startsWith('/api/models/'))) throw new HfError(500, 'blocked_path', 'Blocked upstream path.');
    const headers = { 'User-Agent': this.userAgent, Accept: 'application/json' };
    if (this.token) headers.Authorization = `Bearer ${this.token}`;
    let current = url;
    for (let hop = 0; hop < 3; hop++) {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), this.timeoutMs);
      let res;
      try {
        this.stats.upstreamCalls++;
        res = await this.fetch(current, { headers, signal: ac.signal, redirect: 'manual' });
      } catch (e) {
        throw new HfError(502, e.name === 'AbortError' ? 'upstream_timeout' : 'upstream_unreachable', e.name === 'AbortError' ? 'Hugging Face did not answer in time.' : 'Hugging Face could not be reached.');
      } finally { clearTimeout(timer); }

      if (res.status >= 300 && res.status < 400) {
        const loc = res.headers.get('location');
        let next;
        try { next = new URL(loc, current); } catch { throw new HfError(502, 'bad_redirect', 'Hugging Face returned an invalid redirect.'); }
        if (next.origin !== url.origin || !next.pathname.startsWith('/api/models')) throw new HfError(502, 'blocked_redirect', 'Redirect target not allowed.');
        current = next; continue;
      }
      if (res.status === 429) {
        const ra = Number(res.headers.get('retry-after'));
        const wait = Number.isFinite(ra) && ra > 0 ? Math.min(ra, 600) : 60;
        this.cooldownUntil = Date.now() + wait * 1000;
        throw new HfError(429, 'upstream_rate_limited', 'Hugging Face rate limit reached. Try again shortly.', { retryAfterSeconds: wait });
      }
      if (res.status === 404) throw new HfError(404, 'not_found', 'Model not found on Hugging Face (it may be private, gated or renamed).');
      if (res.status === 401 || res.status === 403) {
        // A single-model path means gated/private; on a search path it means Hugging Face refused this service.
        if (/^\/api\/models\/[^?]+/.test(path)) throw new HfError(404, 'not_accessible', 'This model is not publicly accessible.');
        throw new HfError(502, 'upstream_refused', 'Hugging Face refused the request from this server. Try again later.');
      }
      if (!res.ok) throw new HfError(502, 'upstream_error', `Hugging Face answered with status ${res.status}.`);
      const len = Number(res.headers.get('content-length'));
      if (Number.isFinite(len) && len > this.maxBytes) throw new HfError(502, 'upstream_too_large', 'Upstream response too large.');
      const text = await readLimited(res, this.maxBytes);
      try { return JSON.parse(text); } catch { throw new HfError(502, 'upstream_bad_json', 'Hugging Face returned a response that was not valid JSON.'); }
    }
    throw new HfError(502, 'too_many_redirects', 'Too many redirects.');
  }

  /** Cached fetch with in-flight de-duplication, cooldown handling and stale-if-error. */
  async #cached(key, path, shape) {
    const hit = this.cache.get(key);
    if (hit?.fresh) { this.stats.cacheHits++; return { data: hit.value, retrievedAt: new Date(hit.time).toISOString(), cached: true, stale: false }; }
    const cooling = this.cooldownUntil > Date.now();
    if (cooling && hit) { this.stats.staleServed++; return { data: hit.value, retrievedAt: new Date(hit.time).toISOString(), cached: true, stale: true, note: 'Hugging Face is rate limiting this service; showing the last saved response.' }; }
    if (cooling) throw new HfError(429, 'upstream_rate_limited', 'Hugging Face rate limit reached. Try again shortly.', { retryAfterSeconds: Math.ceil((this.cooldownUntil - Date.now()) / 1000) });
    if (this.inflight.has(key)) return this.inflight.get(key);
    const p = (async () => {
      try {
        const raw = await this.#upstream(path);
        const data = shape(raw);
        this.cache.set(key, data);
        return { data, retrievedAt: new Date().toISOString(), cached: false, stale: false };
      } catch (e) {
        this.stats.failures++;
        if (hit && e instanceof HfError && e.status >= 429) {
          this.stats.staleServed++;
          return { data: hit.value, retrievedAt: new Date(hit.time).toISOString(), cached: true, stale: true, note: 'Hugging Face is unavailable right now; showing the last saved response.' };
        }
        throw e;
      } finally { this.inflight.delete(key); }
    })();
    this.inflight.set(key, p);
    return p;
  }

  search(query) {
    const params = buildSearchParams(query);
    const qs = params.toString();
    return this.#cached(`search:${qs}`, `/api/models?${qs}`, raw => {
      if (!Array.isArray(raw)) throw new HfError(502, 'upstream_bad_shape', 'Unexpected response shape from Hugging Face.');
      return raw.slice(0, 30).map(slimListItem);
    });
  }

  model(owner, repo) {
    const id = validateId(owner, repo);
    return this.#cached(`model:${id.toLowerCase()}`, `/api/models/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}?blobs=true`, raw => {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new HfError(502, 'upstream_bad_shape', 'Unexpected response shape from Hugging Face.');
      return slimModel(raw);
    });
  }
}

async function readLimited(res, maxBytes) {
  if (!res.body || typeof res.body.getReader !== 'function') {
    const t = await res.text();
    if (t.length > maxBytes) throw new HfError(502, 'upstream_too_large', 'Upstream response too large.');
    return t;
  }
  const reader = res.body.getReader();
  const chunks = []; let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) { try { await reader.cancel(); } catch {} throw new HfError(502, 'upstream_too_large', 'Upstream response too large.'); }
    chunks.push(value);
  }
  return Buffer.concat(chunks).toString('utf8');
}
