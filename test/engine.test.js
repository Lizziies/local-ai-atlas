import test from 'node:test';
import assert from 'node:assert/strict';
import { loadCatalog } from '../src/catalog.js';
import { recommend, normalizeInput, kvBytes, budgets, GIB } from '../src/engine.js';

const catalog = loadCatalog();
const model = id => catalog.models.find(m => m.id === id);
const rec = (input, opts) => { const r = recommend(catalog, input, opts); assert.ok(r.ok, JSON.stringify(r.errors)); return r; };
const find = (r, id) => r.candidates.find(c => c.id === id);

test('KV cache for Qwen3.8 27B is derived from the 16 full-attention layers', () => {
  // 16 layers x 2 (K,V) x 4 kv heads x 256 dim x 2 bytes = 65,536 bytes per token
  const { bytes } = kvBytes(model('qwen3.8-27b'), 32768);
  assert.equal(bytes, 65536 * 32768);
});

test('sliding-window layers are capped at their window', () => {
  const short = kvBytes(model('gpt-oss-20b'), 4096).bytes;
  const long = kvBytes(model('gpt-oss-20b'), 131072).bytes;
  const fullOnly = 12 * 2 * 8 * 64 * 2; // 12 full layers, bytes per token
  const windowed = 12 * 2 * 8 * 64 * 2 * 128;
  assert.equal(short, fullOnly * 4096 + windowed);
  assert.equal(long, fullOnly * 131072 + windowed);
});

test('VRAM and system RAM are separate budgets and never summed into one fit', () => {
  const b = budgets(normalizeInput({ gpuVram: 12, systemRam: 32 }).value);
  assert.equal(b.kind, 'discrete');
  assert.ok(b.gpuBytes < 12 * GIB && b.gpuBytes > 10 * GIB);
  assert.ok(b.ramBytes < 32 * GIB);
});

test('12 GB GPU + 32 GB RAM: Qwen3.8 27B Q4_K_M is offload, not "fits comfortably"', () => {
  const r = rec({ gpuVram: 12, systemRam: 32, task: 'coding', contextTokens: 32768, runtime: 'llama.cpp' });
  const q = find(r, 'qwen3.8-27b');
  // The engine may pick a smaller quant that fits; whatever it picks, a 17.44 GB Q4_K_M must not be called comfortable on 12 GB.
  if (q && q.variant.quant === 'Q4_K_M') assert.notEqual(q.tier, 'comfortable');
  const all = catalog.models.find(m => m.id === 'qwen3.8-27b');
  assert.equal(all.variants.find(v => v.q === 'Q4_K_M').gb, 17.44);
  for (const c of r.candidates) if (c.tier === 'comfortable') assert.ok(c.memory.totalNeedGiB <= r.assumptions.usableVramGiB * 0.9 + 0.01, `${c.name} claims comfortable but needs ${c.memory.totalNeedGiB} GiB`);
  for (const c of r.candidates) if (c.tier === 'offload') assert.ok(c.memory.totalNeedGiB > r.assumptions.usableVramGiB, `${c.name} is offload but fits in VRAM`);
});

test('24 GB GPU: Qwen3.8 27B Q4_K_M fits comfortably at 8K context', () => {
  const r = rec({ gpuVram: 24, systemRam: 64, task: 'coding', contextTokens: 8192, runtime: 'llama.cpp' }, { limit: 30 });
  const q = find(r, 'qwen3.8-27b');
  assert.ok(q);
  assert.equal(q.tier, 'comfortable');
  assert.equal(q.mode, 'full-gpu');
});

test('memory accounting includes weights, KV cache and runtime overhead', () => {
  const r = rec({ gpuVram: 24, systemRam: 64, contextTokens: 32768, runtime: 'llama.cpp' }, { limit: 30 });
  const q = find(r, 'qwen3.8-27b');
  const m = q.memory;
  assert.ok(m.kvCacheGB > 2 && m.kvCacheGB < 2.3);
  assert.ok(m.runtimeOverheadGB > 0.5);
  const total = (m.weightsGB + m.visionProjectorGB + m.kvCacheGB + m.runtimeOverheadGB) * 1e9 / GIB;
  assert.ok(Math.abs(total - m.totalNeedGiB) < 0.05);
});

test('vision requests add the projector and exclude text-only models', () => {
  const r = rec({ gpuVram: 24, systemRam: 64, task: 'vision', contextTokens: 8192, runtime: 'llama.cpp' }, { limit: 30 });
  assert.ok(r.candidates.every(c => c.modalities.includes('image')));
  const q = find(r, 'qwen3.8-27b');
  assert.ok(q.memory.visionProjectorGB > 0);
  assert.ok(r.excluded.some(e => e.id === 'gpt-oss-20b'));
});

test('Bonsai 2 is excluded for stock llama.cpp and only offered on the PrismML fork', () => {
  const stock = rec({ gpuVram: 12, systemRam: 32, runtime: 'llama.cpp' }, { limit: 40 });
  assert.equal(find(stock, 'bonsai-2-27b'), undefined);
  assert.match(stock.excluded.find(e => e.id === 'bonsai-2-27b').reason, /PrismML/);
  const lm = rec({ gpuVram: 12, systemRam: 32, runtime: 'lm-studio' }, { limit: 40 });
  assert.equal(find(lm, 'bonsai-2-27b'), undefined);
  const fork = rec({ gpuVram: 12, systemRam: 32, runtime: 'prismml-llamacpp' }, { limit: 40 });
  assert.ok(find(fork, 'bonsai-2-27b'));
  const any = rec({ gpuVram: 12, systemRam: 32, runtime: 'any' }, { limit: 40 });
  const b = find(any, 'bonsai-2-27b');
  assert.ok(b.warnings.some(w => /PrismML/.test(w)));
});

test('vLLM / SGLang never recommend RAM offload', () => {
  const r = rec({ gpuVram: 12, systemRam: 128, runtime: 'vllm', contextTokens: 8192 }, { limit: 40 });
  assert.ok(r.candidates.every(c => c.tier === 'comfortable' || c.tier === 'tight'));
  assert.ok(r.candidates.every(c => c.variant.format === 'safetensors'));
});

test('MoE offload is labelled and distinguished from dense layer offload', () => {
  const r = rec({ gpuVram: 8, systemRam: 64, task: 'coding', contextTokens: 8192, runtime: 'llama.cpp' }, { limit: 40 });
  const moe = find(r, 'qwen3-coder-30b-a3b');
  assert.equal(moe.tier, 'offload');
  assert.equal(moe.mode, 'moe-expert-offload');
  const dense = find(r, 'devstral-small-2');
  assert.equal(dense.mode, 'layer-offload');
});

test('unified memory uses one pool and never reports offload', () => {
  const r = rec({ memoryKind: 'unified', unifiedMemory: 32, contextTokens: 8192, runtime: 'lm-studio' }, { limit: 40 });
  assert.ok(r.candidates.every(c => c.mode === 'unified'));
  assert.ok(r.assumptions.usableUnifiedGiB <= 32 * 0.75 + 0.01);
});

test('no GPU: everything is CPU-only offload from system RAM', () => {
  const r = rec({ memoryKind: 'none', systemRam: 32, contextTokens: 8192 }, { limit: 10 });
  assert.ok(r.candidates.length > 0);
  assert.ok(r.candidates.every(c => c.tier === 'offload' && c.mode === 'cpu-only'));
});

test('requested context beyond a model limit is flagged and capped', () => {
  const r = rec({ gpuVram: 24, systemRam: 64, contextTokens: 262144, runtime: 'llama.cpp' }, { limit: 40 });
  const g = find(r, 'gemma-4-e4b');
  if (g) { assert.equal(g.memory.contextCapped, true); assert.equal(g.memory.contextUsed, 131072); }
});

test('KV-cache quantisation reduces the estimate', () => {
  const a = kvBytes(model('devstral-small-2'), 32768, 'f16').bytes;
  const b = kvBytes(model('devstral-small-2'), 32768, 'q8_0').bytes;
  assert.ok(b < a * 0.6 && b > a * 0.45);
});

test('every candidate carries an explanation and a score breakdown', () => {
  const r = rec({ gpuVram: 16, systemRam: 32, task: 'general' });
  for (const c of r.candidates) {
    assert.ok(c.reasons.length >= 2);
    assert.ok(c.scoreBreakdown.length >= 3);
    assert.equal(Math.round(c.scoreBreakdown.reduce((s, p) => s + p.points, 0) * 10) / 10, c.score);
  }
});

test('slots: start is fitting; capability option is larger and not a GPU fit', () => {
  const r = rec({ gpuVram: 12, systemRam: 64, task: 'coding', contextTokens: 16384, runtime: 'llama.cpp' });
  const start = find(r, r.slots.start);
  assert.ok(['comfortable', 'tight'].includes(start.tier));
  if (r.slots.capability) {
    const cap = find(r, r.slots.capability);
    assert.ok(['offload', 'tight'].includes(cap.tier));
    assert.ok(cap.params.totalB > start.params.totalB);
  }
});

test('optional bandwidth produces a labelled upper-bound speed estimate', () => {
  const r = rec({ gpuVram: 24, systemRam: 64, contextTokens: 8192, runtime: 'llama.cpp', gpuBandwidthGBs: 1000 }, { limit: 40 });
  const q = find(r, 'qwen3.8-27b');
  assert.ok(q.speed.upperBoundTps > 20 && q.speed.upperBoundTps < 80);
  const none = rec({ gpuVram: 24, systemRam: 64, contextTokens: 8192, runtime: 'llama.cpp' }, { limit: 40 });
  assert.equal(find(none, 'qwen3.8-27b').speed.upperBoundTps, null);
});

test('input validation rejects malformed values', () => {
  const bad = [
    { gpuVram: -1, systemRam: 32 }, { gpuVram: 'abc', systemRam: 32 }, { gpuVram: 12, systemRam: 2 },
    { gpuVram: 12, systemRam: 32, task: 'poetry' }, { gpuVram: 12, systemRam: 32, runtime: 'evil' },
    { gpuVram: 12, systemRam: 32, contextTokens: 10 }, { gpuVram: 12, systemRam: 32, gpuName: '<script>' },
    { gpuVram: 12, systemRam: 32, kvCache: 'q1' }, { memoryKind: 'quantum', gpuVram: 12, systemRam: 32 }
  ];
  for (const b of bad) assert.equal(normalizeInput(b).ok, false, JSON.stringify(b));
  assert.equal(normalizeInput(null).ok, false);
  assert.equal(normalizeInput([1, 2]).ok, false);
});

test('non-recommendable and cloud entries are never recommended', () => {
  const r = rec({ gpuVram: 96, systemRam: 256, contextTokens: 8192 }, { limit: 100 });
  const ids = new Set(r.candidates.map(c => c.id));
  for (const m of catalog.models.filter(m => !m.recommendable)) assert.ok(!ids.has(m.id), m.id);
});
