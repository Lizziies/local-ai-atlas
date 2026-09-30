import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { loadCatalog, validateCatalog, searchCatalog, DEFAULT_CATALOG_PATH } from '../src/catalog.js';

test('shipped catalogue passes structural validation', () => {
  const c = loadCatalog();
  assert.deepEqual(validateCatalog(c), []);
  assert.ok(c.models.length >= 30);
});

test('validation catches bad data', () => {
  const c = JSON.parse(fs.readFileSync(DEFAULT_CATALOG_PATH, 'utf8'));
  c.models[0].variants[0].basis = 'made-up';
  c.models[1].params.activeB = c.models[1].params.totalB + 1;
  c.models[2].variants[0].rt = ['nonexistent-runtime'];
  const errs = validateCatalog(c);
  assert.ok(errs.some(e => /unknown basis/.test(e)));
  assert.ok(errs.some(e => /activeB exceeds totalB/.test(e)));
  assert.ok(errs.some(e => /unknown runtime/.test(e)));
});

test('spot-check verified figures against the primary sources recorded during the audit', () => {
  const c = loadCatalog();
  const m = id => c.models.find(x => x.id === id);
  const v = (id, q) => m(id).variants.find(x => x.q === q);
  assert.equal(v('qwen3.8-27b', 'Q4_K_M').gb, 17.44);
  assert.equal(m('deepseek-v4-flash').params.totalB, 284);
  assert.equal(m('deepseek-v4-flash').params.activeB, 13);
  assert.equal(m('gpt-oss-20b').params.activeB, 3.6);
  assert.equal(m('gpt-oss-120b').params.activeB, 5.1);
  assert.equal(m('gemma-4-26b-a4b').params.activeB, 3.8);
  assert.equal(v('bonsai-2-27b', 'PTQ1_0 (1.75 bpw)').gb, 5.95);
  assert.deepEqual(v('bonsai-2-27b', 'PTQ1_0 (1.75 bpw)').rt, ['prismml-llamacpp']);
  assert.ok(!m('deepseek-v4-flash').modalities.includes('image'));
  assert.equal(m('deepseek-v4-pro').recommendable, false);
});

test('MoE entries always distinguish total from active parameters', () => {
  for (const m of loadCatalog().models.filter(x => /mixture-of-experts/.test(x.arch) && x.params.totalB)) {
    assert.ok(m.params.activeB && m.params.activeB < m.params.totalB, m.id);
  }
});

test('search finds models, runtimes, harnesses and concepts', () => {
  const c = loadCatalog();
  assert.ok(searchCatalog(c, { q: 'qwen code' }).some(r => r.kind === 'harness' && r.id === 'qwen-code'));
  assert.ok(searchCatalog(c, { q: 'llama.cpp', kind: 'runtime' }).length >= 1);
  assert.ok(searchCatalog(c, { q: 'retrieval' }).some(r => r.kind === 'concept'));
  assert.equal(searchCatalog(c, { q: 'zzzzzz-no-match' }).length, 0);
  assert.ok(searchCatalog(c, { kind: 'model', task: 'coding' }).length > 3);
});
