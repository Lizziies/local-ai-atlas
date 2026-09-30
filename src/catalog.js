import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_CATALOG_PATH = path.join(here, '..', 'data', 'catalog.json');

const BASES = new Set(['file', 'vendor', 'derived', 'rough']);

/** Structural validation so a bad edit to catalog.json fails at startup and in tests, not in production. */
export function validateCatalog(c) {
  const errors = [];
  const ids = new Set();
  if (!c || typeof c !== 'object') return ['catalog is not an object'];
  for (const k of ['meta', 'basis', 'runtimes', 'models', 'harnesses', 'concepts']) if (!c[k]) errors.push(`missing "${k}"`);
  if (errors.length) return errors;
  const runtimeIds = new Set(c.runtimes.map(r => r.id));
  for (const m of c.models) {
    const where = `model ${m.id}`;
    if (!m.id || ids.has(m.id)) errors.push(`${where}: missing or duplicate id`);
    ids.add(m.id);
    for (const k of ['name', 'vendor', 'kind', 'modalities', 'tasks', 'license', 'summary', 'links', 'variants', 'kv', 'params']) if (m[k] == null) errors.push(`${where}: missing ${k}`);
    if (m.recommendable && !m.variants?.length) errors.push(`${where}: recommendable but has no variants`);
    if (m.recommendable && !(m.kv && (m.kv.layers?.length || typeof m.kv.bytesPerToken === 'number'))) errors.push(`${where}: recommendable but no KV data`);
    for (const v of m.variants || []) {
      if (!(v.gb > 0)) errors.push(`${where}: variant ${v.q} has invalid size`);
      if (!BASES.has(v.basis)) errors.push(`${where}: variant ${v.q} has unknown basis "${v.basis}"`);
      for (const r of [...(v.rt || []), ...(v.rtLikely || [])]) if (!runtimeIds.has(r)) errors.push(`${where}: variant ${v.q} references unknown runtime "${r}"`);
    }
    if (m.params?.totalB && m.params?.activeB && m.params.activeB > m.params.totalB) errors.push(`${where}: activeB exceeds totalB`);
    for (const key of ['official', 'lmStudio', 'github']) {
      const u = m.links?.[key];
      if (u && !/^https:\/\//.test(u)) errors.push(`${where}: link ${key} must be https`);
    }
  }
  const hIds = new Set();
  for (const h of c.harnesses) { if (hIds.has(h.id)) errors.push(`duplicate harness ${h.id}`); hIds.add(h.id); }
  return errors;
}

export function loadCatalog(file = DEFAULT_CATALOG_PATH) {
  const catalog = JSON.parse(fs.readFileSync(file, 'utf8'));
  const errors = validateCatalog(catalog);
  if (errors.length) throw new Error(`Invalid catalog:\n - ${errors.join('\n - ')}`);
  return catalog;
}

const norm = s => String(s).toLowerCase();

/** Search across models, runtimes, harnesses and concepts. */
export function searchCatalog(catalog, { q = '', kind = 'all', task = '' } = {}) {
  const needle = norm(q).trim();
  const rows = [];
  for (const m of catalog.models) rows.push({ kind: 'model', id: m.id, name: m.name, summary: m.summary, tags: [m.vendor, m.family, m.kind, ...m.tasks, ...m.modalities], task: m.tasks, ref: m });
  for (const r of catalog.runtimes) rows.push({ kind: 'runtime', id: r.id, name: r.name, summary: r.summary, tags: [r.kind, ...r.formats, ...r.backends], task: [], ref: r });
  for (const h of catalog.harnesses) rows.push({ kind: 'harness', id: h.id, name: h.name, summary: h.summary, tags: ['harness', h.license], task: ['coding', 'agent'], ref: h });
  for (const c of catalog.concepts) rows.push({ kind: 'concept', id: c.id, name: c.name, summary: c.definition, tags: [c.examples], task: [], ref: c });
  return rows
    .filter(r => (kind === 'all' || r.kind === kind) && (!task || r.task.includes(task)) && (!needle || norm(`${r.name} ${r.summary} ${r.tags.join(' ')}`).includes(needle)))
    .map(({ ref, ...r }) => r);
}
