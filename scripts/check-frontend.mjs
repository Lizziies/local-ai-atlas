// Static sanity checks for the frontend: required ids, no inline handlers/scripts, no nested anchors, balanced key tags.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'public');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const js = fs.readFileSync(path.join(root, 'js', 'atlas.js'), 'utf8');
const errors = [];

const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));
const dup = [...html.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]).filter((x, i, a) => a.indexOf(x) !== i);
if (dup.length) errors.push('duplicate ids: ' + [...new Set(dup)].join(', '));

const dynamic = new Set(['liveOut']); // created at runtime inside the model modal
const wanted = new Set([...js.matchAll(/\$\$?\(\s*['"]#([A-Za-z0-9_-]+)/g)].map(m => m[1]).filter(x => !dynamic.has(x)));
for (const w of wanted) if (!ids.has(w)) errors.push('JS expects missing id: ' + w);

if (/\son[a-z]+\s*=/i.test(html)) errors.push('inline event handler found');
if (/<script(?![^>]*\ssrc=)[^>]*>/i.test(html)) errors.push('inline <script> found');
if (/javascript:/i.test(html)) errors.push('javascript: URL found');

// nested anchors
let depth = 0;
for (const m of html.matchAll(/<(\/?)a\b[^>]*>/gi)) {
  depth += m[1] ? -1 : 1;
  if (depth > 1) { errors.push('nested <a>'); break; }
  if (depth < 0) { errors.push('unbalanced </a>'); break; }
}
const strip = html.replace(/<!--[\s\S]*?-->/g, '').replace(/<script[\s\S]*?<\/script>/gi, '').replace(/<style[\s\S]*?<\/style>/gi, '');
for (const t of ['div', 'section', 'main', 'header', 'footer', 'nav', 'details', 'table', 'ul', 'button', 'a']) {
  const o = (strip.match(new RegExp(`<${t}[\\s>]`, 'gi')) || []).length;
  const c = (strip.match(new RegExp(`</${t}>`, 'gi')) || []).length;
  if (o !== c) errors.push(`<${t}> open ${o} vs close ${c}`);
}
for (const f of ['css/atlas.css', 'css/atlas-extra.css', 'js/atlas.js']) if (!fs.existsSync(path.join(root, f))) errors.push('missing ' + f);

if (errors.length) { console.error('Frontend check FAILED:\n - ' + errors.join('\n - ')); process.exit(1); }
console.log(`Frontend check OK (${ids.size} ids, ${wanted.size} referenced by JS)`);
