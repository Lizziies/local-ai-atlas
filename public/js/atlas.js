/* Local AI Atlas front-end. Plain script (no build step), no inline handlers, CSP-safe.
   All dynamic text goes through esc(); nothing from the network is inserted as raw HTML. */
(function () {
  'use strict';

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; });
  }
  function hfUrl(id) { return 'https://huggingface.co/' + String(id).split('/').map(encodeURIComponent).join('/'); }
  function safeUrl(u) { return /^https:\/\//.test(u || '') ? u : ''; }
  function debounce(fn, ms) { var t; return function () { var a = arguments, s = this; clearTimeout(t); t = setTimeout(function () { fn.apply(s, a); }, ms); }; }
  function fmtDate(iso) { try { return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }); } catch (e) { return iso; } }
  function fmtNum(n) { return n == null ? '–' : Number(n).toLocaleString('en-US'); }
  function fmtCtx(n) { return n ? (n >= 1000000 ? (n / 1000000) + 'M' : Math.round(n / 1024) + 'K') + ' tokens' : 'not verified'; }

  var BASIS = {
    file: ['measured', 'Measured file size from the repository listing'],
    vendor: ['vendor', 'Stated by the vendor / model card'],
    derived: ['derived', 'Computed by the Atlas from official parameters or config.json'],
    rough: ['rough', 'Unverified rough estimate']
  };
  var TIER = {
    comfortable: ['FITS COMFORTABLY', 'tier-ok'],
    tight: ['FITS, TIGHT HEADROOM', 'tier-tight'],
    offload: ['POSSIBLE WITH OFFLOAD', 'tier-offload'],
    'not-recommended': ['NOT RECOMMENDED', 'tier-no']
  };
  var MODE = {
    'full-gpu': 'Entirely in GPU memory',
    'unified': 'Unified memory (single pool)',
    'moe-expert-offload': 'MoE expert offload to system RAM',
    'layer-offload': 'Layer offload to system RAM',
    'cpu-only': 'CPU only, from system RAM'
  };
  var SPEED = { fast: 'Fast', moderate: 'Moderate', 'moderate-slow': 'Moderate to slow', slow: 'Slow', unknown: 'Unknown' };
  var RUNTIME_NAMES = { 'llama.cpp': 'llama.cpp', 'lm-studio': 'LM Studio', ollama: 'Ollama', vllm: 'vLLM', sglang: 'SGLang', mlx: 'MLX', 'prismml-llamacpp': 'PrismML llama.cpp fork', transformers: 'Transformers' };
  var TASK_NAMES = { general: 'General assistant', coding: 'Coding', agent: 'Agentic / tool use', reasoning: 'Reasoning', vision: 'Vision / documents' };

  var state = { catalog: null, backend: false, filter: { kind: 'all', tag: 'all', q: '' }, lastRec: null };

  /* ------------------------------------------------------------------ API */
  function api(path, init) {
    var ac = new AbortController();
    var timer = setTimeout(function () { ac.abort(); }, 15000);
    init = Object.assign({ headers: { Accept: 'application/json' }, signal: ac.signal }, init || {});
    return fetch(path, init).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) { var e = new Error(body.message || ('Request failed (' + r.status + ')')); e.status = r.status; e.body = body; throw e; }
        return body;
      });
    }).finally(function () { clearTimeout(timer); });
  }
  function errText(e) {
    if (e && e.name === 'AbortError') return 'The request timed out.';
    if (e && e.status === 429 || e && e.status === 503) return (e.message || 'Rate limited.') + ' Please wait a moment and try again.';
    return (e && e.message) || 'Something went wrong.';
  }

  /* --------------------------------------------------------- backend status */
  function setStatus(ok, health) {
    var chip = $('#backendStatus'), panel = $('#backendPanelStatus');
    state.backend = ok;
    if (chip) { chip.textContent = ok ? 'BACKEND • LIVE' : 'BACKEND • OFFLINE'; chip.classList.toggle('hot', !ok); }
    if (panel) {
      panel.textContent = ok
        ? 'API connected · ' + health.catalog.models + ' curated model records (' + health.catalog.recommendable + ' recommendable) · catalogue curated ' + health.catalog.curatedAt + ' · Hugging Face client ' + (health.huggingFace.cooldownUntil ? 'is rate-limit paused' : 'ready') + ' · checked ' + fmtDate(health.time)
        : 'The backend did not answer. Static explanations still work; the recommender, model explorer and live Hugging Face features need the API.';
    }
    var banner = $('#offlineBanner');
    if (banner) banner.hidden = ok;
  }

  /* ---------------------------------------------------------- recommender */
  var VRAM_PRESETS = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96];
  var form = { memoryKind: 'discrete', vram: 12, ram: 32, unified: 32 };

  function renderVramButtons() {
    var host = $('#vramButtons'); if (!host) return;
    host.innerHTML = VRAM_PRESETS.map(function (v) {
      return '<button type="button" class="vbtn' + (v === form.vram ? ' active' : '') + '" data-vram="' + v + '" aria-pressed="' + (v === form.vram) + '">' + v + ' GB</button>';
    }).join('');
    var custom = $('#vramCustom'); if (custom && VRAM_PRESETS.indexOf(form.vram) < 0) custom.value = form.vram;
  }
  function syncMemoryKind() {
    $$('[data-kind-panel]').forEach(function (p) { p.hidden = p.getAttribute('data-kind-panel') !== form.memoryKind; });
    $$('#memoryKind button').forEach(function (b) { var on = b.dataset.kind === form.memoryKind; b.classList.toggle('active', on); b.setAttribute('aria-pressed', on); });
    var vis = $('#advGpuRow'); if (vis) vis.hidden = form.memoryKind === 'none';
  }
  function readInput() {
    var v = {
      memoryKind: form.memoryKind,
      task: $('#recTask').value,
      contextTokens: Number($('#recCtx').value),
      runtime: $('#recRuntime').value,
      speedPref: $('#recSpeed').value,
      kvCache: $('#recKv').value,
      needsVision: $('#recVision').checked
    };
    if (form.memoryKind === 'discrete') { v.gpuVram = form.vram; v.systemRam = Number($('#recRam').value); }
    else if (form.memoryKind === 'unified') { v.unifiedMemory = Number($('#recUnified').value); }
    else { v.systemRam = Number($('#recRam2').value); }
    var g = $('#recGpuName').value.trim(); if (g) v.gpuName = g;
    var c = $('#recCpuName').value.trim(); if (c) v.cpuName = c;
    var gb = $('#recGpuBw').value.trim(); if (gb) v.gpuBandwidthGBs = Number(gb);
    var rb = $('#recRamBw').value.trim(); if (rb) v.ramBandwidthGBs = Number(rb);
    return v;
  }
  function hint(input) {
    var out = [];
    if (input.gpuName) {
      var n = input.gpuName.toLowerCase();
      if (/nvidia|geforce|rtx|gtx|quadro|tesla|\ba100\b|\bh100\b/.test(n)) out.push('NVIDIA GPU: CUDA backends in llama.cpp / LM Studio / vLLM.');
      else if (/amd|radeon|\brx\b|instinct/.test(n)) out.push('AMD GPU: llama.cpp supports HIP/ROCm and Vulkan; check your GPU generation.');
      else if (/apple|\bm[1-9]\b|\bm[1-9] (pro|max|ultra)/.test(n)) out.push('Apple silicon: consider the "Apple unified memory" option and MLX / LM Studio.');
      else if (/intel|arc/.test(n)) out.push('Intel GPU: llama.cpp offers SYCL/Vulkan backends; support varies.');
    }
    return out;
  }

  var runRec = debounce(function () {
    var host = $('#recResults'); if (!host) return;
    var input = readInput();
    host.setAttribute('aria-busy', 'true');
    api('/api/recommend', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(input) })
      .then(function (r) { state.lastRec = r; renderRec(r); renderStack(r); })
      .catch(function (e) {
        host.innerHTML = '<div class="notice-box warn" role="alert"><b>Could not compute a recommendation.</b> ' + esc(errText(e)) + (e.body && e.body.details ? '<ul>' + e.body.details.map(function (d) { return '<li>' + esc(d) + '</li>'; }).join('') + '</ul>' : '') + '</div>';
        var s = $('#stackOut'); if (s) s.innerHTML = '<p class="small">Set your hardware in the chooser above to generate a stack.</p>';
      })
      .finally(function () { host.removeAttribute('aria-busy'); });
  }, 250);

  function memBar(c, r) {
    var m = c.memory, a = r.assumptions;
    var cap = a.usableVramGiB != null ? a.usableVramGiB : a.usableUnifiedGiB != null ? a.usableUnifiedGiB : a.usableSystemRamGiB;
    var GBtoGiB = 1e9 / Math.pow(1024, 3);
    var parts = [['w', m.weightsGB * GBtoGiB, 'Weights ' + m.weightsGB + ' GB'], ['v', m.visionProjectorGB * GBtoGiB, 'Vision ' + m.visionProjectorGB + ' GB'], ['k', m.kvCacheGB * GBtoGiB, 'KV cache ' + m.kvCacheGB + ' GB'], ['o', m.runtimeOverheadGB * GBtoGiB, 'Runtime ' + m.runtimeOverheadGB + ' GB']].filter(function (p) { return p[1] > 0; });
    var scale = Math.max(m.totalNeedGiB, cap) || 1;
    var segs = parts.map(function (p) { return '<i class="seg seg-' + p[0] + '" style="width:' + (p[1] / scale * 100).toFixed(2) + '%" title="' + esc(p[2]) + '"></i>'; }).join('');
    var marker = '<b class="cap" style="left:' + Math.min(100, cap / scale * 100).toFixed(2) + '%" title="Usable ' + (a.usableVramGiB != null ? 'VRAM' : a.usableUnifiedGiB != null ? 'unified memory' : 'system RAM') + ': ' + cap + ' GiB"></b>';
    var label = a.usableVramGiB != null ? 'GPU budget' : a.usableUnifiedGiB != null ? 'unified budget' : 'RAM budget';
    return '<div class="membar" role="img" aria-label="Estimated need ' + m.totalNeedGiB + ' GiB against a ' + label + ' of ' + cap + ' GiB">' + segs + marker + '</div>' +
      '<div class="memlegend"><span><i class="lg seg-w"></i>weights</span>' + (m.visionProjectorGB ? '<span><i class="lg seg-v"></i>vision</span>' : '') + '<span><i class="lg seg-k"></i>KV cache</span><span><i class="lg seg-o"></i>runtime</span><span class="capnote">| = ' + label + ' (' + cap + ' GiB)</span></div>';
  }

  function candidateCard(c, r, slotLabel) {
    var t = TIER[c.tier] || TIER['not-recommended'];
    var p = c.params || {};
    var params = p.totalB ? (p.activeB && p.activeB < p.totalB ? p.totalB + 'B total / ' + (p.activeLabel || p.activeB + 'B active') : p.totalB + 'B dense') : 'size not verified';
    var rt = c.runtimes.list.map(function (x) { return RUNTIME_NAMES[x] || x; }).join(', ');
    var speed = SPEED[c.speed.tier] || c.speed.tier;
    var ub = c.speed.upperBoundTps ? ' · theoretical upper bound ≈ ' + c.speed.upperBoundTps + ' tok/s (memory-bandwidth bound; real speed is lower)' : '';
    var other = c.otherVariants.length ? '<div class="small">Other builds: ' + c.otherVariants.map(function (o) { return esc(o.quant) + ' (' + o.sizeGB + ' GB, ' + (TIER[o.tier] ? TIER[o.tier][0].toLowerCase() : o.tier) + ')'; }).join(' · ') + '</div>' : '';
    return '<article class="candidate ' + t[1] + '">' +
      '<div class="candtop"><div>' + (slotLabel ? '<div class="uk">' + esc(slotLabel) + '</div>' : '') + '<h3>' + esc(c.name) + '</h3><div class="small">' + esc(c.vendor) + ' · ' + esc(params) + (c.status === 'experimental-preview' ? ' · <b>experimental preview</b>' : '') + '</div></div><span class="tierbadge ' + t[1] + '">' + t[0] + '</span></div>' +
      '<div class="tagrow"><span class="tag p">' + esc(c.variant.quant) + ' · ' + c.variant.sizeGB + ' GB</span><span class="tag">' + esc(MODE[c.mode] || c.mode) + '</span><span class="tag m">Speed: ' + esc(speed) + '</span>' + (c.taskMatch ? '<span class="tag">task match</span>' : '') + '</div>' +
      memBar(c, r) +
      '<ul class="reasons">' + c.reasons.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') + '</ul>' +
      (c.warnings.length ? '<ul class="warns">' + c.warnings.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') + '</ul>' : '') +
      '<div class="small">' + esc(c.speed.note) + esc(ub) + '</div>' +
      '<div class="small">Runtimes: ' + esc(rt) + (c.runtimes.verified ? '' : ' (not verified for this build)') + ' · ' + esc(c.license) + ' · size basis: ' + esc((BASIS[c.variant.sizeBasis] || [c.variant.sizeBasis])[0]) + '</div>' +
      other +
      '<details class="why"><summary>How was this scored?</summary><table class="mini"><tbody>' + c.scoreBreakdown.map(function (b) { return '<tr><td>' + esc(b.label) + '</td><td class="num">' + (b.points > 0 ? '+' : '') + b.points + '</td></tr>'; }).join('') + '<tr><th>Total (heuristic ranking score)</th><th class="num">' + c.score + '</th></tr></tbody></table></details>' +
      '<div class="chips">' + (c.links && c.links.official ? '<a class="ub" href="' + esc(safeUrl(c.links.official)) + '" target="_blank" rel="noopener">Official ↗</a>' : '') + '<button type="button" class="ub linkbtn" data-open-model="' + esc(c.id) + '">Model card</button></div>' +
      '</article>';
  }

  function renderRec(r) {
    var host = $('#recResults'), a = r.assumptions, i = r.input;
    var budgetLine = i.memoryKind === 'discrete'
      ? 'Usable VRAM <b>' + a.usableVramGiB + ' GiB</b> (of ' + i.vramGiB + ' GB, minus ' + a.gpuReserveGiB + ' GiB driver/desktop reserve) · usable system RAM <b>' + a.usableSystemRamGiB + ' GiB</b> (of ' + i.systemRamGiB + ' GB). Kept separate.'
      : i.memoryKind === 'unified'
        ? 'Usable unified memory <b>' + a.usableUnifiedGiB + ' GiB</b> (' + Math.round(a.unifiedUsableFraction * 100) + '% of ' + i.unifiedGiB + ' GB). One shared pool, so no separate offload tier.'
        : 'No GPU: usable system RAM <b>' + a.usableSystemRamGiB + ' GiB</b> (of ' + i.systemRamGiB + ' GB). Everything runs on the CPU.';
    var html = '<div class="assume"><div>' + budgetLine + '</div><div class="small">Context ' + i.contextTokens.toLocaleString('en-US') + ' tokens · KV cache ' + esc(i.kvCache) + ' · runtime ' + esc(RUNTIME_NAMES[i.runtime] || 'any') + ' · task ' + esc(TASK_NAMES[i.task] || i.task) + '. File sizes are decimal GB; your memory is treated as GiB. Catalogue curated ' + esc(r.catalogCuratedAt) + '.</div>' + hint(i).map(function (h) { return '<div class="small">' + esc(h) + '</div>'; }).join('') + '</div>';
    var byId = {}; r.candidates.forEach(function (c) { byId[c.id] = c; });
    var slots = [];
    if (r.slots.start && byId[r.slots.start]) slots.push(['Suggested starting point', byId[r.slots.start]]);
    if (r.slots.speed && byId[r.slots.speed]) slots.push(['Speed-oriented alternative', byId[r.slots.speed]]);
    if (r.slots.capability && byId[r.slots.capability]) slots.push(['Higher-capability option requiring more memory / offload', byId[r.slots.capability]]);
    if (!slots.length) {
      html += '<div class="notice-box warn" role="status"><b>Nothing in the curated catalogue fits this setup with the chosen runtime and context.</b> Try a shorter context, a quantised KV cache, another runtime, or adding system RAM for offload.</div>';
    } else {
      html += '<div class="slotgrid">' + slots.map(function (s) { return candidateCard(s[1], r, s[0]); }).join('') + '</div>';
    }
    var used = {}; slots.forEach(function (s) { used[s[1].id] = 1; });
    var rest = r.candidates.filter(function (c) { return !used[c.id]; });
    if (rest.length) html += '<h3 class="subhead">More candidates <span class="small">(ranked by fit, documented task focus and a parameter-count capacity proxy, not by benchmark)</span></h3><div class="slotgrid">' + rest.map(function (c) { return candidateCard(c, r); }).join('') + '</div>';
    if (r.notRecommended && r.notRecommended.length) {
      html += '<details class="why nrec"><summary>Not recommended on this hardware (' + r.notRecommended.length + ' closest)</summary><table class="mini"><tbody>' + r.notRecommended.map(function (n) { return '<tr><td>' + esc(n.name) + ' <span class="small">' + esc(n.quant) + '</span></td><td class="small">' + esc(n.reason) + '</td></tr>'; }).join('') + '</tbody></table></details>';
    }
    if (r.excluded && r.excluded.length) {
      var groups = {}; r.excluded.forEach(function (e) { (groups[e.reason] = groups[e.reason] || []).push(e.name); });
      html += '<details class="why nrec"><summary>Excluded by your filters (' + r.excluded.length + ')</summary><table class="mini"><tbody>' + Object.keys(groups).map(function (k) { return '<tr><td class="small">' + esc(groups[k].join(', ')) + '</td><td class="small">' + esc(k) + '</td></tr>'; }).join('') + '</tbody></table></details>';
    }
    html += '<p class="small disclaimer">' + esc(r.disclaimer) + '</p>';
    host.innerHTML = html;
  }

  /* ---------------------------------------------------- stack from result */
  function renderStack(r) {
    var host = $('#stackOut'); if (!host || !state.catalog) return;
    var start = r.candidates.filter(function (c) { return c.id === r.slots.start; })[0];
    if (!start) { host.innerHTML = '<p class="small">No fitting model for the current inputs, so no stack can be suggested.</p>'; return; }
    var task = r.input.task;
    var harnesses = state.catalog.harnesses.filter(function (h) {
      if (h.facts.localModels !== 'yes') return false;
      if (task === 'coding') return ['qwen-code', 'opencode', 'cline', 'openhands'].indexOf(h.id) >= 0;
      if (task === 'agent') return ['qwen-code', 'opencode', 'goose', 'openhands'].indexOf(h.id) >= 0;
      return ['bionic', 'goose'].indexOf(h.id) >= 0;
    });
    if (task === 'general' || task === 'vision' || task === 'reasoning') harnesses = state.catalog.harnesses.filter(function (h) { return h.id === 'bionic' || h.id === 'goose'; });
    var rts = start.runtimes.list.map(function (x) { return RUNTIME_NAMES[x] || x; });
    var unverified = state.catalog.harnesses.filter(function (h) { return harnesses.indexOf(h) >= 0 && h.facts.localModels !== 'yes'; });
    host.innerHTML = '<div class="uflow stackflow"><div class="unode">MODEL<small>' + esc(start.name) + ' · ' + esc(start.variant.quant) + '</small></div><div class="uarrow" aria-hidden="true">→</div><div class="unode">RUNTIME<small>' + esc(rts.join(' / ')) + '</small></div><div class="uarrow" aria-hidden="true">→</div><div class="unode">HARNESS<small>' + esc(harnesses.map(function (h) { return h.name; }).join(' / ') || '–') + '</small></div><div class="uarrow" aria-hidden="true">→</div><div class="unode">TOOLS<small>MCP servers, shell, files, with least privilege</small></div></div>' +
      '<p class="small">Model: ' + esc(start.reasons[0]) + '</p>' +
      '<p class="small">Harness names come from the harness comparison below; ' + (task === 'coding' ? '<b>Qwen Code</b> is the featured coding harness in this Atlas, but it is one option among several, not a ranking. ' : '') + 'Every listed harness can talk to a local OpenAI-compatible server (the runtime), but tool-calling quality depends on the specific model and template.</p>' + (unverified.length ? '' : '');
  }

  /* --------------------------------------------------------- model explorer */
  function kindOf(m) { return m.kind === 'cloud' ? 'cloud' : m.kind === 'open-weight-server' ? 'server' : 'local'; }
  function modelSize(m) {
    var p = m.params || {};
    if (!p.totalB && !p.activeB) return 'size varies';
    if (p.activeB && p.totalB && p.activeB < p.totalB) return p.totalB + 'B total · ' + (p.activeLabel || p.activeB + 'B active');
    return (p.totalB || p.activeB) + 'B dense';
  }
  function modelCard(m) {
    var k = kindOf(m);
    var v = m.variants && m.variants.length ? m.variants.slice().sort(function (a, b) { return a.gb - b.gb; }) : [];
    var mem = k === 'cloud' ? 'Provider managed (no local memory)' : v.length ? 'Files ' + v[0].gb + '–' + v[v.length - 1].gb + ' GB (' + v.length + ' builds)' : 'No verified local builds';
    var pills = [k === 'cloud' ? 'CLOUD' : k === 'server' ? 'SERVER-SCALE' : 'LOCAL / OPEN-WEIGHT'].concat(m.tasks.slice(0, 3).map(function (x) { return x.toUpperCase(); }));
    return '<article class="modelcard" tabindex="0" role="button" aria-label="Open model card: ' + esc(m.name) + '" data-model="' + esc(m.id) + '"><div class="modeltop"><div><div class="modelname">' + esc(m.name) + '</div><div class="meta">' + esc(mem) + '</div></div><span class="badge ' + (k === 'cloud' ? 'pink' : k === 'server' ? 'mint' : 'cyan') + '">' + (k === 'cloud' ? 'CLOUD' : k === 'server' ? 'SERVER' : 'LOCAL') + '</span></div><div class="badges">' + pills.map(function (x, i) { return '<span class="badge' + (i === 0 ? ' pink' : '') + '">' + esc(x) + '</span>'; }).join('') + '</div><p class="small">' + esc(modelSize(m)) + (m.status === 'experimental-preview' ? ' · experimental preview' : m.status === 'legacy' ? ' · legacy generation' : '') + '</p><div class="cardfoot"><span class="small">' + (m.recommendable ? 'In recommender' : 'Reference only') + '</span><span class="go" aria-hidden="true">↗</span></div></article>';
  }
  function renderModels() {
    var host = $('#modelGrid'); if (!host || !state.catalog) return;
    var f = state.filter, q = f.q.toLowerCase();
    var arr = state.catalog.models.filter(function (m) {
      return (f.kind === 'all' || kindOf(m) === f.kind) && (f.tag === 'all' || m.tasks.indexOf(f.tag) >= 0 || (f.tag === 'vision' && m.modalities.indexOf('image') >= 0)) &&
        (!q || (m.name + ' ' + m.vendor + ' ' + m.family + ' ' + m.summary).toLowerCase().indexOf(q) >= 0);
    });
    host.innerHTML = arr.map(modelCard).join('') || '<div class="upanel"><h3>No match.</h3><p>Try clearing a filter.</p></div>';
    var c = $('#modelCount'); if (c) c.textContent = arr.length + ' of ' + state.catalog.models.length + ' curated entries';
  }

  var lastFocus = null;
  function openModel(id) {
    var m = state.catalog && state.catalog.models.filter(function (x) { return x.id === id; })[0]; if (!m) return;
    var modal = $('#modal');
    lastFocus = document.activeElement;
    $('#modalTitle').textContent = m.name;
    var k = kindOf(m);
    $('#modalMeta').textContent = (k === 'cloud' ? 'Cloud' : k === 'server' ? 'Open-weight, server-scale' : 'Local / open-weight') + ' · ' + m.vendor + ' · ' + m.license;
    var p = m.params || {};
    var facts = [
      ['Parameters', p.totalB ? (p.activeB && p.activeB < p.totalB ? p.totalB + 'B total, ' + (p.activeLabel || p.activeB + 'B active') : p.totalB + 'B (dense)') : 'not verified'],
      ['Architecture', m.arch], ['Context limit', fmtCtx(m.contextMax)], ['Modalities', m.modalities.join(', ')], ['Documented focus', m.tasks.join(', ')], ['Status', m.status]
    ];
    var basisLegend = '<div class="small legend">Basis: ' + Object.keys(BASIS).map(function (b) { return '<span class="basis b-' + b + '" title="' + esc(BASIS[b][1]) + '">' + BASIS[b][0] + '</span>'; }).join(' ') + '</div>';
    var variants = m.variants.length ? '<table class="mini variants"><thead><tr><th>Build</th><th>Format</th><th>Size</th><th>Runtimes</th></tr></thead><tbody>' + m.variants.map(function (v) {
      var rt = (v.rt || []).map(function (x) { return RUNTIME_NAMES[x] || x; }).join(', '); var lk = (v.rtLikely || []).map(function (x) { return RUNTIME_NAMES[x] || x; });
      return '<tr><td>' + esc(v.q) + '</td><td>' + esc(v.fmt) + '</td><td>' + v.gb + ' GB <span class="basis b-' + esc(v.basis) + '">' + esc((BASIS[v.basis] || [v.basis])[0]) + '</span>' + (v.mm ? '<br><span class="small">+ vision projector ' + v.mm + ' GB (' + esc((BASIS[v.mmBasis] || [''])[0]) + ')</span>' : '') + '</td><td>' + esc(rt || '–') + (lk.length ? '<br><span class="small">possibly: ' + esc(lk.join(', ')) + ' (unverified)</span>' : '') + '</td></tr>';
    }).join('') + '</tbody></table>' : '<p class="small">No local builds are listed for this entry' + (k === 'cloud' ? ' (cloud only).' : ' because none were verified.') + '</p>';
    var kv = m.kv && m.kv.layers ? '<p class="small">KV cache: ' + (m.kv.basis === 'derived' ? 'derived from the official config.json' : 'approximated from config.json') + '. ' + esc(m.kv.note || '') + '</p>' : (m.kv && m.kv.bytesPerToken ? '<p class="small">KV cache: rough estimate only (not verified). ' + esc(m.kv.note || '') + '</p>' : '');
    var cav = (m.caveats || []).length ? '<ul class="warns">' + m.caveats.map(function (c) { return '<li>' + esc(c) + '</li>'; }).join('') + '</ul>' : '';
    var L = m.links || {}; var links = [];
    if (L.official) links.push(['Official', L.official]);
    if (L.hf) links.push(['Hugging Face', hfUrl(L.hf)]);
    if (L.gguf) links.push(['GGUF files', hfUrl(L.gguf)]);
    if (L.github) links.push(['GitHub', L.github]);
    if (L.lmStudio) links.push(['LM Studio', L.lmStudio]);
    var live = L.hf ? '<div class="livebox"><button type="button" class="btn" id="liveCheck" data-model-id="' + esc(m.id) + '">Compare with live Hugging Face ↻</button><div id="liveOut" class="small" aria-live="polite">Curated data was verified on ' + esc(state.catalog.meta.curatedAt) + '. This button fetches today\'s metadata from Hugging Face and compares license, parameter count and file sizes.</div></div>' : '';
    $('#modalBody').innerHTML = '<p style="font-size:16px">' + esc(m.summary) + '</p><table class="mini facts"><tbody>' + facts.map(function (f) { return '<tr><th>' + esc(f[0]) + '</th><td>' + esc(f[1]) + '</td></tr>'; }).join('') + '</tbody></table>' + '<h3 class="subhead">Builds and sizes</h3>' + basisLegend + variants + kv + cav + live +
      '<div class="chips" style="margin-top:16px">' + links.map(function (l, i) { return '<a class="btn' + (i === 0 ? ' primary' : '') + '" href="' + esc(safeUrl(l[1])) + '" target="_blank" rel="noopener">' + esc(l[0]) + ' ↗</a>'; }).join('') + '</div>';
    modal.classList.add('show'); modal.removeAttribute('hidden');
    document.body.classList.add('modal-open');
    $('#closeModal').focus();
  }
  function closeModal() {
    var modal = $('#modal'); if (!modal.classList.contains('show')) return;
    modal.classList.remove('show'); modal.setAttribute('hidden', ''); document.body.classList.remove('modal-open');
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  function runLiveCheck(btn) {
    var out = $('#liveOut'); btn.disabled = true; out.textContent = 'Asking Hugging Face…';
    api('/api/catalog/models/' + encodeURIComponent(btn.dataset.modelId) + '/live').then(function (r) {
      var mark = { match: '✓ matches', differs: '⚠ differs', unknown: '? not comparable' };
      out.innerHTML = '<b>LIVE FROM HUGGING FACE</b> · retrieved ' + esc(fmtDate(r.retrievedAt)) + (r.cached ? ' (from the server cache)' : '') + (r.stale ? ' · <b>stale copy</b>' : '') + (r.note ? '<br>' + esc(r.note) : '') +
        '<br>Repository <a href="' + esc(safeUrl(r.live.url)) + '" target="_blank" rel="noopener">' + esc(r.live.id) + '</a> · last modified ' + esc(fmtDate(r.live.lastModified)) + ' · ' + fmtNum(r.live.downloads) + ' downloads' +
        '<table class="mini"><thead><tr><th>Field</th><th>Curated</th><th>Live</th><th>Result</th></tr></thead><tbody>' + r.checks.map(function (c) { return '<tr class="chk-' + esc(c.status) + '"><td>' + esc(c.field) + '</td><td>' + esc(c.curated) + '</td><td>' + esc(c.live == null ? '–' : c.live) + '</td><td>' + esc(mark[c.status] || c.status) + (c.note ? '<br><span class="small">' + esc(c.note) + '</span>' : '') + '</td></tr>'; }).join('') + '</tbody></table>';
    }).catch(function (e) {
      out.innerHTML = '<span class="warn-text">Live check failed: ' + esc(errText(e)) + '</span> The curated data above is unchanged.';
    }).finally(function () { btn.disabled = false; });
  }

  /* ------------------------------------------------------- quick-start map */
  var fitBucket = 12;
  function renderFitTabs() {
    var host = $('#fitTabs'); if (!host) return;
    host.innerHTML = [8, 12, 16, 24, 32].map(function (v) { return '<button type="button" class="filter' + (v === fitBucket ? ' active' : '') + '" data-fit="' + v + '" aria-pressed="' + (v === fitBucket) + '">' + v + ' GB VRAM</button>'; }).join('');
  }
  function loadFit() {
    var host = $('#fitGrid'); if (!host) return;
    var body = { gpuVram: fitBucket, systemRam: 32, task: 'general', contextTokens: 32768, runtime: 'llama.cpp', speedPref: 'balanced' };
    host.setAttribute('aria-busy', 'true');
    api('/api/recommend', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(body) }).then(function (r) {
      var byId = {}; r.candidates.forEach(function (c) { byId[c.id] = c; });
      var cards = [['Suggested starting point', r.slots.start], ['Speed-oriented alternative', r.slots.speed], ['Higher-capability option requiring more memory', r.slots.capability]].filter(function (x) { return x[1]; });
      host.innerHTML = cards.map(function (x) { var c = byId[x[1]]; if (!c) return ''; var t = TIER[c.tier]; return '<article class="modelcard fitcard" tabindex="0" role="button" data-model="' + esc(c.id) + '" aria-label="Open model card: ' + esc(c.name) + '"><div class="uk">' + esc(x[0]) + '</div><div class="modelname">' + esc(c.name) + '</div><div class="meta">' + esc(c.variant.quant) + ' · ' + c.variant.sizeGB + ' GB · need ≈ ' + c.memory.totalNeedGiB + ' GiB</div><div class="badges"><span class="tierbadge ' + t[1] + '">' + t[0] + '</span></div><p class="small">' + esc(c.reasons[1] || c.reasons[0]) + '</p></article>'; }).join('') || '<div class="upanel"><p>No entry fits this bucket with these assumptions.</p></div>';
      var n = $('#fitNote'); if (n) n.textContent = 'Computed live by the recommendation engine for ' + fitBucket + ' GB VRAM, 32 GB system RAM, 32K context, llama.cpp-compatible builds, general use. Open the chooser above to change any assumption.';
    }).catch(function (e) { host.innerHTML = '<div class="notice-box warn" role="alert">Could not load starting points: ' + esc(errText(e)) + '</div>'; })
      .finally(function () { host.removeAttribute('aria-busy'); });
  }

  /* --------------------------------------------------- harness / runtimes */
  var FACT_COLS = [['cli', 'CLI'], ['desktop', 'Desktop app'], ['ide', 'IDE'], ['localModels', 'Local models'], ['mcp', 'MCP'], ['subagents', 'Sub-agents'], ['skills', 'Skills'], ['planMode', 'Plan mode'], ['sandbox', 'Sandbox']];
  function factCell(v) {
    if (v === 'yes') return '<td class="yes" title="Documented by the project">✓</td>';
    if (v === 'beta') return '<td class="maybe" title="Documented as beta">beta</td>';
    if (v === 'no') return '<td class="no" title="Not offered">–</td>';
    return '<td class="unv" title="Not confirmed from the sources reviewed; check the project">?</td>';
  }
  function renderHarnesses() {
    var host = $('#harnessTable'); if (!host || !state.catalog) return;
    var hs = state.catalog.harnesses;
    host.innerHTML = '<table><thead><tr><th>Harness</th><th>License</th>' + FACT_COLS.map(function (c) { return '<th>' + esc(c[1]) + '</th>'; }).join('') + '</tr></thead><tbody>' +
      hs.map(function (h) { return '<tr><td><b>' + esc(h.name) + '</b>' + (h.featured ? ' <span class="ub p">featured</span>' : '') + '</td><td>' + esc(h.license) + '</td>' + FACT_COLS.map(function (c) { return factCell(h.facts[c[0]]); }).join('') + '</tr>'; }).join('') + '</tbody></table>';
    var det = $('#harnessDetails');
    if (det) det.innerHTML = hs.map(function (h) {
      return '<details class="upanel hdetail"><summary><b>' + esc(h.name) + '</b> <span class="small">— ' + esc(h.summary) + '</span></summary><ul><li><b>Interfaces:</b> ' + esc(h.details.interfaces) + '</li><li><b>Model providers:</b> ' + esc(h.details.providers) + '</li><li><b>Maturity signals:</b> ' + esc(h.details.maturity) + '</li></ul><div class="chips"><a class="ub" href="' + esc(safeUrl(h.url)) + '" target="_blank" rel="noopener">Project ↗</a><a class="ub" href="' + esc(safeUrl(h.source)) + '" target="_blank" rel="noopener">Source used ↗</a>' + (h.download ? '<a class="ub" href="' + esc(safeUrl(h.download)) + '" target="_blank" rel="noopener">Download ↗</a>' : '') + '</div><div class="utiny">Facts checked ' + esc(h.verifiedOn) + '.</div></details>';
    }).join('');
  }
  function renderRuntimes() {
    var host = $('#runtimeGrid'); if (!host || !state.catalog) return;
    host.innerHTML = state.catalog.runtimes.map(function (r) {
      return '<div class="upanel"><div class="uk">' + esc(r.kind.replace('-', ' ')) + '</div><h3>' + esc(r.name) + '</h3><p>' + esc(r.summary) + '</p><div>' + r.formats.map(function (f) { return '<span class="ub m">' + esc(f) + '</span>'; }).join('') + r.backends.slice(0, 4).map(function (b) { return '<span class="ub">' + esc(b) + '</span>'; }).join('') + '</div><p class="small" style="margin-top:8px">Offload: ' + esc({ 'gpu+cpu': 'GPU + system RAM', gpu: 'GPU memory only', unified: 'unified memory' }[r.offload] || r.offload) + (r.notes ? ' · ' + esc(r.notes) : '') + '</p><a class="ub p" href="' + esc(safeUrl(r.url)) + '" target="_blank" rel="noopener">Project ↗</a></div>';
    }).join('');
  }

  /* ---------------------------------------------------------- search lab */
  var searchPill = 'all';
  var runSearch = debounce(function () {
    var out = $('#uResults'); if (!out) return;
    var q = $('#uSearch').value.trim(), kind = $('#uType').value;
    var url = '/api/catalog/search?kind=' + encodeURIComponent(kind);
    if (q) url += '&q=' + encodeURIComponent(q);
    if (['coding', 'vision', 'agent'].indexOf(searchPill) >= 0) url += '&task=' + searchPill;
    api(url).then(function (d) {
      var rows = d.results.filter(function (x) {
        if (searchPill === 'all' || ['coding', 'vision', 'agent'].indexOf(searchPill) >= 0) return true;
        var m = state.catalog.models.filter(function (mm) { return mm.id === x.id; })[0];
        if (searchPill === 'cloud') return x.kind === 'model' && m && m.kind === 'cloud';
        if (searchPill === 'local') return x.kind !== 'model' || (m && m.kind !== 'cloud');
        return true;
      }).slice(0, 24);
      out.innerHTML = rows.map(function (x) {
        var tags = (x.tags || []).slice(0, 4).map(function (z) { return '<span class="ub">' + esc(z) + '</span>'; }).join('');
        var inner = '<div class="uk">' + esc(x.kind) + '</div><h3>' + esc(x.name) + '</h3><p>' + esc(String(x.summary || '').slice(0, 160)) + '</p><div>' + tags + '</div>';
        return x.kind === 'model' ? '<button type="button" class="upanel resbtn" data-open-model="' + esc(x.id) + '">' + inner + '</button>' : '<div class="upanel">' + inner + '</div>';
      }).join('') || '<div class="upanel"><h3>No match.</h3><p>Try agent, local, coding, vision, MCP or a model name.</p></div>';
    }).catch(function (e) { out.innerHTML = '<div class="notice-box warn" role="alert">Search unavailable: ' + esc(errText(e)) + '</div>'; });
  }, 250);

  /* ------------------------------------------------------ HF live search */
  function inCatalog(id) {
    if (!state.catalog) return null; var l = id.toLowerCase();
    var hit = state.catalog.models.filter(function (m) { return m.links && ((m.links.hf || '').toLowerCase() === l || (m.links.gguf || '').toLowerCase() === l); })[0];
    return hit || null;
  }
  function hfSearch(q) {
    var out = $('#hfSearchResults'); out.textContent = 'Searching Hugging Face…'; out.setAttribute('aria-busy', 'true');
    api('/api/hf/models?search=' + encodeURIComponent(q) + '&limit=10&sort=downloads').then(function (d) {
      var head = '<div class="livehead"><span class="livebadge">LIVE · HUGGING FACE</span> retrieved ' + esc(fmtDate(d.retrievedAt)) + (d.cached ? ' · cached by the server' : '') + (d.stale ? ' · <b>stale copy</b>' : '') + '<br><span class="small">Live results are not curated. Nothing here is verified by the Atlas; check the model card, license and files.</span>' + (d.note ? '<br><span class="small">' + esc(d.note) + '</span>' : '') + '</div>';
      var rows = d.results.map(function (m) {
        var cat = inCatalog(m.id);
        return '<div class="upanel hfrow"><div><a href="' + esc(hfUrl(m.id)) + '" target="_blank" rel="noopener"><b>' + esc(m.id) + '</b> ↗</a> ' + (cat ? '<button type="button" class="ub m linkbtn" data-open-model="' + esc(cat.id) + '">in curated catalogue</button>' : '<span class="ub">not curated</span>') + '<div class="small">' + esc(m.pipelineTag || 'model') + ' · ' + fmtNum(m.downloads) + ' downloads · ' + fmtNum(m.likes) + ' likes · license ' + esc(m.license || 'not stated') + ' · updated ' + esc(m.lastModified ? fmtDate(m.lastModified) : '–') + '</div></div><button type="button" class="btn" data-hf-inspect="' + esc(m.id) + '">Inspect files</button></div><div class="hfdetail" data-for="' + esc(m.id) + '"></div>';
      }).join('');
      out.innerHTML = head + (rows || '<p class="small">No public Hugging Face models matched that search.</p>');
    }).catch(function (e) { out.innerHTML = '<div class="notice-box warn" role="alert">Live Hugging Face search failed: ' + esc(errText(e)) + ' The curated catalogue above is unaffected.</div>'; })
      .finally(function () { out.removeAttribute('aria-busy'); });
  }
  function hfInspect(id, btn) {
    var box = $('.hfdetail[data-for="' + id.replace(/"/g, '') + '"]'); if (!box) return;
    var parts = id.split('/'); btn.disabled = true; box.textContent = 'Loading…';
    api('/api/hf/model/' + encodeURIComponent(parts[0]) + '/' + encodeURIComponent(parts[1])).then(function (d) {
      var m = d.model;
      box.innerHTML = '<div class="small">Retrieved ' + esc(fmtDate(d.retrievedAt)) + (d.stale ? ' (stale copy)' : '') + ' · license ' + esc(m.license || 'not stated') + (m.safetensors && m.safetensors.totalParams ? ' · ' + (m.safetensors.totalParams / 1e9).toFixed(2) + 'B parameters (safetensors)' : '') + (m.gated ? ' · <b>gated</b>' : '') + '</div>' +
        (m.files.length ? '<table class="mini"><tbody>' + m.files.slice(0, 20).map(function (f) { return '<tr><td>' + esc(f.name) + '</td><td class="num">' + (f.sizeBytes / 1e9).toFixed(2) + ' GB</td></tr>'; }).join('') + '</tbody></table>' + (m.files.length > 20 ? '<div class="small">Showing 20 of ' + m.files.length + ' weight files.</div>' : '') : '<div class="small">No weight files listed.</div>');
    }).catch(function (e) { box.innerHTML = '<span class="warn-text">' + esc(errText(e)) + '</span>'; }).finally(function () { btn.disabled = false; });
  }

  /* --------------------------------------------------------------- events */
  function bind() {
    // recommender
    var vb = $('#vramButtons');
    if (vb) vb.addEventListener('click', function (e) { var b = e.target.closest('.vbtn'); if (!b) return; form.vram = Number(b.dataset.vram); $('#vramCustom').value = ''; renderVramButtons(); runRec(); });
    var custom = $('#vramCustom');
    if (custom) custom.addEventListener('input', function () { var n = Number(custom.value); if (n >= 1 && n <= 1024) { form.vram = n; renderVramButtons(); custom.value = n; runRec(); } });
    $$('#memoryKind button').forEach(function (b) { b.addEventListener('click', function () { form.memoryKind = b.dataset.kind; syncMemoryKind(); runRec(); }); });
    ['recRam', 'recRam2', 'recUnified', 'recTask', 'recCtx', 'recRuntime', 'recSpeed', 'recKv', 'recVision', 'recGpuName', 'recCpuName', 'recGpuBw', 'recRamBw'].forEach(function (id) {
      var el = $('#' + id); if (el) el.addEventListener(el.tagName === 'INPUT' && el.type !== 'checkbox' ? 'input' : 'change', runRec);
    });
    // hero + nav shortcut kept from the original design
    // model explorer
    $('#modelFilters') && $('#modelFilters').addEventListener('click', function (e) {
      var b = e.target.closest('.filter'); if (!b) return;
      var kind = b.dataset.kind, tag = b.dataset.tag;
      if (kind) { state.filter.kind = kind; $$('#modelFilters [data-kind]').forEach(function (x) { var on = x === b; x.classList.toggle('active', on); x.setAttribute('aria-pressed', on); }); }
      if (tag) { state.filter.tag = tag; $$('#modelFilters [data-tag]').forEach(function (x) { var on = x === b; x.classList.toggle('active', on); x.setAttribute('aria-pressed', on); }); }
      renderModels();
    });
    var ms = $('#modelSearch'); if (ms) ms.addEventListener('input', debounce(function () { state.filter.q = ms.value.trim(); renderModels(); }, 150));
    // open model from anywhere (cards, buttons)
    document.addEventListener('click', function (e) {
      var t = e.target.closest('[data-model],[data-open-model]');
      if (t) { e.preventDefault(); openModel(t.dataset.model || t.dataset.openModel); return; }
      var lc = e.target.closest('#liveCheck'); if (lc) { runLiveCheck(lc); return; }
      var fb = e.target.closest('[data-fit]'); if (fb) { fitBucket = Number(fb.dataset.fit); renderFitTabs(); loadFit(); return; }
      var ins = e.target.closest('[data-hf-inspect]'); if (ins) { hfInspect(ins.dataset.hfInspect, ins); return; }
      var pill = e.target.closest('#uPills .upill'); if (pill) { $$('#uPills .upill').forEach(function (x) { x.classList.toggle('on', x === pill); x.setAttribute('aria-pressed', x === pill); }); searchPill = pill.dataset.q; runSearch(); }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeModal();
      if ((e.key === 'Enter' || e.key === ' ') && e.target.matches && e.target.matches('.modelcard[data-model]')) { e.preventDefault(); openModel(e.target.dataset.model); }
      if (e.key === 'Tab' && $('#modal').classList.contains('show')) { // simple focus trap
        var f = $$('a[href],button:not([disabled]),input,select,[tabindex="0"]', $('#modal')).filter(function (x) { return x.offsetParent !== null; });
        if (!f.length) return; var first = f[0], last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });
    $('#closeModal').addEventListener('click', closeModal);
    $('#modal').addEventListener('click', function (e) { if (e.target.id === 'modal') closeModal(); });
    // search lab
    var us = $('#uSearch'); if (us) us.addEventListener('input', runSearch);
    var ut = $('#uType'); if (ut) ut.addEventListener('change', runSearch);
    // HF
    var hf = $('#hfSearchForm');
    if (hf) hf.addEventListener('submit', function (e) { e.preventDefault(); var q = $('#hfSearchInput').value.trim(); if (!q) return; hfSearch(q); });
  }

  /* ----------------------------------------------------------------- boot */
  function boot() {
    renderVramButtons(); syncMemoryKind(); renderFitTabs(); bind();
    var results = Promise.all([api('/api/health'), api('/api/catalog')]);
    results.then(function (r) {
      state.catalog = r[1]; setStatus(true, r[0]);
      var n = $('#statModels'); if (n) n.textContent = String(state.catalog.models.length);
      var s = $('#statSnapshot'); if (s) s.textContent = state.catalog.meta.curatedAt;
      renderModels(); renderHarnesses(); renderRuntimes(); runRec.call(null); loadFit(); runSearch();
    }).catch(function () {
      setStatus(false);
      ['#recResults', '#modelGrid', '#fitGrid', '#uResults', '#harnessTable'].forEach(function (s) { var el = $(s); if (el) el.innerHTML = '<div class="notice-box warn" role="alert"><b>Backend unavailable.</b> This section needs the Atlas API. Reload once it is reachable.</div>'; });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
