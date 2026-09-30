// Recommendation engine for Local AI Atlas.
//
// Design rules:
//  * GPU VRAM and system RAM are NEVER added into one number. A model that fits only by
//    spilling into RAM is classified "offload", not "comfortable".
//  * Weights are not the whole footprint: KV cache (from the model's layer config), vision
//    projector and runtime buffers are added on top.
//  * Everything that is a heuristic lives in CONSTANTS and is reported back in `assumptions`.
//  * No quality benchmark numbers are used or invented. Ranking uses fit, documented task
//    match, a parameter-count capacity proxy and quant precision, and every point is explained.

export const GIB = 1024 ** 3;
export const GB = 1e9;

export const CONSTANTS = Object.freeze({
  // Driver / CUDA context / desktop compositor share of a discrete GPU.
  gpuReserveGiB: 1.0,
  // Memory kept back from system RAM for the OS and other programs.
  ramReserveMinGiB: 4,
  ramReserveFraction: 0.15,
  // Share of Apple-style unified memory a GPU workload can typically use by default (adjustable by the user on macOS).
  unifiedUsableFraction: 0.75,
  // Compute buffers / scratch: fixed part + share of weights.
  overheadFixedGB: 0.6,
  overheadWeightFraction: 0.03,
  // Multiplier applied to KV numbers that are rough guesses rather than derived from config.json.
  roughKvSafety: 1.25,
  // "Comfortable" means the GPU-side need is at most this share of the usable budget.
  comfortableShare: 0.9,
  // A model is treated as an MoE for offload purposes when active/total is below this.
  moeRatio: 0.5,
  // Share of the usable system RAM that offloaded weights may occupy (mmap, page cache and other processes need slack).
  ramUsableShare: 0.9
});

export const CONTEXT_OPTIONS = [8192, 16384, 32768, 65536, 131072, 262144];
export const TASKS = ['general', 'coding', 'agent', 'reasoning', 'vision'];
export const SPEED_PREFS = ['speed', 'balanced', 'quality'];
export const MEMORY_KINDS = ['discrete', 'unified', 'none'];
export const KV_CACHE_TYPES = { f16: 1, q8_0: 34 / 64, q4_0: 18 / 64 };
export const RUNTIME_CHOICES = ['any', 'llama.cpp', 'lm-studio', 'ollama', 'vllm', 'sglang', 'mlx', 'prismml-llamacpp'];

export const TIER_LABEL = {
  comfortable: 'Fits comfortably',
  tight: 'Fits, tight headroom',
  offload: 'Possible with offload',
  'not-recommended': 'Not recommended'
};

const NONSTANDARD_RUNTIMES = new Set(['prismml-llamacpp']);
const TIER_ORDER = { comfortable: 3, tight: 2, offload: 1, 'not-recommended': 0 };

/* ------------------------------------------------------------------ input */

function num(v) {
  if (v === '' || v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

/** Validate and normalise user input. Returns { ok, value } or { ok:false, errors }. */
export function normalizeInput(body) {
  const errors = [];
  const b = body && typeof body === 'object' && !Array.isArray(body) ? body : {};
  const memoryKind = MEMORY_KINDS.includes(b.memoryKind) ? b.memoryKind : (b.memoryKind == null ? 'discrete' : null);
  if (memoryKind == null) errors.push(`memoryKind must be one of ${MEMORY_KINDS.join(', ')}`);

  let vramGiB = 0;
  let systemRamGiB = num(b.systemRam);
  let unifiedGiB = 0;

  if (memoryKind === 'discrete') {
    vramGiB = num(b.gpuVram);
    if (vramGiB == null || Number.isNaN(vramGiB) || vramGiB < 1 || vramGiB > 1024) errors.push('gpuVram must be a number between 1 and 1024 (GB)');
    if (systemRamGiB == null || Number.isNaN(systemRamGiB) || systemRamGiB < 4 || systemRamGiB > 4096) errors.push('systemRam must be a number between 4 and 4096 (GB)');
  } else if (memoryKind === 'unified') {
    unifiedGiB = num(b.unifiedMemory ?? b.systemRam);
    if (unifiedGiB == null || Number.isNaN(unifiedGiB) || unifiedGiB < 4 || unifiedGiB > 4096) errors.push('unifiedMemory must be a number between 4 and 4096 (GB)');
    systemRamGiB = unifiedGiB;
  } else if (memoryKind === 'none') {
    if (systemRamGiB == null || Number.isNaN(systemRamGiB) || systemRamGiB < 4 || systemRamGiB > 4096) errors.push('systemRam must be a number between 4 and 4096 (GB)');
  }

  const task = b.task == null ? 'general' : String(b.task);
  if (!TASKS.includes(task)) errors.push(`task must be one of ${TASKS.join(', ')}`);

  let contextTokens = b.contextTokens == null ? 32768 : num(b.contextTokens);
  if (Number.isNaN(contextTokens) || contextTokens < 512 || contextTokens > 1048576) errors.push('contextTokens must be between 512 and 1048576');
  contextTokens = Math.round(contextTokens || 32768);

  const runtime = b.runtime == null ? 'any' : String(b.runtime);
  if (!RUNTIME_CHOICES.includes(runtime)) errors.push(`runtime must be one of ${RUNTIME_CHOICES.join(', ')}`);

  const speedPref = b.speedPref == null ? 'balanced' : String(b.speedPref);
  if (!SPEED_PREFS.includes(speedPref)) errors.push(`speedPref must be one of ${SPEED_PREFS.join(', ')}`);

  const kvCache = b.kvCache == null ? 'f16' : String(b.kvCache);
  if (!(kvCache in KV_CACHE_TYPES)) errors.push(`kvCache must be one of ${Object.keys(KV_CACHE_TYPES).join(', ')}`);

  const needsVision = b.needsVision === true || b.needsVision === 'true' || task === 'vision';

  const gpuName = b.gpuName == null ? '' : String(b.gpuName).trim().slice(0, 80);
  if (gpuName && !/^[\p{L}\p{N} .+\-_/()]+$/u.test(gpuName)) errors.push('gpuName contains unsupported characters');
  const cpuName = b.cpuName == null ? '' : String(b.cpuName).trim().slice(0, 80);
  if (cpuName && !/^[\p{L}\p{N} .+\-_/()]+$/u.test(cpuName)) errors.push('cpuName contains unsupported characters');

  const gpuBandwidthGBs = num(b.gpuBandwidthGBs);
  const ramBandwidthGBs = num(b.ramBandwidthGBs);
  for (const [k, v] of [['gpuBandwidthGBs', gpuBandwidthGBs], ['ramBandwidthGBs', ramBandwidthGBs]]) {
    if (v != null && (Number.isNaN(v) || v <= 0 || v > 20000)) errors.push(`${k} must be a positive number (GB/s)`);
  }

  if (errors.length) return { ok: false, errors };
  return {
    ok: true,
    value: { memoryKind, vramGiB, systemRamGiB, unifiedGiB, task, contextTokens, runtime, speedPref, kvCache, needsVision, gpuName, cpuName, gpuBandwidthGBs, ramBandwidthGBs }
  };
}

/* ------------------------------------------------------------ memory model */

/** Memory budgets in bytes, kept separate for GPU and system RAM. */
export function budgets(input) {
  const c = CONSTANTS;
  if (input.memoryKind === 'discrete') {
    const gpuBytes = Math.max(0, input.vramGiB - c.gpuReserveGiB) * GIB;
    const reserve = Math.max(c.ramReserveMinGiB, c.ramReserveFraction * input.systemRamGiB);
    const ramBytes = Math.max(0, input.systemRamGiB - reserve) * GIB;
    return { kind: 'discrete', gpuBytes, ramBytes, poolBytes: null };
  }
  if (input.memoryKind === 'unified') {
    return { kind: 'unified', gpuBytes: 0, ramBytes: 0, poolBytes: input.unifiedGiB * c.unifiedUsableFraction * GIB };
  }
  const reserve = Math.max(c.ramReserveMinGiB, c.ramReserveFraction * input.systemRamGiB);
  return { kind: 'none', gpuBytes: 0, ramBytes: Math.max(0, input.systemRamGiB - reserve) * GIB, poolBytes: null };
}

/** KV cache size in bytes for `ctx` tokens. */
export function kvBytes(model, ctx, kvCacheType = 'f16') {
  const factor = KV_CACHE_TYPES[kvCacheType] ?? 1;
  const kv = model.kv;
  if (!kv) return { bytes: 0, basis: 'none' };
  if (kv.layers && kv.layers.length) {
    let total = 0;
    for (const l of kv.layers) {
      const tokens = l.w ? Math.min(ctx, l.w) : ctx;
      total += l.n * 2 * l.kv * l.hd * 2 * tokens; // K and V, 2 bytes (f16) per element
    }
    return { bytes: total * factor, basis: kv.basis };
  }
  if (typeof kv.bytesPerToken === 'number') {
    return { bytes: kv.bytesPerToken * ctx * factor * CONSTANTS.roughKvSafety, basis: 'rough' };
  }
  return { bytes: 0, basis: 'none' };
}

export function overheadBytes(weightsBytes) {
  return CONSTANTS.overheadFixedGB * GB + CONSTANTS.overheadWeightFraction * weightsBytes;
}

/* ---------------------------------------------------------------- quantity */

export function quantClass(q) {
  const s = String(q).toUpperCase();
  if (/IQ1|TQ1|PTQ1|1-BIT/.test(s)) return 'very-low';
  if (/IQ2|Q2|PQ2|TERNARY/.test(s)) return 'low';
  if (/IQ3|Q3/.test(s)) return 'reduced';
  if (/IQ4|Q4|MXFP4|NF4/.test(s)) return 'standard';
  if (/Q5|Q6/.test(s)) return 'high';
  if (/Q8|FP8|BF16|F16/.test(s)) return 'full';
  return 'standard';
}
const QUANT_POINTS = { 'very-low': -20, low: -8, reduced: -3, standard: 0, high: 2, full: 3 };
const QUANT_ORDER = { 'very-low': 0, low: 1, reduced: 2, standard: 3, high: 4, full: 5 };

export function isMoE(model) {
  const p = model.params || {};
  return !!(p.totalB && p.activeB && p.activeB / p.totalB < CONSTANTS.moeRatio);
}

/* ------------------------------------------------------------- runtime fit */

function runtimeStatus(variant, runtime, memoryKind, runtimesById) {
  const verified = variant.rt || [];
  const likely = variant.rtLikely || [];
  if (runtime === 'any') {
    if (verified.length) return { ok: true, verified: true, list: verified };
    if (likely.length) return { ok: true, verified: false, list: likely };
    return { ok: false, reason: 'no known runtime for this build' };
  }
  const rt = runtimesById.get(runtime);
  if (runtime === 'mlx' && memoryKind !== 'unified') return { ok: false, reason: 'MLX needs Apple silicon (unified memory)' };
  if (verified.includes(runtime)) return { ok: true, verified: true, list: verified };
  if (likely.includes(runtime)) return { ok: true, verified: false, list: likely };
  const need = rt?.formats?.includes(variant.fmt) ? null : `${rt?.name || runtime} does not load ${variant.fmt.toUpperCase()} files`;
  if (need) return { ok: false, reason: need };
  if (variant.rt?.includes('prismml-llamacpp') && runtime === 'llama.cpp') {
    return { ok: false, reason: 'these files need the PrismML llama.cpp fork; stock llama.cpp rejects the PTQ1_0/PQ2_0 formats' };
  }
  return { ok: false, reason: `${rt?.name || runtime} support for this build is not listed by the model card or vendor` };
}

/* --------------------------------------------------------------- evaluation */

/** Evaluate one variant on the given hardware. */
export function evaluateVariant(model, variant, input, ctx, budget, runtimesById) {
  const weights = variant.gb * GB;
  const mm = input.needsVision && model.modalities.includes('image') ? (variant.mm || 0) * GB : 0;
  const kv = kvBytes(model, ctx, input.kvCache);
  const overhead = overheadBytes(weights);
  const need = weights + mm + kv.bytes + overhead;
  const rtStatus = runtimeStatus(variant, input.runtime, input.memoryKind, runtimesById);
  const gpuOnlyRuntime = input.runtime === 'vllm' || input.runtime === 'sglang';

  let tier = 'not-recommended';
  let mode = 'none';
  let headroomGiB = null;
  let gpuFraction = null;
  const notes = [];

  if (budget.kind === 'discrete') {
    const cap = budget.gpuBytes;
    headroomGiB = (cap - need) / GIB;
    if (need <= cap * CONSTANTS.comfortableShare) { tier = 'comfortable'; mode = 'full-gpu'; }
    else if (need <= cap) { tier = 'tight'; mode = 'full-gpu'; }
    else if (gpuOnlyRuntime) { tier = 'not-recommended'; mode = 'none'; notes.push('vLLM and SGLang expect the weights to live in GPU memory; they are not a good fit for RAM offload.'); }
    else if (cap < 2 * GIB) { // effectively no usable GPU
      if (need <= budget.ramBytes) { tier = 'offload'; mode = 'cpu-only'; } else { tier = 'not-recommended'; }
    } else if (need <= cap + budget.ramBytes * CONSTANTS.ramUsableShare) {
      tier = 'offload';
      mode = isMoE(model) ? 'moe-expert-offload' : 'layer-offload';
      gpuFraction = Math.min(1, cap / need);
    } else { tier = 'not-recommended'; }
  } else if (budget.kind === 'unified') {
    const cap = budget.poolBytes;
    headroomGiB = (cap - need) / GIB;
    if (need <= cap * CONSTANTS.comfortableShare) { tier = 'comfortable'; mode = 'unified'; }
    else if (need <= cap) { tier = 'tight'; mode = 'unified'; }
  } else {
    // no GPU: everything runs on the CPU from system RAM
    headroomGiB = (budget.ramBytes - need) / GIB;
    if (need <= budget.ramBytes) { tier = 'offload'; mode = 'cpu-only'; }
    if (gpuOnlyRuntime) { tier = 'not-recommended'; mode = 'none'; notes.push('vLLM and SGLang need a GPU.'); }
  }

  return { tier, mode, need, weights, mm, kv, overhead, headroomGiB, gpuFraction, rtStatus, notes };
}

function speedFor(model, variant, ev, input) {
  const total = model.params?.totalB, active = model.params?.activeB;
  const activeGB = total && active ? variant.gb * (active / total) : null;
  let tier = 'unknown', note = '';
  if (ev.tier === 'not-recommended') return { tier: 'n/a', note: '', upperBoundTps: null };
  if (ev.mode === 'cpu-only') { tier = 'slow'; note = 'Runs entirely on the CPU from system RAM; expect slow generation, especially for dense models.'; }
  else if (ev.mode === 'full-gpu' || ev.mode === 'unified') {
    tier = active && active <= 8 ? 'fast' : active && active <= 16 ? 'moderate' : 'moderate-slow';
    note = 'Weights are resident in fast memory. Speed scales mainly with ACTIVE parameters and memory bandwidth.';
  } else if (ev.mode === 'moe-expert-offload') {
    tier = 'moderate';
    note = 'MoE: only a few experts are read per token, so keeping attention on the GPU and experts in RAM (llama.cpp --n-cpu-moe) is usually far faster than dense offload. Heuristic, not measured.';
  } else if (ev.mode === 'layer-offload') {
    tier = ev.gpuFraction >= 0.6 ? 'moderate-slow' : 'slow';
    note = 'Dense model with layers in system RAM: every token reads the CPU-side weights over slower memory, so throughput drops sharply compared with a full-GPU fit.';
  }
  let upperBoundTps = null;
  if (activeGB) {
    const kvRead = ev.kv.bytes / GB;
    const bytes = activeGB + kvRead;
    if (ev.mode === 'full-gpu' && input.gpuBandwidthGBs) upperBoundTps = input.gpuBandwidthGBs / bytes;
    else if (ev.mode === 'unified' && input.gpuBandwidthGBs) upperBoundTps = input.gpuBandwidthGBs / bytes;
    else if ((ev.mode === 'layer-offload' || ev.mode === 'moe-expert-offload') && input.gpuBandwidthGBs && input.ramBandwidthGBs) {
      const f = ev.mode === 'layer-offload' ? ev.gpuFraction : 0.25; // expert offload: attention/shared part on GPU (assumed share)
      const t = (f * bytes) / input.gpuBandwidthGBs + ((1 - f) * bytes) / input.ramBandwidthGBs;
      upperBoundTps = 1 / t;
    } else if (ev.mode === 'cpu-only' && input.ramBandwidthGBs) upperBoundTps = input.ramBandwidthGBs / bytes;
  }
  return { tier, note, upperBoundTps: upperBoundTps ? Math.round(upperBoundTps * 10) / 10 : null };
}

/* ------------------------------------------------------------------ scoring */

function scoreCandidate(model, variant, ev, input, ctxInfo) {
  const parts = [];
  const add = (label, points) => parts.push({ label, points: Math.round(points * 10) / 10 });
  const tierBase = { comfortable: 100, tight: 72, offload: ev.mode === 'moe-expert-offload' ? 52 : ev.mode === 'cpu-only' ? 30 : 40, 'not-recommended': 0 }[ev.tier];
  add(`Memory fit: ${TIER_LABEL[ev.tier]}${ev.mode === 'moe-expert-offload' ? ' (MoE expert offload)' : ''}`, tierBase);

  const taskHit = model.tasks.includes(input.task) || (input.task === 'general' && model.tasks.includes('general'));
  if (taskHit) add(`Documented for "${input.task}" tasks`, 25);
  else add(`"${input.task}" is not a documented focus of this model`, -10);
  if (taskHit && model.tasks[0] === input.task) add('Primary documented focus of this model', 12);
  if (input.task === 'agent' && !model.tasks.includes('agent')) add('No documented tool-use / agent training', -10);
  if (ev.rtStatus.list.length && ev.rtStatus.list.every(r => NONSTANDARD_RUNTIMES.has(r))) add('Needs a non-standard runtime build', -12);

  const total = model.params?.totalB || 1;
  const active = model.params?.activeB || total;
  const sizeWeight = input.speedPref === 'quality' ? 6 : input.speedPref === 'speed' ? 1 : 4;
  add('Capacity proxy (parameter count, not a quality measurement)', sizeWeight * Math.log2(Math.max(total, 1)));
  if (input.speedPref === 'speed') add('Speed preference: fewer active parameters', -6 * Math.log2(Math.max(active, 1)) + 20);
  else if (input.speedPref === 'balanced') add('Speed: active parameters', -2 * Math.log2(Math.max(active, 1)) + 6);

  const qc = quantClass(variant.q);
  if (QUANT_POINTS[qc]) add(`Quantisation precision (${qc})`, QUANT_POINTS[qc]);
  if (model.status === 'experimental-preview') add('Experimental preview model', -12);
  if (model.status === 'legacy') add('Superseded generation', -6);
  if (!ev.rtStatus.verified) add('Runtime support not verified for this build', -6);
  if (ev.kv.basis === 'rough') add('KV-cache size is a rough estimate', -3);
  if (ctxInfo.capped) add('Requested context exceeds this model\'s limit', -15);
  return { score: Math.round(parts.reduce((s, p) => s + p.points, 0) * 10) / 10, parts };
}

/* ------------------------------------------------------------ recommendation */

function fmtGiB(bytes) { return Math.round((bytes / GIB) * 10) / 10; }

function explain(model, variant, ev, input, ctxInfo, budget) {
  const reasons = [];
  const warnings = [];
  const gpuTxt = budget.kind === 'discrete' ? `${fmtGiB(budget.gpuBytes)} GiB usable VRAM (of ${input.vramGiB} GB)` : budget.kind === 'unified' ? `${fmtGiB(budget.poolBytes)} GiB usable unified memory (of ${input.unifiedGiB} GB)` : 'no GPU';
  reasons.push(`Needs about ${fmtGiB(ev.need)} GiB in total at ${ctxInfo.effective.toLocaleString('en-US')} tokens of context: weights ${variant.gb} GB (${variant.q}, ${labelBasis(variant.basis)})${ev.mm ? ` + vision projector ${(ev.mm / GB).toFixed(2)} GB` : ''} + KV cache ${(ev.kv.bytes / GB).toFixed(2)} GB (${ev.kv.basis === 'derived' ? 'derived from the model config' : ev.kv.basis === 'derived-approx' ? 'approximated from the model config' : 'rough estimate'}) + runtime buffers ${(ev.overhead / GB).toFixed(2)} GB.`);
  if (ev.tier === 'comfortable') reasons.push(`Fits in ${gpuTxt} with about ${fmtGiB(ev.headroomGiB * GIB)} GiB headroom.`);
  else if (ev.tier === 'tight') { reasons.push(`Fits in ${gpuTxt}, but only ${fmtGiB(ev.headroomGiB * GIB)} GiB is left. A longer context or a bigger prompt batch can tip it over.`); warnings.push('Tight headroom: consider a shorter context, a quantised KV cache, or a smaller quant.'); }
  else if (ev.tier === 'offload') {
    if (ev.mode === 'cpu-only') reasons.push(`No usable GPU memory: the model would run on the CPU from system RAM (${fmtGiB(budget.ramBytes)} GiB usable).`);
    else if (ev.mode === 'moe-expert-offload') reasons.push(`Exceeds ${gpuTxt}. It can run by keeping the always-active part on the GPU and moving expert weights to system RAM (${fmtGiB(budget.ramBytes)} GiB usable). This is offload, not a GPU fit.`);
    else reasons.push(`Exceeds ${gpuTxt}: about ${Math.round((ev.gpuFraction || 0) * 100)}% could stay on the GPU and the rest spills into system RAM (${fmtGiB(budget.ramBytes)} GiB usable). This is offload, not a GPU fit.`);
    warnings.push('Expect noticeably lower speed than a full-GPU fit.');
  }
  else reasons.push(`Does not fit: needs about ${fmtGiB(ev.need)} GiB against ${gpuTxt}${budget.kind === 'discrete' ? ` and ${fmtGiB(budget.ramBytes)} GiB usable system RAM` : ''}.`);
  if (isMoE(model)) reasons.push(`MoE: ${model.params.totalB}B parameters must be stored, ${model.params.activeLabel || model.params.activeB + 'B active'} per token. Storage follows the total, speed follows the active count.`);
  else if (model.params?.totalB && model.params?.activeB && model.params.activeB < model.params.totalB) reasons.push(`Not an MoE: ${model.params.totalB}B parameters (including per-layer embeddings) must be stored, but only ${model.params.activeLabel || model.params.activeB + 'B'} run as the core transformer per token.`);
  if (model.tasks.includes(input.task)) reasons.push(`Documented focus includes ${input.task}.`);
  if (ctxInfo.capped) warnings.push(`You asked for ${ctxInfo.requested.toLocaleString('en-US')} tokens but the model's limit is ${ctxInfo.effective.toLocaleString('en-US')}; the estimate uses the model limit.`);
  if (ev.rtStatus.list.length && ev.rtStatus.list.every(r => NONSTANDARD_RUNTIMES.has(r))) warnings.push('This build only runs on a non-standard runtime (PrismML llama.cpp fork). It will not load in stock llama.cpp, LM Studio, Ollama, vLLM or SGLang.');
  if (!ev.rtStatus.verified) warnings.push('Runtime support for this exact build is not listed by the vendor or model card; verify it with your runtime version.');
  for (const n of ev.notes) warnings.push(n);
  for (const c of model.caveats || []) warnings.push(c);
  return { reasons, warnings };
}

function labelBasis(b) {
  return { file: 'measured file size', vendor: 'vendor-stated', derived: 'derived estimate', rough: 'rough estimate' }[b] || b;
}

/**
 * Produce recommendations.
 * @param {object} catalog parsed data/catalog.json
 * @param {object} rawInput un-validated user input
 */
export function recommend(catalog, rawInput, { limit = 8 } = {}) {
  const norm = normalizeInput(rawInput);
  if (!norm.ok) return { ok: false, errors: norm.errors };
  const input = norm.value;
  const budget = budgets(input);
  const runtimesById = new Map(catalog.runtimes.map(r => [r.id, r]));
  const candidates = [];
  const excluded = [];

  for (const model of catalog.models) {
    if (!model.recommendable || !model.variants.length) continue;
    if (input.needsVision && !model.modalities.includes('image')) {
      excluded.push({ id: model.id, name: model.name, reason: 'no image input' });
      continue;
    }
    const limitCtx = model.contextMax || input.contextTokens;
    const ctxInfo = { requested: input.contextTokens, effective: Math.min(input.contextTokens, limitCtx), capped: input.contextTokens > limitCtx };
    const evals = [];
    let rtReason = null;
    for (const v of model.variants) {
      const ev = evaluateVariant(model, v, input, ctxInfo.effective, budget, runtimesById);
      if (!ev.rtStatus.ok) { rtReason = rtReason || ev.rtStatus.reason; continue; }
      evals.push({ v, ev });
    }
    if (!evals.length) { excluded.push({ id: model.id, name: model.name, reason: rtReason || 'no compatible build' }); continue; }

    const feasible = evals.filter(e => e.ev.tier !== 'not-recommended');
    let pick;
    if (feasible.length) {
      const bestTier = Math.max(...feasible.map(e => TIER_ORDER[e.ev.tier]));
      const pool = feasible.filter(e => TIER_ORDER[e.ev.tier] === bestTier);
      if (bestTier === TIER_ORDER.offload) {
        // prefer the smallest build that is still "standard" precision or better, to limit spilling into RAM
        const good = pool.filter(e => QUANT_ORDER[quantClass(e.v.q)] >= QUANT_ORDER.standard);
        const list = (good.length ? good : pool).sort((a, b) => a.v.gb - b.v.gb);
        pick = good.length ? list[0] : list[list.length - 1];
      } else {
        pick = pool.sort((a, b) => QUANT_ORDER[quantClass(b.v.q)] - QUANT_ORDER[quantClass(a.v.q)] || b.v.gb - a.v.gb)[0];
        // Do not jump to a bigger quant if a smaller "standard" build already fits comfortably and the bigger one is tight.
      }
    } else {
      pick = evals.sort((a, b) => a.v.gb - b.v.gb)[0];
    }
    const { v, ev } = pick;
    const speed = speedFor(model, v, ev, input);
    const sc = scoreCandidate(model, v, ev, input, ctxInfo);
    const { reasons, warnings } = explain(model, v, ev, input, ctxInfo, budget);
    candidates.push({
      id: model.id,
      name: model.name,
      vendor: model.vendor,
      status: model.status,
      params: model.params,
      arch: model.arch,
      license: model.license,
      modalities: model.modalities,
      tasks: model.tasks,
      taskMatch: model.tasks.includes(input.task),
      variant: { quant: v.q, format: v.fmt, sizeGB: v.gb, sizeBasis: v.basis, repo: v.repo || null },
      otherVariants: evals.filter(e => e !== pick).map(e => ({ quant: e.v.q, sizeGB: e.v.gb, tier: e.ev.tier })),
      tier: ev.tier,
      tierLabel: TIER_LABEL[ev.tier],
      mode: ev.mode,
      memory: {
        weightsGB: round(ev.weights / GB), visionProjectorGB: round(ev.mm / GB), kvCacheGB: round(ev.kv.bytes / GB), kvBasis: ev.kv.basis, runtimeOverheadGB: round(ev.overhead / GB),
        totalNeedGiB: round(ev.need / GIB), headroomGiB: ev.headroomGiB == null ? null : round(ev.headroomGiB), contextUsed: ctxInfo.effective, contextCapped: ctxInfo.capped
      },
      speed,
      runtimes: { verified: ev.rtStatus.verified, list: ev.rtStatus.list },
      score: sc.score,
      scoreBreakdown: sc.parts,
      reasons,
      warnings,
      links: model.links
    });
  }

  candidates.sort((a, b) => b.score - a.score);
  const usable = candidates.filter(c => c.tier !== 'not-recommended');
  const slots = pickSlots(usable, input);
  // A slot must never point at a candidate that is not in the returned list.
  if (slots.capability && slots.capability === slots.speed) slots.capability = null;
  const top = usable.slice(0, limit);
  for (const id of [slots.start, slots.speed, slots.capability]) {
    if (id && !top.some(c => c.id === id)) top.push(usable.find(c => c.id === id));
  }
  const notRecommended = candidates.filter(c => c.tier === 'not-recommended').sort((a, b) => a.memory.totalNeedGiB - b.memory.totalNeedGiB).slice(0, 8)
    .map(c => ({ id: c.id, name: c.name, quant: c.variant.quant, totalNeedGiB: c.memory.totalNeedGiB, reason: c.reasons[c.reasons.length - 1], warnings: c.warnings.filter(w => /vLLM|SGLang|GPU/.test(w)) }));
  return {
    ok: true,
    input: { ...input },
    assumptions: {
      units: 'VRAM/RAM entered in GB are treated as GiB (2^30 bytes); file sizes are decimal GB (10^9 bytes).',
      gpuReserveGiB: CONSTANTS.gpuReserveGiB,
      usableVramGiB: budget.kind === 'discrete' ? round(budget.gpuBytes / GIB) : null,
      usableSystemRamGiB: budget.kind !== 'unified' ? round(budget.ramBytes / GIB) : null,
      usableUnifiedGiB: budget.kind === 'unified' ? round(budget.poolBytes / GIB) : null,
      unifiedUsableFraction: budget.kind === 'unified' ? CONSTANTS.unifiedUsableFraction : null,
      comfortableShare: CONSTANTS.comfortableShare,
      overhead: `${CONSTANTS.overheadFixedGB} GB + ${CONSTANTS.overheadWeightFraction * 100}% of weights (heuristic)`,
      kvCache: input.kvCache,
      note: 'GPU VRAM and system RAM are evaluated separately and never summed into one "fits" number. All budgets and overheads are heuristics, not measurements.'
    },
    slots,
    candidates: top,
    notRecommended,
    totalEvaluated: candidates.length,
    excluded,
    disclaimer: 'Starting points, not guarantees. Actual memory use and speed depend on runtime version, backend, batch size and prompt. The ranking uses fit, documented task focus and a parameter-count capacity proxy; it is not a benchmark.'
  };
}

function round(x) { return Math.round(x * 100) / 100; }

function pickSlots(list, input) {
  const fitting = list.filter(c => c.tier === 'comfortable' || c.tier === 'tight');
  const start = fitting[0] || list.find(c => c.tier === 'offload') || null;
  const out = { start: start?.id || null, speed: null, capability: null };
  if (!start) return out;
  const activeOf = c => c.params?.activeB || c.params?.totalB || 1;
  const totalOf = c => c.params?.totalB || 1;
  const speedPool = fitting.filter(c => c.id !== start.id && activeOf(c) < activeOf(start) * 0.75 && (c.taskMatch || !start.taskMatch));
  if (speedPool.length) out.speed = speedPool.sort((a, b) => activeOf(a) - activeOf(b) || b.score - a.score)[0].id;
  const capPool = list.filter(c => c.id !== start.id && (c.tier === 'offload' || c.tier === 'tight') && totalOf(c) > totalOf(start) * 1.15 && (c.taskMatch || !start.taskMatch));
  if (capPool.length) out.capability = capPool.sort((a, b) => b.score - a.score)[0].id;
  return out;
}
