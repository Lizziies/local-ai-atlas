# Local AI Atlas

A visual, source-linked guide to running AI models locally, with an explainable hardware-aware recommender and a small hardened Express backend. Cyber-girly glassmorphism UI, single page, no build step.

## What it does

- **Recommender** – enter GPU VRAM, system RAM (kept separate, never added together), unified memory, task, context length, runtime and KV-cache precision. Every candidate shows a tier (`FITS COMFORTABLY`, `POSSIBLE WITH OFFLOAD`, `NOT RECOMMENDED`), a memory breakdown, the reasons, warnings and a score breakdown. Wording is deliberately soft: "suggested starting point", never "the best model".
- **Curated catalogue** (`data/catalog.json`) – models, runtimes, harnesses and a taxonomy (model / runtime / harness / agent / MCP / skills / memory / RAG). MoE models list *total* and *active* parameters. Every size or number carries a basis label.
- **Live Hugging Face check** – each model card can compare the curated entry with current Hugging Face metadata. Live data is always labelled with a retrieval timestamp and kept apart from curated data.

## How the estimate works

Need = weights + vision projector (if images are needed) + KV cache + runtime overhead (0.6 GB + 3 % of weights).

| Assumption | Value | Nature |
|---|---|---|
| GPU usable | VRAM − 1 GiB reserve | heuristic |
| RAM usable | RAM − max(4 GiB, 15 %) | heuristic |
| Unified memory usable | 75 % of the pool | heuristic |
| "Fits comfortably" | need ≤ 90 % of usable GPU budget | heuristic |
| Offload allowed | need ≤ GPU + 0.9 × RAM | heuristic |
| KV cache | layers × 2 × KV heads × head dim × 2 bytes (sliding-window layers capped, hybrid/linear-attention layers excluded); some entries use a rough per-token figure | derived / rough |
| KV cache types | f16, q8_0 (34/64), q4_0 (18/64) of f16 size | derived |

Basis labels: **file** = measured file size from the repo, **vendor** = stated by the publisher, **derived** = computed from a config, **rough** = approximation. Speed is shown as a qualitative class; an optional bandwidth-based tokens/s value is a theoretical upper bound only. Benchmarks in the catalogue are publisher-reported, not independently verified. vLLM / SGLang are treated as GPU-resident only; MLX only for unified memory; Bonsai 2 needs PrismML's llama.cpp fork.

The catalogue is a **dated snapshot** (`meta.curatedAt`), not an automatically current list.

## API

| Route | Purpose |
|---|---|
| `GET /api/health` | Render health check (not rate limited) |
| `GET /api/catalog` | Whole curated catalogue |
| `GET /api/catalog/search?q=&kind=&task=` | Search the catalogue |
| `GET /api/catalog/models/:id` | One model |
| `GET /api/catalog/models/:id/live` | Compare a catalogue model with Hugging Face now |
| `GET /api/options` | Allowed recommender inputs |
| `POST /api/recommend` | Recommendation (JSON, ≤ 8 kB) |
| `GET /api/hf/models?search=&…` | Validated Hugging Face search |
| `GET /api/hf/model/:owner/:repo` | Validated Hugging Face model metadata |

Hugging Face hardening: fixed upstream host, path allow-list, strict parameter and id validation, manual redirects only inside `/api/models` on the same origin, 8 s timeout, 2 MB response cap, whitelisted response fields, TTL cache with stale-if-error, in-flight de-duplication, cooldown after upstream 429, per-IP rate limits. `HF_TOKEN` stays on the server. The site is not an open proxy.

Security headers include a strict CSP (`script-src 'self'`, no inline scripts or handlers).

## Local development

```bash
npm install
npm start            # http://localhost:10000  (PORT overrides)
npm test             # node:test, uses a mock Hugging Face upstream
npm run check        # syntax + frontend structure checks
```

Requires Node 18.18+ (developed on Node 22).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `10000` | HTTP port (Render sets this) |
| `HF_TOKEN` | unset | Optional Hugging Face token (server side only) |
| `TRUST_PROXY` | `1` | Express `trust proxy` (correct for one proxy hop such as Render) |

## Deploy on Render

`render.yaml` defines a Node web service: build `npm install`, start `npm start`, health check `/api/health`, binds `0.0.0.0:$PORT`, handles SIGTERM. Create a Web Service from the repository (or a Blueprint) and set `HF_TOKEN` in the dashboard if you want it. The free plan sleeps when idle, so the first request after a pause is slow.

## Maintaining the catalogue

Edit `data/catalog.json`, keep `basis` labels honest, bump `meta.curatedAt`, run `npm test` (the tests validate the schema) and commit. Do not add numbers you cannot source.

## Layout

```
server.js            entry (PORT, graceful shutdown)
src/app.js           routes, headers, static hosting
src/engine.js        recommendation engine
src/hf.js            Hugging Face client
src/catalog.js       load / validate / search
src/ratelimit.js     per-IP limiter
data/catalog.json    curated data
public/              frontend (index.html, css/, js/)
legacy/              the original static snapshot
test/                node:test suites
```
