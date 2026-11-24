# INDRA-Sat-Diff — consolidation and app plan

## 1. One folder to back up

```
INDRA-Sat-Diff/
  climate_forecast/        framework (git)
  app/                     web app: backend + static frontend (git)
  docs/                    ROADMAP, paper, LaTeX, poster, Q&A (git; PDFs small)
  archive/                 original SAC scripts, older code snapshots (git, read-only reference)
  artifacts/               NOT in git (large): back up this folder with the rest
    models/<name>/         weights (*.pt) + resolved_config.yaml + metrics/*.json + model_card.yaml
    data/<region>/raw|processed/
    results/<run>/         evaluation tables and plots
    jobs/<id>/             logs of app-launched jobs
  .venv/                   recreate with: uv sync (not backed up)
```

Rules: everything a future run needs is either in git or in `artifacts/`; no paths outside the folder;
each model directory is self-describing (config + weights + where/when it was trained + metrics).

## 2. Framework fixes and additions (status)

| Item | Status |
|---|---|
| Batch-size alignment bug, `--input` semantics, EMA export, UTC, FAR/SUCR, year split, `--no-alignment` | done (untested on GPU) |
| `download` command (IMERG via OPeNDAP subsetting, any region/period) | done; blocked on Earthdata password reset |
| `compute_level` 1/2/4/8/16 presets + `fit-check` memory measurement | done; measuring on RTX 2050 |
| Forecast on a different grid than training (resample in/out), `--precision fp16/bf16` | done |
| **DDIM sampler (≈50 steps instead of 1000)** | next — needed for an interactive Predict button |
| Ensemble sampling (N members) → probability-of-exceedance and spread maps | next |
| `evaluate` command: period × lead time × threshold, model vs persistence (vs PySTEPS optional), POD/FAR/CSI/BIAS/FSS, CSV + plots | next; reuse metrics from `datasets/evaluation.py` |
| Model registry (`artifacts/models/*/model_card.yaml`) and loader that rebuilds the module from the card | next |
| Causal KA target (estimate future intensity from context, not from truth) as an option | later; changes reported numbers, label clearly |
| Distillation of a level-1 model into a level-8/16 student (knowledge distillation on the same data) | later; this is the "shrink a trained model" path besides fp16 |
| Tests: preprocessing, split, metrics, shapes for every level (CPU, tiny) | later |

## 3. Web app

### Principles for "efficient"
- **One Python backend process** (FastAPI + uvicorn) that imports `climate_forecast` directly; no second ML stack.
- **Lazy data**: the browser never loads a whole dataset. Frames are fetched per timestep as **uint8-quantized 2-D arrays** (128×128 = 16 KB) and colour-mapped on a `<canvas>` in the browser; server keeps an LRU cache of decoded frames. Timeline index is built once per dataset from file timestamps and cached.
- **Model stays warm**: one model cached on the GPU, swapped only when another model is selected.
- **One GPU, one queue**: predictions, evaluations, fit-checks and training are jobs in a single FIFO worker so they never fight for the 4 GB GPU; progress streams to the browser via Server-Sent Events.
- **No frontend build step**: plain HTML/JS modules, two small vendored libraries (Leaflet for the map/coastlines, uPlot for charts, ~250 KB total), served by the backend. Works offline.

### Pages
1. **Data** — pick region/dataset; timeline with availability bar (gaps shown); play/scrub frames on a map with coastlines and lat/lon grid; hover value in mm/hr; "download period" form (calls `download`, shows progress); preprocess button.
2. **Predict** — pick model + T0 on the timeline; options: KA on/off, ensemble size, steps (DDIM), precision. Shows input → forecast → truth (if available) → difference rows, per-lead POD/FAR/CSI against truth, exceedance-probability map for ensembles. Export PNG/GIF/NPZ.
3. **Simulate (replay)** — choose an event window; the app steps T0 through it and shows rolling forecasts against what actually happened, with a running skill chart; compare two models or KA vs no-KA side by side.
4. **Train** — region + period + split years + compute level; "check memory" runs fit-check and recommends the level; start/resume/stop; live log tail and loss/metric curves from `metrics_history.json`; resulting model is registered automatically.
5. **Models** — registry table (region, grid, level, params, size, trained on, metrics); actions: evaluate, export fp16, delete, set default.
6. **Evaluate** — run model vs persistence over a period; tables and curves (POD/FAR/CSI/FSS vs lead time, per threshold), spatial skill maps; results saved to `artifacts/results/`.

### API sketch
```
GET  /api/datasets                         list processed datasets (region, grid, time span, count)
GET  /api/datasets/{id}/timeline           timestamps + gaps
GET  /api/datasets/{id}/frame?t=ISO        uint8 array (+ scale) — cached
GET  /api/models                           registry
POST /api/jobs  {type: predict|evaluate|train|fitcheck|download|preprocess, params}
GET  /api/jobs/{id}/events                 SSE: progress, logs, result refs
GET  /api/results/{id}/...                 forecast arrays / plots
```

### Build order
1. Consolidate files into this folder (after Drive downloads finish), register the trained South model, verify it loads and forecasts.
2. DDIM + ensembles; measure forecast latency on the RTX 2050.
3. Backend: datasets/frames/timeline + job queue + SSE.
4. Data and Predict pages (the demo-critical part).
5. Evaluate (gives FAR/CSI for the South model too) → Simulate.
6. Train + Models pages, fit-check integration.
