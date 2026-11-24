# Roadmap

Status as of November 2026. Results and protocol: `docs/RESULTS.md`.

## Folder layout (back up the whole folder; only the code part is in git)

```
climate_forecast/   framework: data, models, diffusion + knowledge alignment, training, CLI
app/                web app (FastAPI + build-free frontend)
tools/              validation study, PySTEPS baseline, overnight runner
docs/               results, roadmap, InGARSS 2026 poster (sources, figures, Q&A)
artifacts/          not in git: models, datasets, cached forecasts, results, job logs, original SAC code
private/            not in git: registration, copyright, internship report
.venv, .venv-steps  environments (recreate with uv; .venv-steps is Python 3.11 for PySTEPS)
```

## Done
- Data: IMERG download for any box/period (OPeNDAP subsetting); CF time epochs; netCDF input;
  file-name time source for the SAC data; year-based split.
- Inference: DDIM (20–50 steps), fp16, ensembles, post-processing (probability matching with intensities,
  probability threshold), KA target multiplier; per-model validated settings in each model's config.
- Training: compute levels 1–16 with `fit-check`; stage reuse; resume from Lightning checkpoints
  (full-size diffusion stage fits a 4 GB GPU with EMA in RAM and gradient checkpointing); causal,
  DDIM-based validation during training; correct FSS; EMA per optimizer step.
- Evaluation: SAC protocol reproduced; tuned on 2022, tested on 2023; PySTEPS on the same sequences;
  bootstrap CIs; heavy-rain CSI, Brier, MAE/RMSE.
- App: viewer, predict, replay, evaluate, train, resume, download, models, jobs.

## Next (needs GPU time)
1. **Northeast VAE.** The Northeast model reuses the South VAE; train alignment + diffusion on the
   Northeast VAE already trained in September (`artifacts/models/ne_sep2025_vae_only`).
2. **Finish the September South run** (epoch 3 of 50): Models → Resume training. About 14 h per epoch
   on a 4 GB laptop GPU; 1–2 days on a 24–80 GB GPU (Kaggle, Colab, RunPod, AWS).
3. **More ensemble members** (PySTEPS used 50; we used 4) and evaluation on all ~900 sequences per region.
4. Retrain the diffusion stage with the KA target at 1× (currently only the sampler uses 1×).
5. Fine-tune on INSAT-3DS HEM with the IMERG VAE as a frozen encoder.
6. Distil sampling to 1–4 steps for operational latency.

## Later
- Unit tests for preprocessing, splits, metrics, post-processing and layouts (CPU, tiny tensors).
- Probabilistic scores in training-time validation (CRPS, reliability).
- Extended paper (journal/arXiv) with the corrections listed in `docs/RESULTS.md`.

## Known limitations
- The "combined" June 2025 model's region box is not recorded anywhere; it is listed but not mapped.
- SAC-processed files embed times five days late; datasets from them use `time_source: filename`.
- Random splits over overlapping windows leak between train and validation; use `split.mode: year` for
  any reported number.
- IMERG Final has ~3.5 months latency; operational use needs IMERG Early/Late or INSAT-3DS HEM.
