# Results: calibrated inference on unseen monsoon data

All numbers below come from `tools/val_study.py` on the cached forecasts of `tools/gen_validation.py`
(results in `artifacts/results/val_study/`). Reproduce:

```bash
.venv/bin/python tools/gen_validation.py south_jul2025 sac_valid_south --year 2022 --every 8
.venv/bin/python tools/gen_validation.py south_jul2025 sac_valid_south --year 2023 --every 6
.venv-steps/bin/python tools/run_pysteps.py south_jul2025 sac_valid_south
.venv/bin/python tools/val_study.py south_jul2025 sac_valid_south --tune-year 2022 --test-year 2023 --out artifacts/results/val_study
```
(same for `ne_jul2025` / `sac_valid_ne`).

## Protocol
- Data: IMERG Final V07, 0.1°, 30 min, 128 × 128 boxes; validation June–September 2022 and 2023, never used
  in training. Sequences as in the SAC evaluation: non-overlapping runs of 7 input + 6 target frames.
- Subset: every 8th sequence of 2022 (56, **tuning**) and every 6th of 2023 (75, **test**) per region.
- Scores: threshold 1.0 mm/h; POD, FAR, CSI and FSS (11/22/44/89 km) per sequence and lead time, then
  averaged (as the SAC script). 95 % CIs: paired bootstrap over sequences (2000 resamples).
- Choices made on 2022 only: KA target (off, 1×, 2×), how members are combined (single, mean, probability
  threshold tuned per lead, probability matching), and — by a rule fixed before looking at 2023 — whether
  probability matching also matches intensities (adopted if CSI at 2.5 and 7.6 mm/h improves on 2022).
- Baselines on the identical sequences: persistence; PySTEPS (STEPS, Lucas–Kanade motion, 50 members,
  as in the SAC script) scored with a probability threshold tuned on 2022 the same way.
- Models: the July 2025 SAC models (South, Northeast), unchanged weights. 4 members, 20 DDIM steps, fp16.

## South India (75 test sequences, JJAS 2023) — selected: KA 1×, probability + intensity matching
| CSI ≥ 1 mm/h | +30 | +60 | +90 | +120 | +150 | +180 |
|---|---|---|---|---|---|---|
| Persistence | 0.552 | 0.420 | 0.346 | 0.286 | 0.245 | 0.213 |
| Single sample, KA 2× (protocol of the paper's evaluation) | 0.521 | 0.378 | 0.300 | 0.244 | 0.200 | 0.173 |
| PySTEPS, 50 members | 0.539 | 0.426 | 0.361 | **0.311** | **0.272** | **0.246** |
| **INDRA-Sat-Diff, calibrated** | **0.576** | **0.445** | **0.369** | 0.307 | 0.259 | 0.225 |
| 95 % CI, ours − persistence | +0.014…+0.035 | +0.013…+0.037 | +0.010…+0.037 | +0.008…+0.033 | +0.002…+0.027 | −0.001…+0.024 |
| 95 % CI, ours − PySTEPS | +0.027…+0.047 | +0.007…+0.032 | −0.007…+0.024 | −0.019…+0.011 | −0.029…+0.004 | −0.036…−0.007 |

Heavier rain (CSI) and amount errors, test year:

| | +30 | +60 | +90 | +120 | +150 | +180 |
|---|---|---|---|---|---|---|
| ≥ 2.5 mm/h ours / persistence / PySTEPS | **0.509** / 0.486 / 0.481 | **0.375** / 0.343 / 0.358 | 0.289 / 0.264 / **0.291** | **0.227** / 0.207 / **0.227** | **0.181** / 0.165 / 0.178 | **0.148** / 0.137 / 0.143 |
| ≥ 7.6 mm/h ours / persistence / PySTEPS | 0.345 / 0.335 / **0.348** | **0.226** / 0.198 / 0.217 | **0.158** / 0.137 / 0.150 | **0.098** / 0.088 / 0.093 | **0.074** / 0.065 / 0.068 | 0.052 / 0.051 / **0.056** |
| MAE mm/h ours (ens. mean) / persistence / PySTEPS | **0.124** / 0.136 / 0.143 | **0.169** / 0.192 / 0.188 | **0.202** / 0.232 / 0.221 | **0.231** / 0.267 / 0.248 | **0.257** / 0.293 / 0.271 | **0.271** / 0.307 / 0.283 |

Intensity matching was adopted on 2022 (mean heavy-rain CSI 0.225 → 0.253); at 1 mm/h it is identical to
area-only probability matching. Brier skill vs persistence at 1 mm/h: ours 0.28–0.33, PySTEPS 0.23–0.33.

## Northeast India (75 test sequences, JJAS 2023) — selected: no KA, probability threshold
| CSI ≥ 1 mm/h | +30 | +60 | +90 | +120 | +150 | +180 |
|---|---|---|---|---|---|---|
| Persistence | **0.608** | 0.452 | 0.362 | 0.299 | 0.256 | 0.224 |
| Single sample, KA 2× | 0.519 | 0.367 | 0.290 | 0.237 | 0.204 | 0.181 |
| PySTEPS, 50 members | 0.603 | **0.466** | **0.385** | **0.330** | **0.295** | **0.265** |
| INDRA-Sat-Diff, calibrated | 0.585 | 0.440 | 0.341 | 0.282 | 0.246 | 0.218 |
| 95 % CI, ours − persistence | −0.030…−0.017 | −0.019…−0.006 | −0.030…−0.014 | −0.024…−0.009 | −0.019…−0.002 | −0.013…+0.002 |

Calibration improves the Northeast model by +0.04 to +0.07 CSI over the single-sample protocol but it remains
below persistence and PySTEPS. Known cause to fix first: the Northeast model uses the South VAE (byte-identical
file), so its encoder never saw Northeast rain. KA made no measurable difference in the Northeast on 2022.

## What was available (and what more would buy)
| | South | Northeast |
|---|---|---|
| Training archive | 35,569 half-hourly frames, 2000–2021 | 84,251 frames, 2000–2021 |
| Training sequences | 15,677 (top 10 % intense + 5 % random) | 21,300 |
| Diffusion training | 50 epochs, batch 16, one shared 32 GB GPU | config up to 100 epochs; actual count not recoverable |
| September 2025 retraining | VAE 48 epochs, alignment 50, diffusion stopped at 3/50 | VAE only (37 epochs) |
| Evaluation here | 56 + 75 sequences of ~450 + 450 | same |
| Inference here | 4 members on a 4 GB laptop GPU | same |

Expected gains with more compute: more ensemble members (PySTEPS used 50, we used 4; its lead appears at
long lead times), a Northeast-trained VAE, finishing the September retraining (`Resume training` in the
app), training on all sequences, and evaluating all ~900 sequences per region for tighter intervals.

## Corrections to the InGARSS 2026 paper
- Fig. 7 (POD curve) is from **Northeast** India at **1.0 mm/h**, not Southern India at 2.5 mm/h.
- The latent is **64 × 16 × 16** per frame (not 4 × 16 × 16).
- The paper's evaluation set the KA target to 2× the mean input intensity; this inflates false alarms. With
  FAR/CSI added, that protocol scores below persistence in both regions (rows "single sample" above).
