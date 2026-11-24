# INDRA-Sat-Diff — poster Q&A (InGARSS 2026, POS-0401P, 4 Dec 13:45–15:00)

## 30-second pitch
Monsoon storms in India grow from nothing to 100 mm/h in under an hour, radar is sparse, so satellites are
the only gap-free source. We adapted PreDiff, a latent diffusion nowcaster, to NASA IMERG rainfall over
Southern and Northeast India: from 3.5 h of past frames it samples sharp 3-hour forecasts. The key finding
since the paper: the same trained network, used correctly — knowledge-alignment target at 1× the observed
rain, four samples combined by probability matching — beats persistence at every lead time on unseen 2023
data, and beats a 50-member PySTEPS in the first hour (South).

## Numbers to know (South, JJAS 2023, 75 test sequences, CSI at ≥ 1 mm/h)
| lead | persistence | single sample (old protocol) | PySTEPS 50 members | ours (fixed) |
|---|---|---|---|---|
| +30 | 0.552 | 0.521 | 0.539 | **0.576** |
| +60 | 0.420 | 0.378 | 0.426 | **0.445** |
| +90 | 0.346 | 0.300 | 0.361 | **0.369** |
| +120 | 0.286 | 0.244 | 0.311 | 0.307 |
| +180 | 0.213 | 0.173 | **0.246** | 0.225 |
95 % CI of ours − persistence is above zero up to +150 min (+180: −0.001…+0.024); ours − PySTEPS: better at
+30/60 min, tied +90–150, worse at +180. Heavy rain (≥ 7.6 mm/h): ours beats both at +60–150 min.
MAE/RMSE: lowest of the three at every lead.

Northeast (same protocol): calibrated 0.585 / 0.440 / 0.341 / 0.282 / 0.246 / 0.218 vs persistence
0.608 / 0.452 / 0.362 / 0.299 / 0.256 / 0.224 and PySTEPS 0.603 … 0.265 — better than the old protocol
(+0.04–0.07) but below both baselines. Known reason: the NE model reuses the South VAE.

Data: training 2000–2021 (35.6 k South / 84.3 k NE half-hourly frames; 15.7 k / 21.3 k sequences, top 10 %
most intense + 5 % random); validation JJAS 2022 (tuning) and 2023 (test), never used in training.
Latent 64 × 16 × 16 per frame. One shared 32 GB GPU for training; inference runs on a 4 GB laptop GPU.

## Likely questions

**1. Your paper shows only POD. Why CSI now?**
POD alone rewards over-forecasting. When we added FAR and CSI, the paper's protocol (one sample, KA target
2×) lost to persistence. That told us how the model was being used was wrong, not the network.

**2. What exactly changed? Did you retrain?**
No retraining. (a) KA target = 1× mean input rain instead of 2× (2× told the sampler to make double rain).
(b) Four samples instead of one. (c) Probability matching: the ensemble decides where it rains; the
observed field at T0 gives how much area is wet and how intense (sorted rain rates).

**3. Isn't choosing the best post-processing on your data cheating?**
All choices were made on 2022 only; 2023 was used once, for reporting. PySTEPS got the same treatment
(threshold tuned on 2022) on the identical sequences. The confidence intervals are paired bootstraps over
sequences.

**4b. Why is the Northeast worse?**
Its VAE is literally the South VAE file; the encoder never learned Northeast rain (orography, Brahmaputra
valley regimes). Training a Northeast VAE is the first fix. KA made no difference there on 2022.

**4. Why does PySTEPS win at 3 h?**
Its advection of the observed field holds up at longer leads and it averages 50 members; we used 4.
More members and the unfinished 50-epoch retraining are the obvious next steps.

**5. Probability matching uses the observed rain area at T0 — isn't that persistence?**
Only the *amount* of wet area is carried over; *where* it rains comes from the diffusion ensemble. That is
why it beats persistence at every lead.

**6. Is the KA target causal now?**
Yes. In both the poster results and the training code, the KA target is computed from the input frames only.

**7. Why IMERG Final? It is not real-time.**
Clean training and evaluation proxy. Operational use: IMERG Early/Late (supported by the download tool) or
INSAT-3DS HEM via fine-tuning.

**8. How long does a forecast take?**
4 members × 20 DDIM steps: about 20–40 s on a 4 GB laptop GPU (seconds on a data-centre GPU).

**9. Why only ~75 test sequences per region?**
Compute: every sequence needs several diffusion samples per setting on a laptop GPU. The full set is ~450
sequences per region per year; the code runs it unchanged with more GPU time.

**10. What does the web app do?**
View IMERG for any region, run and compare forecasts, replay events, evaluate against persistence, train
or resume models with settings that fit your GPU — no YAML editing.

**11. What was your role?**
(Fill in.)
