# INDRA-Sat-Diff — Poster Q&A prep (InGARSS 2026)

## 30-second pitch
Indian monsoon storms grow from nothing to 100 mm/hr in under an hour. Radar coverage is sparse, so satellites are the only gap-free source. We adapted PreDiff, a latent diffusion nowcaster, to NASA IMERG satellite rainfall over Southern and Northeastern India. From 3.5 h of past frames it generates sharp 3-hour forecasts that keep storm cores intact and can show storms growing. Knowledge Alignment is a physics prior applied only at sampling time, and it improves detection over the plain diffusion model at every lead time. The gain widens to about +0.06 POD at 3 h. The design is built to move to ISRO's INSAT-3DS HEM rainfall product.

## Likely questions

**1. Persistence beats your model at +30 min. Why use the model?**
POD only counts hits on observed rain. Persistence copies the current rain field, so at 30 min most rain is still in place and it scores high by construction. It cannot grow, decay or start storms, so its POD falls off fastest. We beat it from about 90 min onward. POD alone also ignores false alarms. FAR/CSI is our first next step, and that is where persistence should look worse.

**2. Why only POD? Where are FAR, CSI and CRPS?**
Honest answer: the conference paper reports POD only because of time and compute limits during the internship. FAR, CSI and CRPS (for the ensemble) are the top future-work item and are already computable in our evaluation code. (Only claim this once the FAR code is merged.)

**3. What exactly does Knowledge Alignment do?**
At each reverse diffusion step, the sampling distribution is multiplied by exp(−λ‖U(z_t) − F₀(y)‖). Here U is the model's predicted clean latent and F₀ is a learned prediction of a physical statistic (e.g. total rain mass) from the context frames. Samples that disagree with the physics estimate are steered back. Weights are not changed, so the constraint can be swapped without retraining.

**4. Is the KA target computed from the future (the true answer)?** ⚠ Be careful here.
The PreDiff evaluation protocol computes the alignment target from the observed future, as an oracle upper bound. In our code, evaluation follows that protocol, and the operational forecast path estimates it from the context. If asked, say plainly: "The reported KA gain follows PreDiff's protocol. A fully causal F₀ estimate is the next ablation, and the gain may shrink." Do not claim the gain is fully operational.

**5. Why IMERG Final? It has about a 3.5-month latency, so it isn't real-time.**
Correct. IMERG Final is used as a clean training and evaluation proxy. Operational use would run on IMERG Early (≈4 h latency) or, better, INSAT-3DS HEM (half-hourly, near real-time). The VAE is meant to be reused as a frozen encoder and only the LDM fine-tuned.

**6. Why latent diffusion instead of pixel-space diffusion or a GAN?**
The latent is 64× smaller, which cuts compute enough to train on one GPU. Diffusion training is more stable than GANs and gives natural ensemble diversity. The cost is about 50 denoising steps per sample, which distillation can reduce to 1–4.

**7. How long does one forecast take?**
Roughly seconds per ensemble member on a 32 GB GPU with ≈50 steps. (Measure this before the talk if you can; otherwise say "seconds, not minutes, and distillation is the fix".)

**8. How did you avoid leakage between train and validation?**
Training years are 2001–2020 and validation is 2022–2023, two completely unseen monsoon seasons, so it is a chronological split.

**9. Why 128×128 at 0.1°?**
That is about 1400 km × 1400 km, which covers a full mesoscale convective system plus its inflow region. It also matches the PreDiff/Earthformer input size.

**10. Why the curriculum (top 10% intense plus 5% random)?**
Most IMERG frames are dry. Oversampling intense sequences teaches convection, and the random 5% keeps the model calibrated on ordinary weather.

**11. What does the spatial POD map show?**
High POD (0.7–0.9) appears where rain is frequent and terrain forces uplift: the Khasi Hills, Brahmaputra valley and Himalayan foothills. Lower POD (0.3–0.5) appears in the drier east. Forecasters can treat it as a reliability map.

**12. How is it probabilistic? Show me uncertainty.**
Each run from a different noise seed gives a different plausible future, and running N seeds gives an ensemble. The poster shows a single member. Ensemble spread and CRPS are future work.

**13. Comparison with PySTEPS / your GAN (INDRA-SatGAN)?**
The qualitative comparison is on the poster (Table I). A quantitative head-to-head on POD and FAR is the next step and is explicitly listed as future work.

**14. What was your role / the team's?**
(Fill in: who did data, VAE, LDM, KA, evaluation, writing.)

## Numbers to remember
- Domains: South 5–17.8°N, 71–83.8°E; Northeast 19.1–31.9°N, 86.1–98.9°E
- 7 input frames (210 min) → 6 output frames (180 min), 30-min steps, 0.1°
- Latent: 4×16×16 per frame (64× compression)
- Train 2001–2020; validation 2022–2023
- POD at 2.5 mm/hr, South: persistence ≈0.74 at +30 min; KA ≈0.70–0.72 at +30 min; KA − noKA ≈ +0.06 at +180 min
- Normalisation: log1p(x)/log1p(100), μ = 0.60, σ = 0.19
- Training: VAE ≈24 h, LDM ≈72 h, one 32 GB GPU
