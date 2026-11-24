# INDRA-Sat-Diff

Precipitation nowcasting from satellite rainfall with a latent diffusion model.
Given 7 half-hourly frames (3.5 h), the model samples the next 6 frames (3 h).
It was developed for the Indian monsoon on NASA GPM IMERG data during an internship at the
Space Applications Centre (SAC), ISRO, and is presented at IEEE InGARSS 2026
(paper 1571324784, poster session POS-0401P, 4 December 2026).

The method follows PreDiff (Gao et al., NeurIPS 2023):

1. a VAE compresses each 128 × 128 frame to a 64 × 16 × 16 latent;
2. an Earthformer-style cuboid-attention U-Net learns to denoise the 6-frame latent forecast,
   conditioned on the 7 context latents;
3. Knowledge Alignment (KA) adds the gradient of a small learned network (predicting the
   mean future intensity) to each sampling step, steering samples without retraining.

## Status

Trained models (64-channel latent, 128 × 128, 0.1°): Southern India, Northeast India and a
combined model, all from July 2025; a September 2025 South run whose diffusion stage stopped
at epoch 3 of 50 (resumable). On the 2022–23 validation sets the models are sharp and keep
storm intensity, but over-forecast rain: CSI is below persistence and PySTEPS at all lead
times, and KA raises POD at the cost of more false alarms. Work on this is tracked in
`docs/ROADMAP.md`.

## Install

Requires Python 3.10+ and, for training, an NVIDIA GPU.

```bash
git clone https://github.com/vishwajitsarnobat/INDRA-Sat-Diff
cd INDRA-Sat-Diff
uv venv && uv pip install -e .
uv pip install fastapi "uvicorn[standard]" sse-starlette   # only for the web app
```

## Use

Everything is driven by one YAML file that overrides `climate_forecast/configs/train.yaml`.
Example configs are in `climate_forecast/examples/` (full model, low-end laptop, and
ready-made South / Northeast India regions in `examples/regions/`).

```bash
# data: IMERG half-hourly, cut to your box on the server side (needs an Earthdata login in ~/.netrc)
climate-forecast download   --config my_region.yaml --start 2023-12-01T00:00 --end 2023-12-06T23:30
climate-forecast preprocess --config my_region.yaml

# training: check what fits your GPU, then train all three stages
climate-forecast fit-check --config my_region.yaml
climate-forecast train     --config my_region.yaml

# forecast from the frame at T0 (the last of the 7 inputs)
climate-forecast forecast --config my_region.yaml --input path/to/T0_file.HDF5 \
    --steps 50 --ensemble 8 --precision fp16 [--no-alignment]
```

The web app does the same from a browser (data viewer, forecasts, event replay, evaluation
against persistence, training with live logs, model registry):

```bash
.venv/bin/python -m app        # http://127.0.0.1:8765
```

### Compute levels

`compute_level: 1 | 2 | 4 | 8 | 16` scales the networks so training fits smaller GPUs; values you
set explicitly still win. Peak training memory at 128 × 128 on a 4 GB RTX 2050:

| level | change | peak memory |
|---|---|---|
| 1 | full model (as trained at SAC) | out of memory |
| 2 | gradient checkpointing, micro-batch 1 | out of memory |
| 4 | half-width networks, 32-channel latent | 3.5 GB |
| 8 | quarter width, shallow, 16-channel latent | 1.4 GB |
| 16 | smallest, 8-channel latent | 0.9 GB |

Grid size is a data choice: use a smaller region or `preprocess.downscale_factor`. A model
trained on one grid can forecast on another; inputs and outputs are resampled.

### Data format

One HDF5/netCDF4 file per time step with a 2-D field, latitude, longitude and a time variable.
Variable names, coordinate order and the lat/lon box are set in the config. The time epoch is
read from the variable's CF `units` attribute (IMERG uses seconds since 1980-01-06).

## Repository layout

```
climate_forecast/   package: data, models, diffusion + KA, training stages, pipelines, CLI
app/                web app (FastAPI backend, static frontend, single-GPU job queue)
tools/              research scripts
docs/               roadmap, InGARSS 2026 paper and poster sources, talks, diagrams
archive/            original SAC training/evaluation scripts (reference only)
artifacts/          models, data, results, job logs (not in git)
```

## Citation

V. Sarnobat, V. Yadav, A. Chhari, Y. Vaviya, "INDRA-Sat-Diff: High-Fidelity Generative
Precipitation Nowcasting for the Indian Monsoon Using a Satellite-Adapted Latent Diffusion
Model," IEEE India Geoscience and Remote Sensing Symposium (InGARSS), Hyderabad, 2026.

## Acknowledgements

Dr. Prashant Kumar (SAC, ISRO) for guidance; SAC, ISRO and Sardar Patel Institute of
Technology, Mumbai. Code adapted from PreDiff (Apache 2.0), Earthformer, Hugging Face
Diffusers, Stable Diffusion and OpenAI guided-diffusion.

## License

MIT, except files adapted from the projects above, which keep their original licenses.
