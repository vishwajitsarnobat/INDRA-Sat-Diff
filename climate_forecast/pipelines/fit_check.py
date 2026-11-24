# climate_forecast/pipelines/fit_check.py
"""Measure peak GPU memory of each training stage at each compute level, using random data.

Runs a few real optimisation steps (same precision, batch size and model as `train`), so the numbers
reflect what training will need. Nothing is written outside a temporary directory.
"""
import copy
import gc
import os
import tempfile

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ..training.alignment_module import AlignmentModule
from ..training.indra_sat_diff_module import IndraSatDiffModule
from ..training.vae_module import VAEModule
from ..utils.config_loader import _deep_merge_dicts, apply_compute_level, VALID_LEVELS


class _RandomSequences(Dataset):
    def __init__(self, n, shape):
        self.n, self.shape = n, shape

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.rand(self.shape)


def set_derived_shapes(cfg):
    """Fill the shape fields that `train.run` derives between stages."""
    h, w = cfg.layout.img_height, cfg.layout.img_width
    f = 2 ** (len(cfg.model.vae.down_block_types) - 1)
    lh, lw, lc = h // f, w // f, cfg.model.vae.latent_channels
    OmegaConf.update(cfg, "model.align.model_args.input_shape", [cfg.layout.out_len, lh, lw, lc], merge=True)
    OmegaConf.update(cfg, "model.diffusion.data_shape", [cfg.layout.out_len, h, w, cfg.model.vae.in_channels], merge=True)
    OmegaConf.update(cfg, "model.latent_model.input_shape", [cfg.layout.in_len, lh, lw, lc], merge=True)
    OmegaConf.update(cfg, "model.latent_model.target_shape", [cfg.layout.out_len, lh, lw, lc], merge=True)
    OmegaConf.update(cfg, "model.diffusion.latent_shape", [cfg.layout.out_len, lh, lw, lc], merge=True)
    return (lh, lw, lc)


def _measure(module_cls, cfg, steps):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    module = module_cls(cfg)
    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    L = cfg.layout
    shape = (L.in_len + L.out_len, L.img_height, L.img_width, cfg.model.vae.in_channels)
    loader = DataLoader(_RandomSequences(steps * cfg.optim.micro_batch_size, shape),
                        batch_size=cfg.optim.micro_batch_size)
    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=1, limit_train_batches=steps,
                      limit_val_batches=0, num_sanity_val_steps=0, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                      precision=cfg.trainer.get("precision", "16-mixed"))
    status = "ok"
    try:
        trainer.fit(module, train_dataloaders=loader)
    except torch.cuda.OutOfMemoryError:
        status = "OOM"
    peak = torch.cuda.max_memory_reserved() / 2**30
    state = {k: v.detach().cpu() for k, v in module.torch_nn_module.state_dict().items()}
    del trainer, module, loader
    gc.collect()
    torch.cuda.empty_cache()
    return status, peak, n_params, state


def run(user_config: dict, default_config: dict, levels=None, steps: int = 3):
    if not torch.cuda.is_available():
        raise RuntimeError("fit-check needs a CUDA GPU.")
    levels = [int(x) for x in (levels or VALID_LEVELS)]
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    name = torch.cuda.get_device_name(0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for level in levels:
            user = copy.deepcopy(user_config)
            user["compute_level"] = level
            merged = _deep_merge_dicts(apply_compute_level(copy.deepcopy(default_config), level), user)
            cfg = OmegaConf.create(merged)
            OmegaConf.update(cfg, "pipeline.stage_output_dir", tmp, merge=True)
            OmegaConf.update(cfg, "pipeline.total_num_steps", 100, merge=True)
            latent = set_derived_shapes(cfg)

            results = {}
            vae_pt, align_pt = os.path.join(tmp, "vae.pt"), os.path.join(tmp, "align.pt")
            for stage, cls in (("vae", VAEModule), ("alignment", AlignmentModule), ("diffusion", IndraSatDiffModule)):
                if any(r[0] == "OOM" for r in results.values()):
                    results[stage] = ("skipped", 0.0, 0)
                    continue
                OmegaConf.update(cfg, "model.vae.pretrained_ckpt_path", vae_pt, merge=True)
                OmegaConf.update(cfg, "model.align.model_ckpt_path", align_pt, merge=True)
                status, peak, n_params, state = _measure(cls, cfg, steps)
                if stage == "vae":
                    torch.save(state, vae_pt)
                elif stage == "alignment":
                    torch.save(state, align_pt)
                results[stage] = (status, peak, n_params)
            rows.append((level, cfg.layout.img_height, cfg.layout.img_width, latent, results))

    print(f"\nGPU: {name}, {total:.1f} GB. Peak reserved memory and trainable parameters per stage:\n")
    print(f"{'level':>5} {'grid':>9} {'latent':>12} | {'VAE':>14} | {'Alignment':>14} | {'Diffusion':>14} | verdict")
    best = None
    for level, h, w, (lh, lw, lc), res in rows:
        cells = []
        fits = True
        for stage in ("vae", "alignment", "diffusion"):
            status, peak, n = res[stage]
            if status != "ok":
                fits = False
                cells.append(f"{status:>14}")
            else:
                fits &= peak < total * 0.95
                cells.append(f"{peak:5.2f} GB {n / 1e6:4.1f}M")
        worst = max((res[s][1] for s in res if res[s][0] == "ok"), default=0.0)
        verdict = ("fits (tight)" if worst > 0.85 * total else "fits") if fits else "does not fit"
        if fits and (best is None or level < best):
            best = level
        print(f"{level:>5} {h:>4}x{w:<4} {lc:>3}x{lh}x{lw:<4} | " + " | ".join(cells) + f" | {verdict}")
    if best:
        print(f"\nRecommended: compute_level: {best} (largest model that fits; choose a higher level "
              f"if a 'tight' fit runs out of memory on real data)")
    else:
        print("\nNothing fits. Use a smaller region or preprocess.downscale_factor: 2 (4x fewer pixels).")
