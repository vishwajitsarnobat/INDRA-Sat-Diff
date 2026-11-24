import json
import os
import torch
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from ..datasets.torch_wrapper import ClimateDataModule
from ..training.vae_module import VAEModule
from ..training.alignment_module import AlignmentModule
from ..training.indra_sat_diff_module import IndraSatDiffModule
from ..utils.callbacks import MetricsLoggerCallback


def _extract_and_save_pt_weights(ckpt_path: str, output_pt_path: str, model_key_prefix: str, ema_prefix: str = None):
    """Extract submodule weights from a Lightning checkpoint and save as .pt.

    If `ema_prefix` is given (e.g. "model_ema."), parameters are taken from the LitEma
    shadow buffers instead, matching the weights used for validation/test sampling.
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
        pl_state_dict = checkpoint.get('state_dict', checkpoint)
        model_state_dict = OrderedDict()
        for key, val in pl_state_dict.items():
            if key.startswith(model_key_prefix):
                model_state_dict[key[len(model_key_prefix):]] = val
        if not model_state_dict:
            raise ValueError(f"No weights found with prefix '{model_key_prefix}' in {ckpt_path}")
        if ema_prefix:
            # LitEma stores each trainable parameter under its name with '.' removed
            num_ema = 0
            for name in model_state_dict:
                ema_key = ema_prefix + name.replace('.', '')
                if ema_key in pl_state_dict:
                    model_state_dict[name] = pl_state_dict[ema_key]
                    num_ema += 1
            if num_ema == 0:
                raise ValueError(f"No EMA weights found with prefix '{ema_prefix}' in {ckpt_path}")
            print(f"  > Using EMA weights for {num_ema}/{len(model_state_dict)} tensors.")
        tmp_path = output_pt_path + ".tmp"
        torch.save(model_state_dict, tmp_path)
        os.replace(tmp_path, output_pt_path)        # never leave a half-written weights file
        print(f"  > Extracted model weights to: {output_pt_path}")
    except Exception as e:
        print(f"  > ERROR extracting weights from {ckpt_path}: {e}")
        raise


def _stage_signature(config, keys):
    """JSON of the config sections a stage depends on; a saved stage is reused only if this matches."""
    sig = {k: OmegaConf.to_container(OmegaConf.select(config, k), resolve=True) for k in keys}
    for k in ("pretrained_ckpt_path", "model_ckpt_path"):
        for v in sig.values():
            if isinstance(v, dict):
                v.pop(k, None)
    return json.dumps(sig, sort_keys=True, default=str)


def _stage_done(pt_path, signature):
    sig_path = pt_path + ".config.json"
    return os.path.exists(pt_path) and (not os.path.exists(sig_path) or open(sig_path).read() == signature)


def _mark_done(pt_path, signature):
    open(pt_path + ".config.json", "w").write(signature)


def _run_stage(pl_module_class, config: DictConfig, dm: ClimateDataModule, stage_name: str) -> str:
    """Train a single pipeline stage, returning the path to the best checkpoint."""
    stage_output_dir = os.path.join(config.pipeline.output_dir, stage_name)
    checkpoints_dir = os.path.join(stage_output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n--- [Stage: {stage_name.upper()}] ---")
    print(f"  > Outputs: {stage_output_dir}")
    OmegaConf.update(config, "pipeline.stage_output_dir", stage_output_dir, merge=True)
    total_steps = (dm.num_train_samples // config.optim.total_batch_size) * config.optim.max_epochs
    OmegaConf.update(config, "pipeline.total_num_steps", total_steps, merge=True)

    with open(os.path.join(stage_output_dir, "resolved_config.yaml"), 'w') as f:
        OmegaConf.save(config, f)

    pl_module = pl_module_class(config)

    monitor = config.optim.monitor
    mode = "min" if any(kw in monitor.lower() for kw in ['loss', 'mse', 'mae', 'error']) else "max"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=checkpoints_dir,
        filename="best_model-{epoch}",
        save_top_k=1,
        save_last=True,
        mode=mode,
    )
    metrics_callback = MetricsLoggerCallback()

    trainer_kwargs = {
        "default_root_dir": stage_output_dir,
        "accelerator": "auto",
        "devices": "auto",
        "strategy": "auto",
        "max_epochs": config.optim.max_epochs,
        "callbacks": [checkpoint_callback, metrics_callback],
        "precision": config.trainer.get("precision", "16-mixed"),
        "log_every_n_steps": config.trainer.get("log_every_n_steps", 50),
        "limit_val_batches": config.trainer.get("limit_val_batches", 1.0),
    }

    if pl_module.automatic_optimization:
        if config.optim.get("gradient_clip_val"):
            trainer_kwargs["gradient_clip_val"] = float(config.optim.gradient_clip_val)
        grad_accum_steps = config.optim.total_batch_size // config.optim.micro_batch_size
        if grad_accum_steps > 1:
            trainer_kwargs["accumulate_grad_batches"] = grad_accum_steps
            print(f"  > Gradient accumulation: {grad_accum_steps} steps")
    else:
        print("  > Manual optimization; gradient accumulation disabled.")

    trainer = Trainer(**trainer_kwargs)

    last_ckpt_file = os.path.join(checkpoints_dir, "last.ckpt")
    resume_ckpt_path = last_ckpt_file if os.path.exists(last_ckpt_file) else None
    if resume_ckpt_path:
        print(f"  > Resuming from checkpoint: {resume_ckpt_path}")
    else:
        print("  > Starting training from scratch.")

    print(f"  > Training {stage_name.upper()}...")
    trainer.fit(model=pl_module, datamodule=dm, ckpt_path=resume_ckpt_path)
    print(f"--- [Stage: {stage_name.upper()}] Complete ---")

    # Evaluate on held-out test set
    print(f"  > Running test evaluation...")
    try:
        trainer.test(model=pl_module, datamodule=dm, ckpt_path="best")
    except Exception as e:
        print(f"  > Warning: Test eval failed: {e}")

    if not checkpoint_callback.best_model_path:
        if os.path.exists(last_ckpt_file):
            print(f"  > No best model found; using last checkpoint: {last_ckpt_file}")
            return last_ckpt_file
        raise RuntimeError(f"No best checkpoint for stage {stage_name}.")
    return checkpoint_callback.best_model_path


def run(config: dict):
    """Run the full three-stage training pipeline (VAE → Alignment → Latent Diffusion)."""
    torch.set_float32_matmul_precision('high')
    cfg = OmegaConf.create(config)
    output_dir = cfg.pipeline.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        OmegaConf.save(cfg, f)

    print("Initializing DataModule...")
    dm = ClimateDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    skip_done = cfg.pipeline.get("skip_completed_stages", True)

    # --- Stage 1: VAE ---
    vae_pt_path = os.path.join(output_dir, "vae", "checkpoints", "vae.pt")
    vae_sig = _stage_signature(cfg, ["model.vae", "layout"])
    if skip_done and _stage_done(vae_pt_path, vae_sig):
        print(f"\n--- [Stage: VAE] already trained, using {vae_pt_path} ---")
    else:
        best_vae_ckpt_path = _run_stage(VAEModule, cfg, dm, "vae")
        _extract_and_save_pt_weights(best_vae_ckpt_path, vae_pt_path, "torch_nn_module.")
        _mark_done(vae_pt_path, vae_sig)

    # --- Stage 2: Alignment ---
    print("\nUpdating config for Alignment stage...")
    OmegaConf.update(cfg, "model.vae.pretrained_ckpt_path", vae_pt_path, merge=True)
    img_height, img_width = cfg.layout.img_height, cfg.layout.img_width
    num_downsample_ops = len(cfg.model.vae.down_block_types) - 1
    downsample_factor = 2 ** num_downsample_ops
    latent_h, latent_w = img_height // downsample_factor, img_width // downsample_factor
    latent_channels = cfg.model.vae.latent_channels
    align_input_shape = [cfg.layout.out_len, latent_h, latent_w, latent_channels]
    OmegaConf.update(cfg, "model.align.model_args.input_shape", align_input_shape, merge=True)

    align_pt_path = os.path.join(output_dir, "alignment", "checkpoints", "alignment.pt")
    align_sig = _stage_signature(cfg, ["model.vae", "model.align", "layout"])
    if skip_done and _stage_done(align_pt_path, align_sig):
        print(f"\n--- [Stage: ALIGNMENT] already trained, using {align_pt_path} ---")
    else:
        best_alignment_ckpt_path = _run_stage(AlignmentModule, cfg, dm, "alignment")
        _extract_and_save_pt_weights(best_alignment_ckpt_path, align_pt_path, "torch_nn_module.")
        _mark_done(align_pt_path, align_sig)

    # --- Stage 3: Latent Diffusion (INDRA-Sat-Diff) ---
    print("\nUpdating config for INDRA-Sat-Diff stage...")
    OmegaConf.update(cfg, "model.vae.pretrained_ckpt_path", vae_pt_path, merge=True)
    OmegaConf.update(cfg, "model.align.model_ckpt_path", align_pt_path, merge=True)

    pixel_space_shape = [cfg.layout.out_len, img_height, img_width, cfg.model.vae.in_channels]
    input_shape = [cfg.layout.in_len, latent_h, latent_w, latent_channels]
    target_shape = [cfg.layout.out_len, latent_h, latent_w, latent_channels]

    OmegaConf.update(cfg, "model.diffusion.data_shape", pixel_space_shape, merge=True)
    OmegaConf.update(cfg, "model.latent_model.input_shape", input_shape, merge=True)
    OmegaConf.update(cfg, "model.latent_model.target_shape", target_shape, merge=True)
    OmegaConf.update(cfg, "model.diffusion.latent_shape", target_shape, merge=True)
    print(f"  > Latent shape: ({latent_h}, {latent_w})")

    best_indra_ckpt_path = _run_stage(IndraSatDiffModule, cfg, dm, "indra_sat_diff")

    final_model_path = os.path.join(output_dir, "indra_sat_diff", "checkpoints", "indra_sat_diff_final.pt")
    ema_prefix = "model_ema." if cfg.model.diffusion.get("use_ema", True) else None
    # With EMA, export the EMA weights of the last checkpoint: "best" is chosen on the noisy non-EMA
    # denoising loss, which says little about forecast skill.
    last_ckpt = os.path.join(output_dir, "indra_sat_diff", "checkpoints", "last.ckpt")
    source_ckpt = last_ckpt if ema_prefix and os.path.exists(last_ckpt) else best_indra_ckpt_path
    print(f"\nExtracting final model weights from: {source_ckpt}")
    _extract_and_save_pt_weights(source_ckpt, final_model_path, "torch_nn_module.", ema_prefix=ema_prefix)

    print(f"\n--- ✅ Training Complete ---")
    print(f"Final weights: {final_model_path}")