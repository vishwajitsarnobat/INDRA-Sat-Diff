# climate_forecast/pipelines/train.py

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

def _extract_and_save_pt_weights(ckpt_path: str, output_pt_path: str, model_key_prefix: str):
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
        pl_state_dict = checkpoint.get('state_dict', checkpoint)
        model_state_dict = OrderedDict()
        for key, val in pl_state_dict.items():
            if key.startswith(model_key_prefix):
                model_state_dict[key.replace(model_key_prefix, "")] = val
        if not model_state_dict:
            raise ValueError(f"Could not find model weights with prefix '{model_key_prefix}' in checkpoint: {ckpt_path}")
        torch.save(model_state_dict, output_pt_path)
        print(f"  > Successfully extracted model weights to: {output_pt_path}")
    except Exception as e:
        print(f"  > ERROR: Failed to extract weights from {ckpt_path}. Error: {e}")
        raise e

def _run_stage(pl_module_class, config: DictConfig, dm: ClimateDataModule, stage_name: str) -> str:
    stage_output_dir = os.path.join(config.pipeline.output_dir, stage_name)
    checkpoints_dir = os.path.join(stage_output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"\n--- [Stage: {stage_name.upper()}] ---")
    print(f"  > Outputs will be saved to: {stage_output_dir}")
    OmegaConf.update(config, "pipeline.stage_output_dir", stage_output_dir, merge=True)
    total_steps = (dm.num_train_samples // config.optim.total_batch_size) * config.optim.max_epochs
    OmegaConf.update(config, "pipeline.total_num_steps", total_steps, merge=True)
    
    # This is the crucial step: The 'config' object passed to this function MUST be complete.
    # This file will then be used as the single source of truth for forecasting.
    with open(os.path.join(stage_output_dir, "resolved_config.yaml"), 'w') as f:
        OmegaConf.save(config, f)
        
    pl_module = pl_module_class(config)
    checkpoint_callback = ModelCheckpoint(
        monitor=config.optim.monitor,
        dirpath=checkpoints_dir,
        filename="best_model-{epoch}",
        save_top_k=1,
        save_last=True,
        mode="min" if "loss" in config.optim.monitor else "max"
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
    }
    if pl_module.automatic_optimization:
        grad_accum_steps = config.optim.total_batch_size // config.optim.micro_batch_size
        if grad_accum_steps > 1:
            trainer_kwargs["accumulate_grad_batches"] = grad_accum_steps
            print(f"  > Automatic optimization detected. Using {grad_accum_steps} gradient accumulation steps.")
    else:
        print("  > Manual optimization detected. Gradient accumulation is disabled for this stage.")
    trainer = Trainer(**trainer_kwargs)
    resume_ckpt_path = None
    last_ckpt_file = os.path.join(checkpoints_dir, "last.ckpt")
    if os.path.exists(last_ckpt_file):
        resume_ckpt_path = last_ckpt_file
        print(f"  > Found last checkpoint: {resume_ckpt_path}. Resuming training.")
    else:
        print("  > No checkpoint found. Starting training from scratch.")
    print(f"  > Starting training for {stage_name.upper()}...")
    trainer.fit(model=pl_module, datamodule=dm, ckpt_path=resume_ckpt_path)
    print(f"--- [Stage: {stage_name.upper()}] Training Complete ---")
    if not checkpoint_callback.best_model_path:
        if os.path.exists(last_ckpt_file):
             print(f"  > No 'best' model path found. Using last checkpoint: {last_ckpt_file}")
             return last_ckpt_file
        raise RuntimeError(f"No best model checkpoint found for stage {stage_name}.")
    return checkpoint_callback.best_model_path

def run(config: dict):
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

    # --- VAE Stage ---
    best_vae_ckpt_path = _run_stage(VAEModule, cfg, dm, "vae")
    vae_pt_path = os.path.join(output_dir, "vae", "checkpoints", "vae.pt")
    _extract_and_save_pt_weights(best_vae_ckpt_path, vae_pt_path, "torch_nn_module.")

    # --- Alignment Stage ---
    print("\nUpdating configuration for Alignment stage...")
    OmegaConf.update(cfg, "model.vae.pretrained_ckpt_path", vae_pt_path, merge=True)
    img_height, img_width = cfg.layout.img_height, cfg.layout.img_width
    num_down_blocks = len(cfg.model.vae.down_block_types)
    downsample_factor = 2 ** num_down_blocks
    latent_h, latent_w = img_height // downsample_factor, img_width // downsample_factor
    latent_channels = cfg.model.vae.latent_channels
    align_input_shape = [cfg.layout.out_len, latent_h, latent_w, latent_channels]
    OmegaConf.update(cfg, "model.align.model_args.input_shape", align_input_shape, merge=True)
    best_alignment_ckpt_path = _run_stage(AlignmentModule, cfg, dm, "alignment")
    align_pt_path = os.path.join(output_dir, "alignment", "checkpoints", "alignment.pt")
    _extract_and_save_pt_weights(best_alignment_ckpt_path, align_pt_path, "torch_nn_module.")

    # ---
    # THE DEFINITIVE FIX:
    # Prepare the configuration for the final stage COMPLETELY before running it.
    # ---
    print("\nUpdating configuration for INDRA-Sat-Diff stage...")
    # 1. Add all artifact paths
    OmegaConf.update(cfg, "model.vae.pretrained_ckpt_path", vae_pt_path, merge=True)
    OmegaConf.update(cfg, "model.align.model_ckpt_path", align_pt_path, merge=True)
    
    # 2. Calculate ALL dynamic shapes required by the final model
    pixel_space_shape = [cfg.layout.out_len, img_height, img_width, cfg.model.vae.in_channels]
    input_shape = [cfg.layout.in_len, latent_h, latent_w, latent_channels]
    target_shape = [cfg.layout.out_len, latent_h, latent_w, latent_channels]
    
    OmegaConf.update(cfg, "model.diffusion.data_shape", pixel_space_shape, merge=True)
    OmegaConf.update(cfg, "model.latent_model.input_shape", input_shape, merge=True)
    OmegaConf.update(cfg, "model.latent_model.target_shape", target_shape, merge=True)
    OmegaConf.update(cfg, "model.diffusion.latent_shape", target_shape, merge=True)
    print(f"  > Final model shapes configured: latent_shape=({latent_h}, {latent_w})")

    # 3. Now, with the config fully prepared, run the final stage.
    #    The _run_stage function will save this completed config as 'resolved_config.yaml'.
    best_indra_ckpt_path = _run_stage(IndraSatDiffModule, cfg, dm, "indra_sat_diff")

    # Final extraction
    final_model_path = os.path.join(output_dir, "indra_sat_diff", "checkpoints", "indra_sat_diff_final.pt")
    print(f"\nExtracting final model weights from: {best_indra_ckpt_path}")
    _extract_and_save_pt_weights(best_indra_ckpt_path, final_model_path, "torch_nn_module.")

    print(f"\n--- ✅ Full Training Pipeline Finished Successfully ---")
    print(f"Final usable model weights saved to: {final_model_path}")