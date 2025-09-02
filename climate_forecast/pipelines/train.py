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
from ..training.prediff_module import PreDiffModule
from ..utils.callbacks import MetricsLoggerCallback

def _extract_and_save_pt_weights(ckpt_path: str, output_pt_path: str, model_key_prefix: str):
    """Loads a PyTorch Lightning checkpoint, extracts the state_dict for the
    core model, and saves it as a clean .pt file for dependency injection."""
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


def _run_stage(pl_module_class, config: DictConfig, dm: ClimateDataModule, stage_name: str):
    """A generic helper function to configure and run a single training stage,
    now with checkpoint resuming capabilities."""
    stage_output_dir = os.path.join(config.pipeline.output_dir, stage_name)
    checkpoints_dir = os.path.join(stage_output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n--- [Stage: {stage_name.upper()}] ---")
    print(f"  > Outputs will be saved to: {stage_output_dir}")

    # --- Dynamic Configuration Injection ---
    OmegaConf.update(config, "pipeline.stage_output_dir", stage_output_dir, merge=True)
    total_steps = (dm.num_train_samples // config.optim.total_batch_size) * config.optim.max_epochs
    OmegaConf.update(config, "pipeline.total_num_steps", total_steps, merge=True)

    # --- Instantiate the Lightning Module ---
    pl_module = pl_module_class(config)

    # --- Configure Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor=config.optim.monitor,
        dirpath=checkpoints_dir,
        filename="best_model-{epoch}",
        save_top_k=1,
        save_last=True, # Important for resuming
        mode="min" if "loss" in config.optim.monitor else "max"
    )
    metrics_callback = MetricsLoggerCallback()

    # --- Configure the Trainer ---
    trainer = Trainer(
        default_root_dir=stage_output_dir,
        accelerator="auto",
        devices="auto",
        max_epochs=config.optim.max_epochs,
        callbacks=[checkpoint_callback, metrics_callback],
        precision=config.trainer.get("precision", 32),
        log_every_n_steps=config.trainer.get("log_every_n_steps", 50)
    )

    # --- Checkpoint Resuming Logic ---
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

    return checkpoint_callback.best_model_path


def run(config: dict):
    """The main entry point for the training pipeline.
    Orchestrates the sequential training of VAE, Alignment, and PreDiff models."""
    cfg = OmegaConf.create(config)

    # --- Initial Setup ---
    output_dir = cfg.pipeline.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Master output directory: {output_dir}")

    print("Initializing DataModule...")
    dm = ClimateDataModule(OmegaConf.to_container(cfg, resolve=True))
    dm.prepare_data()
    dm.setup()

    # === STAGE 1: TRAIN VAE ===
    best_vae_ckpt = _run_stage(VAEModule, cfg, dm, "vae")
    vae_pt_path = os.path.join(output_dir, "vae", "checkpoints", "vae.pt")
    _extract_and_save_pt_weights(best_vae_ckpt, vae_pt_path, "torch_nn_module.")

    # === STAGE 2: TRAIN ALIGNMENT MODEL ===
    print("\nUpdating configuration for Alignment stage...")
    OmegaConf.update(cfg, "model.vae.pretrained_ckpt_path", vae_pt_path, merge=True)

    # --- DYNAMIC SHAPE CALCULATION FOR ALIGNMENT ---
    # The Alignment model also needs to know the shape of the VAE's latent space.
    # We calculate it here and inject it into the config.
    vae_downsample_len = len(cfg.model.vae.block_out_channels)
    downsample_factor = 2 ** vae_downsample_len
    latent_h = cfg.layout.img_height // downsample_factor
    latent_w = cfg.layout.img_width // downsample_factor
    latent_channels = cfg.model.vae.latent_channels

    # The Alignment model processes the output sequence length in the latent space.
    align_input_shape = [cfg.layout.out_len, latent_h, latent_w, latent_channels]
    OmegaConf.update(cfg, "model.align.model_args.input_shape", align_input_shape, merge=True)
    print(f"  > Dynamically calculated Alignment input shape: {align_input_shape}")

    # Now, run the stage with the updated config
    best_alignment_ckpt = _run_stage(AlignmentModule, cfg, dm, "alignment")
    alignment_pt_path = os.path.join(output_dir, "alignment", "checkpoints", "alignment.pt")
    _extract_and_save_pt_weights(best_alignment_ckpt, alignment_pt_path, "torch_nn_module.")

    # === STAGE 3: TRAIN PREDIFF MODEL ===
    print("\nUpdating configuration for PreDiff stage...")
    OmegaConf.update(cfg, "model.vae.pretrained_ckpt_path", vae_pt_path, merge=True)
    OmegaConf.update(cfg, "model.align.model_ckpt_path", alignment_pt_path, merge=True)

    # Dynamically calculate latent shape and inject it
    vae_downsample_len = len(cfg.model.vae.block_out_channels)
    downsample_factor = 2 ** vae_downsample_len
    latent_h = cfg.layout.img_height // downsample_factor
    latent_w = cfg.layout.img_width // downsample_factor
    latent_channels = cfg.model.vae.latent_channels

    input_shape = [cfg.layout.in_len, latent_h, latent_w, latent_channels]
    target_shape = [cfg.layout.out_len, latent_h, latent_w, latent_channels]
    OmegaConf.update(cfg, "model.latent_model.input_shape", input_shape, merge=True)
    OmegaConf.update(cfg, "model.latent_model.target_shape", target_shape, merge=True)
    print(f"  > Dynamically calculated latent shape: H={latent_h}, W={latent_w}")

    best_prediff_ckpt = _run_stage(PreDiffModule, cfg, dm, "prediff")
    prediff_pt_path = os.path.join(output_dir, "prediff", "checkpoints", "prediff_final.pt")
    _extract_and_save_pt_weights(best_prediff_ckpt, prediff_pt_path, "torch_nn_module.")

    print(f"\n--- ✅ Full Training Pipeline Finished Successfully ---")
    print(f"Final model weights saved to: {prediff_pt_path}")