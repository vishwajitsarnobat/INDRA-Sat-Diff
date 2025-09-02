# climate_forecast/training/alignment_module.py

import os
import warnings
import numpy as np
import torch
import torchmetrics
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict

from ..taming import AutoencoderKL
from ..diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from ..utils.layout import layout_to_in_out_slice

class AlignmentModule(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)

        align_cfg = self.hparams.model.align
        vae_cfg = self.hparams.model.vae
        diffusion_cfg = self.hparams.model.diffusion

        self.alignment_obj = AverageIntensityAlignment(**align_cfg)
        self.torch_nn_module = self.alignment_obj.model

        # --- VAE (First Stage Model) Initialization ---
        vae_model_args = vae_cfg.copy()
        vae_ckpt_path = vae_model_args.pop("pretrained_ckpt_path")
        vae_model_args.pop("loss", None) # Safely remove loss key
        self.first_stage_model = AutoencoderKL(**vae_model_args)

        # --- CORRECTLY LOAD THE VAE STATE DICT ---
        if vae_ckpt_path and os.path.exists(vae_ckpt_path):
            print(f"Loading pretrained VAE weights from: {vae_ckpt_path}")
            # The .pt file contains the state_dict directly
            vae_weights = torch.load(vae_ckpt_path, map_location=torch.device("cpu"))
            
            # This is a robust way to check if the weights are what we expect.
            # If the file was a full checkpoint, the keys would be prefixed.
            # Since we extracted them, they should load directly.
            self.first_stage_model.load_state_dict(vae_weights)
            print("Successfully loaded VAE weights into AlignmentModule.")
        else:
            # This should now be a fatal error, as the VAE is required.
            raise FileNotFoundError(
                "VAE checkpoint (.pt file) not found! "
                f"The path specified in the config was: {vae_ckpt_path}"
            )
        
        # Freeze the VAE
        self.first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        # --- Diffusion Noise Schedule Initialization ---
        self.num_timesteps = diffusion_cfg.timesteps
        betas = torch.linspace(diffusion_cfg.linear_start, diffusion_cfg.linear_end, self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        # --- Metrics ---
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()

    def configure_optimizers(self):
        optim_cfg = self.hparams.optim
        optimizer = torch.optim.AdamW(
            self.torch_nn_module.parameters(), 
            lr=optim_cfg.lr, 
            betas=optim_cfg.betas, 
            weight_decay=optim_cfg.wd
        )
        total_steps = self.hparams.pipeline.total_num_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @torch.no_grad()
    def _get_input_and_target(self, batch: torch.Tensor):
        layout_cfg = self.hparams.layout
        in_slice, out_slice = layout_to_in_out_slice(
            layout=layout_cfg.layout, 
            in_len=layout_cfg.in_len, 
            out_len=layout_cfg.out_len
        )
        
        context_seq = batch[tuple(in_slice)].clone()
        target_seq = batch[tuple(out_slice)].clone()
        
        ground_truth_target = AverageIntensityAlignment.calculate_ground_truth_target(target_seq)
        return context_seq, target_seq, ground_truth_target

    def _shared_step(self, batch: torch.Tensor):
        _ , target_seq, ground_truth_target = self._get_input_and_target(batch)

        b, t, h, w, c = target_seq.shape
        target_seq_permuted = target_seq.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        
        with torch.no_grad():
            posterior = self.first_stage_model.encode(target_seq_permuted)
            z_flat = posterior.sample()

        _, latent_c, latent_h, latent_w = z_flat.shape
        z = z_flat.reshape(b, t, latent_c, latent_h, latent_w)

        # --- ROBUST DEVICE HANDLING ---
        model_device = z.device
        
        t_noise = torch.randint(0, self.num_timesteps, (b,), device=model_device).long()
        noise = torch.randn_like(z)
        
        alphas_cumprod_t = self.alphas_cumprod[t_noise]
        
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t).view(b, *([1]*(z.ndim-1)))
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod_t).view(b, *([1]*(z.ndim-1)))
        
        z_noisy = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise

        predicted_target = self.alignment_obj.model_objective(self.torch_nn_module, x_t=z_noisy, t=t_noise)

        loss = torch.nn.functional.mse_loss(
            predicted_target.squeeze(), 
            ground_truth_target.squeeze().to(model_device)
        )
        return loss, predicted_target, ground_truth_target

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        loss, predicted_target, ground_truth_target = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.valid_mse.update(predicted_target.squeeze(), ground_truth_target.squeeze().to(predicted_target.device))
        self.valid_mae.update(predicted_target.squeeze(), ground_truth_target.squeeze().to(predicted_target.device))

    def on_validation_epoch_end(self):
        self.log("valid_mse_epoch", self.valid_mse.compute(), on_epoch=True, prog_bar=True)
        self.log("valid_mae_epoch", self.valid_mae.compute(), on_epoch=True, prog_bar=True)
        self.valid_mse.reset()
        self.valid_mae.reset()