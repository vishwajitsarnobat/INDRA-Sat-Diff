# In climate_forecast/training/alignment_module.py

import os
import torch
import torchmetrics
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..taming import AutoencoderKL
from ..diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from ..utils.layout import layout_to_in_out_slice
from ..diffusion.knowledge_alignment.models import NoisyCuboidTransformerEncoder

class AlignmentModule(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)

        align_cfg = self.hparams.model.align
        vae_cfg = self.hparams.model.vae
        diffusion_cfg = self.hparams.model.diffusion

        model_args = OmegaConf.to_container(align_cfg.model_args, resolve=True)
        self.alignment_obj = AverageIntensityAlignment(
            alignment_type=align_cfg.alignment_type,
            guide_scale=align_cfg.guide_scale,
            model_type=align_cfg.model_type,
            model_args=model_args,
            model_ckpt_path=None # We load weights manually below, not in the constructor
        )
        self.torch_nn_module = self.alignment_obj.model

        # --- CORRECTED VAE LOADING LOGIC ---
        vae_model_args_copy = vae_cfg.copy()
        vae_pt_path = vae_model_args_copy.pop("pretrained_ckpt_path")
        vae_model_args_copy.pop("loss", None)
        self.first_stage_model = AutoencoderKL(**vae_model_args_copy)

        if vae_pt_path and os.path.exists(vae_pt_path):
            print(f"AlignmentModule: Loading VAE weights from clean state_dict: {vae_pt_path}")
            vae_weights = torch.load(vae_pt_path, map_location="cpu")
            self.first_stage_model.load_state_dict(vae_weights)
            print("AlignmentModule: Successfully loaded VAE weights.")
        else:
            raise FileNotFoundError(f"VAE weights file (.pt) not found at: {vae_pt_path}")
        # --- END OF FIX ---

        self.first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        self.num_timesteps = diffusion_cfg.timesteps
        betas = torch.linspace(diffusion_cfg.linear_start, diffusion_cfg.linear_end, self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.save_dir = self.hparams.pipeline.stage_output_dir
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()

    def configure_optimizers(self):
        optim_cfg = self.hparams.optim
        optimizer = torch.optim.AdamW(self.torch_nn_module.parameters(), lr=optim_cfg.lr, betas=optim_cfg.betas, weight_decay=optim_cfg.wd)
        total_steps = self.hparams.pipeline.total_num_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @torch.no_grad()
    def _get_input_and_target(self, batch: torch.Tensor):
        layout_cfg = self.hparams.layout
        _, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout, in_len=layout_cfg.in_len, out_len=layout_cfg.out_len)
        target_seq = batch[tuple(out_slice)]
        ground_truth_target = AverageIntensityAlignment.model_objective(target_seq)
        return target_seq, ground_truth_target

    def _shared_step(self, batch: torch.Tensor):
        target_seq, ground_truth_target = self._get_input_and_target(batch)
        b, t, h, w, c = target_seq.shape
        target_seq_permuted = target_seq.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        with torch.no_grad():
            posterior = self.first_stage_model.encode(target_seq_permuted)
            z_flat = posterior.sample()
        _, latent_c, latent_h, latent_w = z_flat.shape
        z = z_flat.view(b, t, latent_c, latent_h, latent_w)
        t_noise = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        noise = torch.randn_like(z)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t_noise]).view(b, *([1]*(z.ndim-1)))
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alphas_cumprod[t_noise]).view(b, *([1]*(z.ndim-1)))
        z_noisy = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise
        z_noisy_for_model = z_noisy.permute(0, 1, 3, 4, 2).contiguous()
        predicted_target = self.torch_nn_module(x=z_noisy_for_model, t=t_noise)
        loss = torch.nn.functional.mse_loss(predicted_target.squeeze(), ground_truth_target.squeeze())
        return loss, predicted_target, ground_truth_target

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        loss, pred, target = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.valid_mse.update(pred.squeeze(), target.squeeze())
        self.valid_mae.update(pred.squeeze(), target.squeeze())

    def on_validation_epoch_end(self):
        self.log("valid_mse_epoch", self.valid_mse.compute(), on_epoch=True, prog_bar=True)
        self.log("valid_mae_epoch", self.valid_mae.compute(), on_epoch=True, prog_bar=True)
        self.valid_mse.reset()
        self.valid_mae.reset()