# climate_forecast/training/alignment_module.py

import os
import torch
import torchmetrics
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..taming import AutoencoderKL
from ..diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from ..utils.layout import layout_to_in_out_slice
from ..diffusion.utils import extract_into_tensor


class AlignmentModule(pl.LightningModule):
    """Train a noisy latent encoder to predict physical properties (e.g., average intensity)."""

    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        align_cfg = self.hparams.model.align
        vae_cfg = self.hparams.model.vae
        diffusion_cfg = self.hparams.model.diffusion

        model_args = OmegaConf.to_container(align_cfg.model_args, resolve=True)
        self.alignment_obj = AverageIntensityAlignment(
            alignment_type=align_cfg.alignment_type,
            guide_scale=align_cfg.guide_scale,
            model_type=align_cfg.model_type,
            model_args=model_args,
            model_ckpt_path=None,
        )
        self.torch_nn_module = self.alignment_obj.model

        # Load frozen VAE for encoding targets into latent space
        vae_args = vae_cfg.copy()
        vae_pt_path = vae_args.pop("pretrained_ckpt_path")
        vae_args.pop("loss", None)
        self.first_stage_model = AutoencoderKL(**vae_args)

        if vae_pt_path and os.path.exists(vae_pt_path):
            self.first_stage_model.load_state_dict(torch.load(vae_pt_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"VAE weights not found at: {vae_pt_path}")

        self.first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        # Diffusion noise schedule for corrupting latent states
        self.num_timesteps = diffusion_cfg.timesteps
        betas = torch.linspace(diffusion_cfg.linear_start, diffusion_cfg.linear_end, self.num_timesteps)
        alphas_cumprod = torch.cumprod(1. - betas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.save_dir = self.hparams.pipeline.stage_output_dir
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()

    def configure_optimizers(self):
        optim_cfg = self.hparams.optim
        opt = torch.optim.AdamW(self.torch_nn_module.parameters(), lr=optim_cfg.lr, betas=tuple(optim_cfg.betas), weight_decay=optim_cfg.wd)
        total_steps = self.hparams.pipeline.total_num_steps
        return {"optimizer": opt, "lr_scheduler": {"scheduler": CosineAnnealingLR(opt, T_max=total_steps), "interval": "step"}}

    def extract_into_tensor(self, a, t, x_shape):
        return extract_into_tensor(a=a, t=t, x_shape=x_shape, batch_axis=0)

    @torch.no_grad()
    def _get_input_and_target(self, batch: torch.Tensor):
        _, out_slice = layout_to_in_out_slice(layout=self.hparams.layout.layout, in_len=self.hparams.layout.in_len, out_len=self.hparams.layout.out_len)
        target_seq = batch[tuple(out_slice)]
        ground_truth_target = AverageIntensityAlignment.model_objective(target_seq)
        return target_seq, ground_truth_target

    def _shared_step(self, batch: torch.Tensor):
        target_seq, ground_truth_target = self._get_input_and_target(batch)
        b, t, h, w, c = target_seq.shape
        target_permuted = target_seq.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)

        with torch.no_grad():
            posterior = self.first_stage_model.encode(target_permuted)
            z_flat = posterior.sample()
        _, latent_c, latent_h, latent_w = z_flat.shape
        z = z_flat.view(b, t, latent_c, latent_h, latent_w)

        t_noise = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        sqrt_alpha_bar = self.extract_into_tensor(self.alphas_cumprod.sqrt(), t_noise, z.shape)
        sqrt_one_minus_alpha_bar = self.extract_into_tensor((1. - self.alphas_cumprod).sqrt(), t_noise, z.shape)

        noise = torch.randn_like(z)
        z_noisy = sqrt_alpha_bar * z + sqrt_one_minus_alpha_bar * noise
        z_noisy = z_noisy.permute(0, 1, 3, 4, 2).contiguous()

        pred = self.torch_nn_module(x=z_noisy, t=t_noise)
        loss = torch.nn.functional.mse_loss(pred.squeeze(), ground_truth_target.squeeze())
        return loss, pred, ground_truth_target

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        loss, pred, target = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.valid_mse.update(pred.squeeze(), target.squeeze())
        self.valid_mae.update(pred.squeeze(), target.squeeze())

    def on_validation_epoch_end(self):
        self.log("val/mse_epoch", self.valid_mse.compute(), on_epoch=True, prog_bar=True)
        self.log("val/mae_epoch", self.valid_mae.compute(), on_epoch=True, prog_bar=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        loss, pred, target = self._shared_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.valid_mse.update(pred.squeeze(), target.squeeze())
        self.valid_mae.update(pred.squeeze(), target.squeeze())

    def on_test_epoch_end(self):
        import json
        mse = self.valid_mse.compute()
        mae = self.valid_mae.compute()
        self.log("test/mse_epoch", mse, on_epoch=True, prog_bar=True)
        self.log("test/mae_epoch", mae, on_epoch=True, prog_bar=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

        test_results = {"test/mse_epoch": mse.item(), "test/mae_epoch": mae.item()}
        test_json_path = os.path.join(self.save_dir, "test_metrics.json")
        with open(test_json_path, 'w') as f:
            json.dump(test_results, f, indent=2)