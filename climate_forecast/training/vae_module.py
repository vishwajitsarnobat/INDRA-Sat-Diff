# climate_forecast/training/vae_module.py

import os
from collections import OrderedDict
import torch
import torchmetrics
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

# Corrected Relative Imports
from ..taming import AutoencoderKL, LPIPSWithDiscriminator
from ..datasets.visualization import visualize_sequence

class VAEModule(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)

        model_cfg = self.hparams.model.vae
        self.torch_nn_module = AutoencoderKL(
            down_block_types=model_cfg.down_block_types,
            in_channels=model_cfg.in_channels,
            block_out_channels=model_cfg.block_out_channels,
            act_fn=model_cfg.act_fn,
            latent_channels=model_cfg.latent_channels,
            up_block_types=model_cfg.up_block_types,
            norm_num_groups=model_cfg.norm_num_groups,
            layers_per_block=model_cfg.layers_per_block,
            out_channels=model_cfg.out_channels,
        )
        self.loss = LPIPSWithDiscriminator(
            disc_start=model_cfg.loss.disc_start,
            kl_weight=model_cfg.loss.kl_weight,
            disc_weight=model_cfg.loss.disc_weight,
            perceptual_weight=model_cfg.loss.perceptual_weight,
            disc_in_channels=model_cfg.loss.disc_in_channels,
        )

        self.automatic_optimization = False
        self.save_dir = self.hparams.pipeline.stage_output_dir
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_last_layer(self):
        return self.torch_nn_module.decoder.conv_out.weight

    def configure_optimizers(self):
        optim_cfg = self.hparams.optim
        lr = optim_cfg.lr
        opt_ae = torch.optim.Adam(self.torch_nn_module.parameters(), lr=lr, betas=optim_cfg.betas)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=optim_cfg.betas)
        total_steps = self.hparams.pipeline.total_num_steps
        scheduler_ae = CosineAnnealingLR(opt_ae, T_max=total_steps)
        scheduler_disc = CosineAnnealingLR(opt_disc, T_max=total_steps)
        return (
            {"optimizer": opt_ae, "lr_scheduler": {"scheduler": scheduler_ae, "interval": "step"}},
            {"optimizer": opt_disc, "lr_scheduler": {"scheduler": scheduler_disc, "interval": "step"}},
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        opt_ae, opt_disc = self.optimizers()
        b, t, h, w, c = batch.shape
        frames = batch.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        reconstructions, posterior = self.torch_nn_module(frames, return_posterior=True)

        aeloss, log_dict_ae = self.loss(frames, reconstructions, posterior, 0, self.trainer.global_step, split="train", last_layer=self.get_last_layer())
        self.manual_backward(aeloss)
        opt_ae.step()
        opt_ae.zero_grad()

        discloss, log_dict_disc = self.loss(frames, reconstructions, posterior, 1, self.trainer.global_step, split="train", last_layer=self.get_last_layer())
        self.manual_backward(discloss)
        opt_disc.step()
        opt_disc.zero_grad()

        self.log_dict(log_dict_ae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log_dict(log_dict_disc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss_step", aeloss + discloss)

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        b, t, h, w, c = batch.shape
        frames = batch.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        reconstructions, posterior = self.torch_nn_module(frames, return_posterior=True)

        aeloss, log_dict_ae = self.loss(frames, reconstructions, posterior, 0, self.trainer.global_step, split="val", last_layer=self.get_last_layer())
        self.log_dict(log_dict_ae, on_step=False, on_epoch=True, logger=True)
        self.log("val_loss", aeloss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.valid_mse.update(reconstructions, frames)
        self.valid_mae.update(reconstructions, frames)

        if batch_idx == 0:
            save_path = os.path.join(self.example_save_dir, f"epoch_{self.current_epoch}_reconstruction.png")
            num_vis = min(frames.shape[0], 8)
            vis_frames = frames[:num_vis].permute(0, 2, 3, 1).cpu().numpy()
            vis_recons = reconstructions[:num_vis].permute(0, 2, 3, 1).cpu().numpy()
            visualize_sequence(save_path=save_path, sequences=[vis_frames, vis_recons], labels=["Ground Truth", "VAE Reconstruction"], config=OmegaConf.to_container(self.hparams, resolve=True))

    def on_validation_epoch_end(self):
        self.log("valid_mse_epoch", self.valid_mse.compute(), on_epoch=True, prog_bar=True)
        self.log("valid_mae_epoch", self.valid_mae.compute(), on_epoch=True, prog_bar=True)
        self.valid_mse.reset()
        self.valid_mae.reset()