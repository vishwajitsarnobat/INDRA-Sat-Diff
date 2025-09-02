# climate_forecast/training/prediff_module.py

import os
import warnings
import numpy as np
import torch
from omegaconf import OmegaConf
import lightning.pytorch as pl

from ..taming import AutoencoderKL
from ..models.cuboid_transformer import CuboidTransformerUNet
from ..diffusion.latent_diffusion import LatentDiffusion
from ..diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from ..utils.optim import disable_train
from ..utils.layout import layout_to_in_out_slice
from ..datasets.visualization import visualize_sequence
from ..datasets.evaluation import ClimateSkillScore

class PreDiffModule(LatentDiffusion):
    """
    A PyTorch Lightning Module for training the final Latent Diffusion model (PreDiff).
    """
    def __init__(self, config: OmegaConf):
        self.config = config
        self.save_hyperparameters(self.config)

        # --- Component Initialization from Config ---
        latent_model_cfg = self.hparams.model.latent_model
        vae_cfg = self.hparams.model.vae
        diffusion_cfg = self.hparams.model.diffusion
        align_cfg = self.hparams.model.align

        # The CuboidTransformerUNet is instantiated with its config.
        # The orchestrator is responsible for ensuring `input_shape` is set.
        torch_nn_module = CuboidTransformerUNet(**latent_model_cfg)

        # --- 2. Initialize the pre-trained VAE (frozen) ---
        vae_cfg = self.hparams.model.vae

        # Create a copy and remove non-architectural keys
        vae_model_args = vae_cfg.copy()
        vae_ckpt_path = vae_model_args.pop("pretrained_ckpt_path")
        vae_model_args.pop("loss") # Add this line here as well

        # Now, initialize the model with the cleaned dictionary
        first_stage_model = AutoencoderKL(**vae_model_args)

        if vae_ckpt_path and os.path.exists(vae_ckpt_path):
            state_dict = torch.load(vae_ckpt_path, map_location=torch.device("cpu"))
            first_stage_model.load_state_dict(state_dict=state_dict)
        else:
            raise FileNotFoundError(f"VAE checkpoint not found at {vae_ckpt_path}.")
        
        # --- Call the Superclass Constructor (LatentDiffusion) ---
        super().__init__(
            torch_nn_module=torch_nn_module,
            first_stage_model=first_stage_model,
            **diffusion_cfg
        )

        # The Alignment model is loaded from a checkpoint and frozen.
        alignment_ckpt_path = align_cfg.model_ckpt_path
        if not (alignment_ckpt_path and os.path.exists(alignment_ckpt_path)):
            raise FileNotFoundError(f"Alignment model checkpoint not found at {alignment_ckpt_path}.")
        
        self.alignment_obj = AverageIntensityAlignment(model_ckpt_path=alignment_ckpt_path, **align_cfg)
        disable_train(self.alignment_obj.model)
        self.alignment_model = self.alignment_obj.model
        self.set_alignment(alignment_fn=self.alignment_obj.get_mean_shift)

        # --- Setup Metrics and Logging ---
        self.save_dir = self.hparams.pipeline.stage_output_dir
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)
        
        metric_kwargs = {
            "layout": self.hparams.layout.layout,
            "seq_len": self.hparams.layout.out_len,
            "threshold_list": self.hparams.data.threshold_list,
            "metrics_list": self.hparams.data.metrics_list,
            "denormalize_clip_value": self.hparams.visualization.get('denorm_clip_value', 100.0)
        }
        self.valid_score = ClimateSkillScore(**metric_kwargs)
        self.test_score = ClimateSkillScore(**metric_kwargs)

    @torch.no_grad()
    def get_input(self, batch: torch.Tensor, return_verbose=False):
        layout_cfg = self.hparams.layout
        in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout, in_len=layout_cfg.in_len, out_len=layout_cfg.out_len)
        in_seq = batch[in_slice]
        out_seq = batch[out_slice]
        cond = {"y": in_seq}
        if return_verbose:
            return out_seq, cond, in_seq
        return out_seq, cond

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss, loss_dict = self(batch)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        _, loss_dict_no_ema = self(batch)
        self.log_dict(loss_dict_no_ema, on_step=False, on_epoch=True, logger=True)

        target_seq_norm, cond, context_seq_norm = self.get_input(batch, return_verbose=True)

        alignment_target = AverageIntensityAlignment.calculate_ground_truth_target(target_seq_norm)
        alignment_kwargs = {"avg_x_gt": alignment_target}
        
        pred_seq_norm = self.sample(cond=cond, batch_size=target_seq_norm.shape[0], use_alignment=True, alignment_kwargs=alignment_kwargs).contiguous()

        self.valid_score.update(pred_seq_norm, target_seq_norm)

        if batch_idx == 0:
            save_path = os.path.join(self.example_save_dir, f"epoch_{self.current_epoch}_forecast.png")
            context_np = context_seq_norm[0].cpu().numpy()
            target_np = target_seq_norm[0].cpu().numpy()
            pred_np = pred_seq_norm[0].cpu().numpy()

            visualize_sequence(
                save_path=save_path,
                sequences=[context_np, target_np, pred_np],
                labels=["Input Context", "Ground Truth", "PreDiff Forecast"],
                config=OmegaConf.to_container(self.hparams, resolve=True)
            )

    def on_validation_epoch_end(self):
        score_dict = self.valid_score.compute()
        avg_scores = score_dict.get("avg", {})
        for key, value in avg_scores.items():
            self.log(f"valid_{key}_avg_epoch", value, on_epoch=True, prog_bar=True, logger=True)
        self.valid_score.reset()