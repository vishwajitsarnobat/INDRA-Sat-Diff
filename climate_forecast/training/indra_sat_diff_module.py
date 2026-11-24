# climate_forecast/training/indra_sat_diff_module.py

import os
import numpy as np
import torch
from omegaconf import OmegaConf, MissingMandatoryValue
import lightning.pytorch as pl

from ..taming import AutoencoderKL
from ..models.cuboid_transformer import CuboidTransformerUNet
from ..diffusion.latent_diffusion import LatentDiffusion
from ..diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from ..utils.optim import disable_train
from ..utils.layout import layout_to_in_out_slice
from ..datasets.visualization import visualize_sequence
from ..datasets.evaluation import ClimateSkillScore

class IndraSatDiffModule(LatentDiffusion):
    """Latent diffusion forecasting model with CuboidTransformer UNet and knowledge alignment guidance."""

    def __init__(self, config: OmegaConf):
        self.config = config
        self.save_hyperparameters(config)

        try:
            latent_model_cfg = self.hparams.model.latent_model
            vae_cfg = self.hparams.model.vae
            diffusion_cfg = self.hparams.model.diffusion
            assert diffusion_cfg.latent_shape is not None
            assert latent_model_cfg.input_shape is not None
            assert latent_model_cfg.target_shape is not None
        except (AttributeError, MissingMandatoryValue, AssertionError) as e:
            raise RuntimeError(f"Config incomplete: {e}")

        torch_nn_module = CuboidTransformerUNet(**OmegaConf.to_container(latent_model_cfg, resolve=True))

        vae_args = vae_cfg.copy()
        vae_pt_path = vae_args.pop("pretrained_ckpt_path")
        vae_args.pop("loss", None)
        first_stage_model = AutoencoderKL(**vae_args)

        if not vae_pt_path or not os.path.exists(vae_pt_path):
            raise FileNotFoundError(f"VAE weights not found: {vae_pt_path}")
        first_stage_model.load_state_dict(torch.load(vae_pt_path, map_location="cpu"))

        super().__init__(
            torch_nn_module=torch_nn_module,
            first_stage_model=first_stage_model,
            **OmegaConf.to_container(diffusion_cfg, resolve=True),
        )
        self.learning_rate = self.hparams.optim.lr
        self.weight_decay = self.hparams.optim.get("wd", 0.01)
        self.betas = tuple(self.hparams.optim.get("betas", (0.9, 0.999)))

        # Load alignment model for guided diffusion sampling
        align_cfg = self.hparams.model.align
        align_pt_path = align_cfg.model_ckpt_path
        if not (align_pt_path and os.path.exists(align_pt_path)):
            raise FileNotFoundError(f"Alignment weights not found: {align_pt_path}")

        align_cfg_copy = align_cfg.copy()
        align_cfg_copy.model_ckpt_path = None
        self.alignment_obj = AverageIntensityAlignment(**OmegaConf.to_container(align_cfg_copy, resolve=True))
        self.alignment_obj.model.load_state_dict(torch.load(align_pt_path, map_location="cpu"))

        disable_train(self.alignment_obj.model)
        self.alignment_model = self.alignment_obj.model
        self.set_alignment(alignment_fn=self.alignment_obj.get_mean_shift)
        self.use_alignment = self.hparams.model.get("use_alignment", True)

        self.save_dir = self.hparams.pipeline.stage_output_dir
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

        metric_kwargs = {
            "config": OmegaConf.to_container(self.config, resolve=True),
            "layout": self.hparams.layout.layout,
            "seq_len": self.hparams.layout.out_len,
            "threshold_list": self.hparams.data.threshold_list,
            "spatial_scales": self.hparams.data.get('spatial_scales', [1, 2, 4, 8]),
            "denormalize_clip_value": self.hparams.visualization.get('denorm_clip_value', 100.0),
        }
        self.valid_score = ClimateSkillScore(**metric_kwargs)
        self.test_score = ClimateSkillScore(**metric_kwargs)

    @torch.no_grad()
    def get_input(self, batch: torch.Tensor, return_verbose=False):
        layout_cfg = self.hparams.layout
        in_slice, out_slice = layout_to_in_out_slice(layout=layout_cfg.layout, in_len=layout_cfg.in_len, out_len=layout_cfg.out_len)
        in_seq = batch[tuple(in_slice)]
        out_seq = batch[tuple(out_slice)]
        cond = {"y": in_seq}
        return (out_seq, cond, in_seq) if return_verbose else (out_seq, cond)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss, loss_dict = self(batch)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _eval_forecast(self, cond, context_seq_norm, batch_size):
        """Forecast for validation/test as it would be made operationally: KA target from the inputs only
        (never the true future), DDIM sampling, and the configured ensemble post-processing."""
        inf = self.hparams.get("inference", {}) or {}
        members = int(inf.get("val_ensemble_size", 1))
        y = cond["y"] if isinstance(cond, dict) else cond
        if members > 1:
            y = y.repeat_interleave(members, dim=0)
        target = AverageIntensityAlignment.model_objective(y) * float(inf.get("ka_multiplier", 1.0))
        pred = self.sample(cond={"y": y} if isinstance(cond, dict) else y, batch_size=y.shape[0],
                           use_alignment=self.use_alignment,
                           alignment_kwargs={"avg_x_gt": target} if self.use_alignment else None,
                           sampler=inf.get("sampler", "ddim"), ddim_steps=int(inf.get("steps", 20))).contiguous()
        if members > 1:
            from ..diffusion.postprocess import combine
            pred = pred.view(batch_size, members, *pred.shape[1:])
            last = context_seq_norm[:, -1, ..., 0].float().cpu().numpy()
            out = [combine(pred[b, ..., 0].float().cpu().numpy(), last[b], inf.get("postprocess", "pm"),
                           float(inf.get("verify_threshold_mm", 1.0)), inf.get("prob_threshold", 0.5))
                   for b in range(batch_size)]
            pred = torch.from_numpy(np.stack(out))[..., None].to(pred.device, pred.dtype)
        return pred

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        _, loss_dict_no_ema = self(batch)
        self.log_dict(loss_dict_no_ema, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        target_seq_norm, cond, context_seq_norm = self.get_input(batch, return_verbose=True)
        with self.ema_scope():
            pred_seq_norm = self._eval_forecast(cond, context_seq_norm, target_seq_norm.shape[0])

        self.valid_score.update(pred_seq_norm, target_seq_norm)

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            save_path = os.path.join(self.example_save_dir, f"epoch_{self.current_epoch}_batch_{batch_idx}.png")
            visualize_sequence(
                save_path=save_path,
                sequences=[context_seq_norm[0].cpu().numpy(), target_seq_norm[0].cpu().numpy(), pred_seq_norm[0].cpu().numpy()],
                labels=["Input Context", "Ground Truth", "Forecast"],
                config=OmegaConf.to_container(self.hparams, resolve=True),
                title=f"INDRA-Sat-Diff | Epoch {self.current_epoch} | Batch {batch_idx}",
            )

    def on_validation_epoch_end(self):
        score_dict = self.valid_score.compute()
        self.log_dict(score_dict, on_epoch=True, prog_bar=False, sync_dist=True)
        self.valid_score.reset()

    @torch.no_grad()
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        target_seq_norm, cond, context_seq_norm = self.get_input(batch, return_verbose=True)
        with self.ema_scope():
            pred_seq_norm = self._eval_forecast(cond, context_seq_norm, target_seq_norm.shape[0])

        self.test_score.update(pred_seq_norm, target_seq_norm)

    def on_test_epoch_end(self):
        import json
        score_dict = self.test_score.compute()
        test_results = {k.replace('val/', 'test/'): v for k, v in score_dict.items()}
        self.log_dict(test_results, on_epoch=True, prog_bar=False, sync_dist=True)
        self.test_score.reset()

        serializable = {k: round(v.item(), 6) if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
        test_json_path = os.path.join(self.save_dir, "test_metrics.json")
        with open(test_json_path, 'w') as f:
            json.dump(serializable, f, indent=2)