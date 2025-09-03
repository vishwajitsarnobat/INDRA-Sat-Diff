# climate_forecast/training/prediff_module.py

import os
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
    def __init__(self, config: OmegaConf):
        self.config = config
        self.save_hyperparameters(self.config)

        latent_model_cfg = self.hparams.model.latent_model
        vae_cfg = self.hparams.model.vae
        diffusion_cfg = self.hparams.model.diffusion

        torch_nn_module = CuboidTransformerUNet(**OmegaConf.to_container(latent_model_cfg, resolve=True))

        # --- CORRECTED VAE LOADING LOGIC ---
        vae_model_args = vae_cfg.copy()
        vae_pt_path = vae_model_args.pop("pretrained_ckpt_path")
        vae_model_args.pop("loss", None)
        first_stage_model = AutoencoderKL(**vae_model_args)

        if vae_pt_path and os.path.exists(vae_pt_path):
            print(f"PreDiffModule: Loading VAE weights from clean state_dict: {vae_pt_path}")
            vae_weights = torch.load(vae_pt_path, map_location="cpu")
            first_stage_model.load_state_dict(vae_weights)
        else:
            raise FileNotFoundError(f"VAE weights file (.pt) not found at: {vae_pt_path}")

        super().__init__(
            torch_nn_module=torch_nn_module,
            first_stage_model=first_stage_model,
            **OmegaConf.to_container(diffusion_cfg, resolve=True)
        )
        self.learning_rate = self.hparams.optim.lr

        # --- CORRECTED ALIGNMENT LOADING LOGIC ---
        align_cfg = self.hparams.model.align
        alignment_pt_path = align_cfg.model_ckpt_path
        if not (alignment_pt_path and os.path.exists(alignment_pt_path)):
            raise FileNotFoundError(f"Alignment model weights file (.pt) not found at {alignment_pt_path}.")

        print(f"PreDiffModule: Initializing Alignment model and loading weights from: {alignment_pt_path}")
        # Initialize the object, but don't load weights in constructor
        align_cfg_copy = align_cfg.copy()
        align_cfg_copy.model_ckpt_path = None
        self.alignment_obj = AverageIntensityAlignment(**OmegaConf.to_container(align_cfg_copy, resolve=True))
        
        # Load the clean .pt state dict into the model
        align_weights = torch.load(alignment_pt_path, map_location="cpu")
        self.alignment_obj.model.load_state_dict(align_weights)
        # --- END OF FIX ---
        
        disable_train(self.alignment_obj.model)
        self.alignment_model = self.alignment_obj.model
        self.set_alignment(alignment_fn=self.alignment_obj.get_mean_shift)

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
        in_seq = batch[tuple(in_slice)]; out_seq = batch[tuple(out_slice)]
        cond = {"y": in_seq}
        return (out_seq, cond, in_seq) if return_verbose else (out_seq, cond)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss, loss_dict = self(batch)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        _, loss_dict_no_ema = self(batch)
        self.log_dict(loss_dict_no_ema, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        target_seq_norm, cond, context_seq_norm = self.get_input(batch, return_verbose=True)

        alignment_target = AverageIntensityAlignment.model_objective(target_seq_norm)
        alignment_kwargs = {"avg_x_gt": alignment_target}
        
        with self.ema_scope():
            pred_seq_norm = self.sample(cond=cond, batch_size=target_seq_norm.shape[0], use_alignment=True, alignment_kwargs=alignment_kwargs).contiguous()

        self.valid_score.update(pred_seq_norm, target_seq_norm)

        if batch_idx == 0 and self.trainer.is_global_zero:
            save_path = os.path.join(self.example_save_dir, f"epoch_{self.current_epoch}_forecast.png")
            context_np, target_np, pred_np = context_seq_norm[0].cpu().numpy(), target_seq_norm[0].cpu().numpy(), pred_seq_norm[0].cpu().numpy()
            visualize_sequence(save_path=save_path, sequences=[context_np, target_np, pred_np], labels=["Input Context", "Ground Truth", "PreDiff Forecast"], config=OmegaConf.to_container(self.hparams, resolve=True))

    def on_validation_epoch_end(self):
        score_dict = self.valid_score.compute()
        avg_scores = score_dict.get("avg", {})
        for key, value in avg_scores.items(): self.log(f"valid_{key}_avg_epoch", value, on_epoch=True, prog_bar=True, sync_dist=True)
        self.valid_score.reset()