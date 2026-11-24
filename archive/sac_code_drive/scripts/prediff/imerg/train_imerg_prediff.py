import warnings
from typing import Sequence, Union, Dict, Optional
from shutil import copyfile
import inspect
from collections import OrderedDict
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torchmetrics
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import (
    Callback, LearningRateMonitor, DeviceStatsMonitor,
    EarlyStopping, ModelCheckpoint,
)
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf, DictConfig
import os
import argparse
from einops import rearrange

from src.prediff.datasets.imerg.imerg_torch_wrap import IMERGLightningDataModule
from src.prediff.datasets.imerg.visualization import vis_imerg_seq
from src.prediff.datasets.imerg.evaluation import IMERGSkillScore

from src.prediff.utils.pl_checkpoint import pl_load
from src.prediff.utils.optim import warmup_lambda, disable_train
from src.prediff.utils.layout import layout_to_in_out_slice
from src.prediff.utils.path import (
    default_exps_dir,
    default_pretrained_vae_dir,
    default_pretrained_earthformerunet_dir,
    default_pretrained_alignment_dir,
)

DEFAULT_CLIP_VALUE = 100.0

def _denormalize_log_np(data: np.ndarray, clip_value: float = DEFAULT_CLIP_VALUE) -> np.ndarray:
    """
    Denormalizes log-transformed numpy data back to the physical scale.
    Assumes normalization was roughly: log1p(x) / log1p(clip_value)
    """
    if clip_value <= 0:
        warnings.warn(f"Clip value ({clip_value}) non-positive during denormalization. Returning original data.")
        return data.astype(np.float32)

    data = data.astype(np.float32) # Ensure float32 for calculations
    scale_factor = np.log1p(clip_value) # log(1 + clip_value)

    if abs(scale_factor) < 1e-9: # Avoid division by zero or near-zero
        warnings.warn(f"Scale factor is near zero ({scale_factor}). Potential issues in denormalization.")
        return data

    try:
        # Inverse operation: exp(data * scale_factor) - 1
        denormalized = np.expm1(data * scale_factor)
    except OverflowError:
        warnings.warn("Overflow encountered during np.expm1 denormalization. Output will be clipped.")
        large_val_indicator = np.where(data * scale_factor > 700, np.inf, 0) # approx log(realmax)
        denormalized = np.expm1(np.clip(data * scale_factor, -np.inf, 700)) # Clip input to expm1
        denormalized = np.where(large_val_indicator == np.inf, np.inf, denormalized)

    denormalized = np.nan_to_num(denormalized, nan=0.0, posinf=clip_value, neginf=0.0)
    denormalized = np.clip(denormalized, 0.0, clip_value)
    return denormalized.astype(np.float32)


from src.prediff.taming import AutoencoderKL
from src.prediff.models.cuboid_transformer import CuboidTransformerUNet
from src.prediff.diffusion.latent_diffusion import LatentDiffusion
from src.prediff.diffusion.knowledge_alignment.imerg import IMERGAvgIntensityAlignment

pytorch_state_dict_name = "imerg_earthformerunet.pt"


def get_alignment_kwargs_avg_x(context_seq=None, target_seq=None):
    multiplier = 2.0
    batch_size = target_seq.shape[0]
    ret = torch.mean(target_seq.view(batch_size, -1), dim=1, keepdim=True) * multiplier
    return {"avg_x_gt": ret}


class PreDiffIMERGPLModule(LatentDiffusion):
    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        if oc_file is None:
            raise ValueError("A config file must be provided.")
        oc = OmegaConf.load(open(oc_file, "r"))
        # Store config accessibility via self.hparams
        self.save_hyperparameters(oc)
        # Keep a direct reference as well if preferred
        self.oc: DictConfig = self.hparams # Use hparams directly

        latent_model_cfg = OmegaConf.to_object(self.oc.model.latent_model)
        num_blocks = len(latent_model_cfg["depth"])
        if isinstance(latent_model_cfg["self_pattern"], str):
            block_attn_patterns = [latent_model_cfg["self_pattern"]] * num_blocks
        else:
            block_attn_patterns = OmegaConf.to_container(latent_model_cfg["self_pattern"])
        latent_model = CuboidTransformerUNet(
            input_shape=latent_model_cfg["input_shape"],
            target_shape=latent_model_cfg["target_shape"],
            base_units=latent_model_cfg["base_units"],
            scale_alpha=latent_model_cfg["scale_alpha"],
            num_heads=latent_model_cfg["num_heads"],
            attn_drop=latent_model_cfg["attn_drop"],
            proj_drop=latent_model_cfg["proj_drop"],
            ffn_drop=latent_model_cfg["ffn_drop"],
            downsample=latent_model_cfg["downsample"],
            downsample_type=latent_model_cfg["downsample_type"],
            upsample_type=latent_model_cfg["upsample_type"],
            upsample_kernel_size=latent_model_cfg["upsample_kernel_size"],
            depth=latent_model_cfg["depth"],
            block_attn_patterns=block_attn_patterns,
            num_global_vectors=latent_model_cfg["num_global_vectors"],
            use_global_vector_ffn=latent_model_cfg["use_global_vector_ffn"],
            use_global_self_attn=latent_model_cfg["use_global_self_attn"],
            separate_global_qkv=latent_model_cfg["separate_global_qkv"],
            global_dim_ratio=latent_model_cfg["global_dim_ratio"],
            ffn_activation=latent_model_cfg["ffn_activation"],
            gated_ffn=latent_model_cfg["gated_ffn"],
            norm_layer=latent_model_cfg["norm_layer"],
            padding_type=latent_model_cfg["padding_type"],
            checkpoint_level=latent_model_cfg["checkpoint_level"],
            pos_embed_type=latent_model_cfg["pos_embed_type"],
            use_relative_pos=latent_model_cfg["use_relative_pos"],
            self_attn_use_final_proj=latent_model_cfg["self_attn_use_final_proj"],
            attn_linear_init_mode=latent_model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=latent_model_cfg["ffn_linear_init_mode"],
            ffn2_linear_init_mode=latent_model_cfg["ffn2_linear_init_mode"],
            attn_proj_linear_init_mode=latent_model_cfg["attn_proj_linear_init_mode"],
            conv_init_mode=latent_model_cfg["conv_init_mode"],
            down_linear_init_mode=latent_model_cfg["down_up_linear_init_mode"],
            up_linear_init_mode=latent_model_cfg["down_up_linear_init_mode"],
            global_proj_linear_init_mode=latent_model_cfg["global_proj_linear_init_mode"],
            norm_init_mode=latent_model_cfg["norm_init_mode"],
            time_embed_channels_mult=latent_model_cfg["time_embed_channels_mult"],
            time_embed_use_scale_shift_norm=latent_model_cfg["time_embed_use_scale_shift_norm"],
            time_embed_dropout=latent_model_cfg["time_embed_dropout"],
            unet_res_connect=latent_model_cfg["unet_res_connect"],
        )

        vae_cfg = OmegaConf.to_object(self.oc.model.vae)
        first_stage_model = AutoencoderKL(
            down_block_types=vae_cfg["down_block_types"],
            in_channels=vae_cfg["in_channels"],
            block_out_channels=vae_cfg["block_out_channels"],
            act_fn=vae_cfg["act_fn"],
            latent_channels=vae_cfg["latent_channels"],
            up_block_types=vae_cfg["up_block_types"],
            norm_num_groups=vae_cfg["norm_num_groups"],
            layers_per_block=vae_cfg["layers_per_block"],
            out_channels=vae_cfg["out_channels"],
        )
        pretrained_ckpt_path = vae_cfg["pretrained_ckpt_path"]
        if pretrained_ckpt_path is not None:
            # Load VAE weights to CPU first
            state_dict = torch.load(pretrained_ckpt_path, map_location=torch.device("cpu"))
            first_stage_model.load_state_dict(state_dict=state_dict)
        else:
            warnings.warn("Pretrained weights for `AutoencoderKL` not set. Run for sanity check only.")

        diffusion_cfg = OmegaConf.to_object(self.oc.model.diffusion)
        super(PreDiffIMERGPLModule, self).__init__(
            torch_nn_module=latent_model,
            layout=self.oc.layout.layout,
            data_shape=diffusion_cfg["data_shape"],
            timesteps=diffusion_cfg["timesteps"],
            beta_schedule=diffusion_cfg["beta_schedule"],
            loss_type=self.oc.optim.loss_type,
            monitor=self.oc.optim.monitor,
            use_ema=diffusion_cfg["use_ema"],
            log_every_t=diffusion_cfg["log_every_t"],
            clip_denoised=diffusion_cfg["clip_denoised"],
            linear_start=diffusion_cfg["linear_start"],
            linear_end=diffusion_cfg["linear_end"],
            cosine_s=diffusion_cfg["cosine_s"],
            given_betas=diffusion_cfg["given_betas"],
            original_elbo_weight=diffusion_cfg["original_elbo_weight"],
            v_posterior=diffusion_cfg["v_posterior"],
            l_simple_weight=diffusion_cfg["l_simple_weight"],
            parameterization=diffusion_cfg["parameterization"],
            learn_logvar=diffusion_cfg["learn_logvar"],
            logvar_init=diffusion_cfg["logvar_init"],
            latent_shape=diffusion_cfg["latent_shape"],
            first_stage_model=first_stage_model,
            cond_stage_model=diffusion_cfg["cond_stage_model"],
            num_timesteps_cond=diffusion_cfg["num_timesteps_cond"],
            cond_stage_trainable=diffusion_cfg["cond_stage_trainable"],
            cond_stage_forward=diffusion_cfg["cond_stage_forward"],
            scale_by_std=diffusion_cfg["scale_by_std"],
            scale_factor=diffusion_cfg["scale_factor"],
        )
        # knowledge alignment
        knowledge_alignment_cfg = OmegaConf.to_object(self.oc.model.align)
        self.alignment_type = knowledge_alignment_cfg["alignment_type"]
        self.use_alignment = self.alignment_type is not None
        if self.use_alignment:
            alignment_ckpt_path = knowledge_alignment_cfg["model_ckpt_path"]
            self.alignment_obj = IMERGAvgIntensityAlignment(
                alignment_type=knowledge_alignment_cfg["alignment_type"],
                guide_scale=knowledge_alignment_cfg["guide_scale"],
                model_type=knowledge_alignment_cfg["model_type"],
                model_args=knowledge_alignment_cfg["model_args"],
                model_ckpt_path=alignment_ckpt_path,
            )
            disable_train(self.alignment_obj.model)
            self.alignment_model = self.alignment_obj.model
            alignment_fn = self.alignment_obj.get_mean_shift
        else:
            alignment_fn = None
        self.set_alignment(alignment_fn=alignment_fn)

        # lr_scheduler and logging setup
        self.total_num_steps = total_num_steps
        self.save_dir = save_dir
        self.logging_prefix = self.oc.logging.logging_prefix
        self.train_example_data_idx_list = list(self.oc.eval.train_example_data_idx_list)
        self.val_example_data_idx_list = list(self.oc.eval.val_example_data_idx_list)
        self.test_example_data_idx_list = list(self.oc.eval.test_example_data_idx_list)
        self.eval_example_only = self.oc.eval.eval_example_only

        # Get denormalization clip value from config, fallback to default
        self.denorm_clip_value = self.oc.dataset.get("denormalize_clip_value", DEFAULT_CLIP_VALUE)
        # Get SSIM data range, default to physical max after denorm
        self.ssim_data_range = self.oc.dataset.get("ssim_data_range", self.denorm_clip_value)

        metric_kwargs = dict(
                            mode=self.oc.dataset.metrics_mode,
                            seq_len=self.oc.layout.out_len,
                            layout=self.layout,
                            threshold_list=self.oc.dataset.threshold_list,
                            metrics_list=self.oc.dataset.metrics_list,
                            fss_scale=self.oc.dataset.get("fss_scale", 10), # Use config value or default
                            eps=self.oc.dataset.get("metrics_eps", 1e-6),  # Use config value or default
                            denormalize_clip_value=self.denorm_clip_value # Pass clip value to metric
                        )
        if self.oc.eval.eval_unaligned:
            self.valid_mse = torchmetrics.MeanSquaredError()
            self.valid_mae = torchmetrics.MeanAbsoluteError()
            self.valid_score = IMERGSkillScore(**metric_kwargs)
            self.test_mse = torchmetrics.MeanSquaredError()
            self.test_mae = torchmetrics.MeanAbsoluteError()
            # Set SSIM data range explicitly
            self.test_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=self.ssim_data_range)
            self.test_score = IMERGSkillScore(**metric_kwargs)
        if self.oc.eval.eval_aligned:
            self.valid_aligned_mse = torchmetrics.MeanSquaredError()
            self.valid_aligned_mae = torchmetrics.MeanAbsoluteError()
            self.valid_aligned_score = IMERGSkillScore(**metric_kwargs)
            self.test_aligned_mse = torchmetrics.MeanSquaredError()
            self.test_aligned_mae = torchmetrics.MeanAbsoluteError()
            # Set SSIM data range explicitly
            self.test_aligned_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=self.ssim_data_range)
            self.test_aligned_score = IMERGSkillScore(**metric_kwargs)

        self.configure_save(cfg_file_path=oc_file)

    @staticmethod
    def _denormalize_tensor(tensor_norm: torch.Tensor, clip_value: float) -> torch.Tensor:
        """Denormalizes a Torch tensor using numpy func, moving data between devices."""
        if not isinstance(tensor_norm, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor_norm)}")

        original_device = tensor_norm.device
        tensor_np = tensor_norm.detach().cpu().numpy()
        denorm_np = _denormalize_log_np(tensor_np, clip_value=clip_value)
        denorm_tensor = torch.from_numpy(denorm_np).to(original_device)
        return denorm_tensor

    def configure_save(self, cfg_file_path=None):
        if self.save_dir is not None:
            self.save_dir = os.path.join(default_exps_dir, self.save_dir)
            os.makedirs(self.save_dir, exist_ok=True)

            if cfg_file_path is not None:
                cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
                if (not os.path.exists(cfg_file_target_path)) or \
                   (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                    try: # Add try-except for robustness
                        copyfile(cfg_file_path, cfg_file_target_path)
                    except Exception as e:
                        warnings.warn(f"Could not copy config file to save_dir: {e}")

            self.example_save_dir = os.path.join(self.save_dir, "examples")
            os.makedirs(self.example_save_dir, exist_ok=True)
            self.npy_save_dir = os.path.join(self.save_dir, "npy")
            os.makedirs(self.npy_save_dir, exist_ok=True)
        else:
            self.example_save_dir = None
            self.npy_save_dir = None

    def configure_optimizers(self):
        optim_cfg = self.hparams.optim # Use self.hparams
        params = list(self.torch_nn_module.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params += list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)

        if optim_cfg.method == "adamw":
            optimizer = torch.optim.AdamW(params, lr=optim_cfg.lr, betas=optim_cfg.betas)
        else:
            raise NotImplementedError(f"Optimization method {optim_cfg.method} not supported.")

        warmup_iter = int(np.round(self.hparams.optim.warmup_percentage * self.total_num_steps))
        if optim_cfg.lr_scheduler_mode == 'none':
            return {'optimizer': optimizer}
        else:
            if optim_cfg.lr_scheduler_mode == 'cosine':
                warmup_scheduler = LambdaLR(optimizer,
                                            lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                    min_lr_ratio=optim_cfg.warmup_min_lr_ratio))
                cosine_scheduler = CosineAnnealingLR(optimizer,
                                                     T_max=(self.total_num_steps - warmup_iter),
                                                     eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr)
                lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                            milestones=[warmup_iter])
                lr_scheduler_config = {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            else:
                raise NotImplementedError
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        # Setup profiler if needed.
        if self.hparams.logging.profiler is None:
            profiler = None
        elif self.hparams.logging.profiler == "pytorch":
            profiler = PyTorchProfiler(filename=f"{self.hparams.logging.logging_prefix}_PyTorchProfiler.log")
        else:
            raise NotImplementedError
        checkpoint_callback = ModelCheckpoint(
            monitor=self.hparams.optim.monitor,
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=self.hparams.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks += [checkpoint_callback]
        if self.hparams.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step')]
        if self.hparams.logging.monitor_device:
            callbacks += [DeviceStatsMonitor()]
        if self.hparams.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.hparams.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.hparams.optim.early_stop_mode)]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.hparams.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.hparams.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger]

        log_every_n_steps = max(1, int(self.hparams.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            profiler=profiler,
            default_root_dir=self.save_dir,
            # accelerator will be set in main based on --device arg
            max_epochs=self.hparams.optim.max_epochs,
            check_val_every_n_epoch=self.hparams.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.hparams.optim.gradient_clip_val,
            precision=self.hparams.trainer.precision,
            num_sanity_val_steps=self.hparams.trainer.num_sanity_val_steps,
            inference_mode=False, # Required for gradient tracking if needed, keep False
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.hparams.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        # Avoid overriding accelerator/devices from config if set_trainer_kwargs is called before device arg processing
        oc_trainer_kwargs.pop('accelerator', None)
        oc_trainer_kwargs.pop('devices', None)
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(cls, num_samples: int, total_batch_size: int, epoch: int) -> int:
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_imerg_datamodule(dataset_cfg: Union[dict, DictConfig], # Allow DictConfig
                             micro_batch_size: int = 1,
                             num_workers: int = 8):
        # Convert OmegaConf to dict if necessary
        if isinstance(dataset_cfg, DictConfig):
            dataset_cfg = OmegaConf.to_object(dataset_cfg)

        dm = IMERGLightningDataModule(
            input_seq_len=dataset_cfg["input_seq_len"],
            output_seq_len=dataset_cfg["output_seq_len"],
            data_dir=dataset_cfg["data_dir"],
            aug_mode=dataset_cfg["aug_mode"],
            ret_contiguous=dataset_cfg["ret_contiguous"],
            val_ratio=dataset_cfg["val_ratio"],
            batch_size=micro_batch_size,
            num_workers=num_workers,
            seed=dataset_cfg["seed"],
            downsample=dataset_cfg["downsample"],
        )
        return dm

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.hparams.layout.layout,
                in_len=self.hparams.layout.in_len,
                out_len=self.hparams.layout.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.hparams.layout.layout,
                in_len=self.hparams.layout.in_len,
                out_len=self.hparams.layout.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    @torch.no_grad()
    def get_input(self, batch, **kwargs):
        return self._get_input_imerg(batch=batch, return_verbose=kwargs.get("return_verbose", False))

    @torch.no_grad()
    def _get_input_imerg(self, batch, return_verbose=False):
        # Assuming batch is already normalized from dataloader
        seq = batch
        in_seq = seq[self.in_slice]
        out_seq = seq[self.out_slice].contiguous()
        if return_verbose:
            # target_seq, cond, context_seq
            return out_seq, {"y": in_seq}, in_seq
        else:
            # target_seq, cond
            return out_seq, {"y": in_seq}

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        micro_batch_size = batch.shape[self.hparams.layout.layout.find("N")] # Get batch dim size correctly
        data_idx = int(batch_idx * micro_batch_size)
        # Check if visualization is needed for this epoch and data index
        if self.current_epoch % self.hparams.trainer.check_val_every_n_epoch == 0:
            if data_idx in self.train_example_data_idx_list:
                target_seq_norm, cond, context_seq_norm = self.get_input(batch, return_verbose=True)
                aligned_pred_seq_norm_list = []
                aligned_pred_label_list = []
                pred_seq_norm_list = []
                pred_label_list = []
                for i in range(self.hparams.eval.num_samples_per_context):
                    # Generate aligned prediction (normalized)
                    if self.use_alignment and self.hparams.eval.eval_aligned:
                        if self.alignment_type == "avg_x":
                            alignment_kwargs = get_alignment_kwargs_avg_x(context_seq=context_seq_norm,
                                                                          target_seq=target_seq_norm)
                        else:
                            raise NotImplementedError
                        pred_seq_norm_aligned = self.sample(
                            cond=cond,
                            batch_size=micro_batch_size,
                            return_intermediates=False,
                            use_alignment=True,
                            alignment_kwargs=alignment_kwargs,
                            verbose=False,
                        ).contiguous()
                        # Store normalized prediction for visualization
                        aligned_pred_seq_norm_list.append(pred_seq_norm_aligned[0].detach().float().cpu().numpy())
                        aligned_pred_label_list.append(f"{self.hparams.logging.logging_prefix}_aligned_{i}")

                    # Generate unaligned prediction (normalized)
                    if self.hparams.eval.eval_unaligned:
                        pred_seq_norm_unaligned = self.sample(
                            cond=cond,
                            batch_size=micro_batch_size,
                            return_intermediates=False,
                            verbose=False,
                        ).contiguous()
                        # Store normalized prediction for visualization
                        pred_seq_norm_list.append(pred_seq_norm_unaligned[0].detach().float().cpu().numpy())
                        pred_label_list.append(f"{self.hparams.logging.logging_prefix}_{i}")

                # Combine lists for visualization (all normalized numpy arrays)
                vis_pred_seq_list = aligned_pred_seq_norm_list + pred_seq_norm_list
                vis_label_list = aligned_pred_label_list + pred_label_list

                # Call visualization with NORMALIZED data
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=context_seq_norm[0].detach().float().cpu().numpy(), # Normalized
                    target_seq=target_seq_norm[0].detach().float().cpu().numpy(),   # Normalized
                    pred_seq=vis_pred_seq_list, # List of normalized numpy arrays
                    pred_label=vis_label_list,
                    mode="train",
                )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self(batch)
        with self.ema_scope():
            _, loss_dict_ema = self(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        micro_batch_size = batch.shape[self.hparams.layout.layout.find("N")]
        data_idx = int(batch_idx * micro_batch_size)

        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            target_seq_norm, cond, context_seq_norm = self.get_input(batch, return_verbose=True)

            # --- Denormalize target ONCE for standard metrics ---
            target_seq_denorm = self._denormalize_tensor(target_seq_norm, self.denorm_clip_value)

            vis_aligned_pred_seq_norm_list = [] # For visualization (normalized)
            vis_aligned_pred_label_list = []
            vis_pred_seq_norm_list = []         # For visualization (normalized)
            vis_pred_label_list = []

            for i in range(self.hparams.eval.num_samples_per_context):
                # --- Aligned Prediction ---
                if self.use_alignment and self.hparams.eval.eval_aligned:
                    if self.alignment_type == "avg_x":
                        alignment_kwargs = get_alignment_kwargs_avg_x(context_seq=context_seq_norm,
                                                                      target_seq=target_seq_norm)
                    else: raise NotImplementedError

                    pred_seq_norm_aligned = self.sample(
                        cond=cond, batch_size=micro_batch_size, return_intermediates=False,
                        use_alignment=True, alignment_kwargs=alignment_kwargs, verbose=False,
                    ).contiguous()

                    # Store normalized for visualization
                    vis_aligned_pred_seq_norm_list.append(pred_seq_norm_aligned[0].detach().float().cpu().numpy())
                    vis_aligned_pred_label_list.append(f"{self.hparams.logging.logging_prefix}_aligned_pred_{i}")

                    # Update SkillScore with NORMALIZED data
                    self.valid_aligned_score.update(pred_seq_norm_aligned, target_seq_norm)

                    # Denormalize prediction for standard metrics
                    pred_seq_denorm_aligned = self._denormalize_tensor(pred_seq_norm_aligned, self.denorm_clip_value)

                    # Update standard metrics with DENORMALIZED data
                    self.valid_aligned_mse(pred_seq_denorm_aligned, target_seq_denorm)
                    self.valid_aligned_mae(pred_seq_denorm_aligned, target_seq_denorm)

                # --- Unaligned Prediction ---
                if self.hparams.eval.eval_unaligned:
                    pred_seq_norm_unaligned = self.sample(
                        cond=cond, batch_size=micro_batch_size, return_intermediates=False, verbose=False,
                    ).contiguous()

                    # Store normalized for visualization
                    vis_pred_seq_norm_list.append(pred_seq_norm_unaligned[0].detach().float().cpu().numpy())
                    vis_pred_label_list.append(f"{self.hparams.logging.logging_prefix}_pred_{i}")

                    # Update SkillScore with NORMALIZED data
                    self.valid_score.update(pred_seq_norm_unaligned, target_seq_norm)

                    # Denormalize prediction for standard metrics
                    pred_seq_denorm_unaligned = self._denormalize_tensor(pred_seq_norm_unaligned, self.denorm_clip_value)

                    # Update standard metrics with DENORMALIZED data
                    self.valid_mse(pred_seq_denorm_unaligned, target_seq_denorm)
                    self.valid_mae(pred_seq_denorm_unaligned, target_seq_denorm)

            # Combine lists for visualization (all normalized numpy arrays)
            vis_pred_seq_list = vis_aligned_pred_seq_norm_list + vis_pred_seq_norm_list
            vis_label_list = vis_aligned_pred_label_list + vis_pred_label_list

            # Call visualization with NORMALIZED data
            self.save_vis_step_end(
                data_idx=data_idx,
                context_seq=context_seq_norm[0].detach().float().cpu().numpy(), # Normalized
                target_seq=target_seq_norm[0].detach().float().cpu().numpy(),   # Normalized
                pred_seq=vis_pred_seq_list, # List of normalized numpy arrays
                pred_label=vis_label_list,
                mode="val",
                suffix="",
            )

    def on_validation_epoch_end(self):
        # Unaligned Metrics
        if self.hparams.eval.eval_unaligned:
            # Standard metrics were updated with denormalized data
            valid_mse = self.valid_mse.compute()
            valid_mae = self.valid_mae.compute()
            # Skill score was updated with normalized data (handled internally)
            valid_score_dict = self.valid_score.compute()
            avg_scores = valid_score_dict.get("avg", {})
            valid_loss = -avg_scores.get("csi", 0.0) # Use 0.0 if CSI not available

            self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True)
            self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True)
            self.log_score_epoch_end(score_dict=valid_score_dict, prefix="valid") # Log detailed skill scores
            # Reset metrics
            self.valid_mse.reset()
            self.valid_mae.reset()
            self.valid_score.reset()

        # Aligned Metrics
        if self.hparams.eval.eval_aligned:
            # Standard metrics were updated with denormalized data
            valid_aligned_mse = self.valid_aligned_mse.compute()
            valid_aligned_mae = self.valid_aligned_mae.compute()
            # Skill score was updated with normalized data (handled internally)
            valid_aligned_score_dict = self.valid_aligned_score.compute()
            avg_aligned_scores = valid_aligned_score_dict.get("avg", {})
            valid_aligned_loss = -avg_aligned_scores.get("csi", 0.0)

            # Log aligned metrics
            self.log('valid_aligned_loss_epoch', valid_aligned_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('valid_aligned_mse_epoch', valid_aligned_mse, prog_bar=True, on_step=False, on_epoch=True)
            self.log('valid_aligned_mae_epoch', valid_aligned_mae, prog_bar=True, on_step=False, on_epoch=True)
            self.log_score_epoch_end(score_dict=valid_aligned_score_dict, prefix="valid_aligned") # Log detailed skill scores
            # Reset metrics
            self.valid_aligned_mse.reset()
            self.valid_aligned_mae.reset()
            self.valid_aligned_score.reset()

    def test_step(self, batch, batch_idx):
        micro_batch_size = batch.shape[self.hparams.layout.layout.find("N")]
        data_idx = int(batch_idx * micro_batch_size)

        if not self.eval_example_only or data_idx in self.test_example_data_idx_list: # Use test list here
            target_seq_norm, cond, context_seq_norm = self.get_input(batch, return_verbose=True)

            # --- Denormalize target ONCE for standard metrics ---
            target_seq_denorm = self._denormalize_tensor(target_seq_norm, self.denorm_clip_value)

            # --- Prepare target for SSIM ---
            if target_seq_denorm.ndim == 5 and target_seq_denorm.shape[-1] == 1: # NTHWC
                 target_ssim_input = rearrange(target_seq_denorm, 'b t h w c -> (b t) c h w')
            elif target_seq_denorm.ndim == 4: # Assume NTHW -> (B*T, 1, H, W)
                 target_ssim_input = rearrange(target_seq_denorm, 'b t h w -> (b t) 1 h w')
            else:
                raise ValueError(f"Unsupported target shape for SSIM: {target_seq_denorm.shape}")
            if abs(self.ssim_data_range - self.denorm_clip_value) > 1e-6:
                target_ssim_input = target_ssim_input / self.denorm_clip_value * self.ssim_data_range


            vis_aligned_pred_seq_norm_list = [] # For visualization (normalized)
            vis_aligned_pred_label_list = []
            vis_pred_seq_norm_list = []         # For visualization (normalized)
            vis_pred_label_list = []

            for i in range(self.hparams.eval.num_samples_per_context):
                # --- Aligned Prediction ---
                if self.use_alignment and self.hparams.eval.eval_aligned:
                    if self.alignment_type == "avg_x":
                        alignment_kwargs = get_alignment_kwargs_avg_x(context_seq=context_seq_norm,
                                                                      target_seq=target_seq_norm)
                    else: raise NotImplementedError

                    pred_seq_norm_aligned = self.sample(
                        cond=cond, batch_size=micro_batch_size, return_intermediates=False,
                        use_alignment=True, alignment_kwargs=alignment_kwargs, verbose=False,
                    ).contiguous()

                    # Store normalized for visualization & optional saving
                    pred_norm_np_aligned = pred_seq_norm_aligned[0].detach().float().cpu().numpy()
                    vis_aligned_pred_seq_norm_list.append(pred_norm_np_aligned)
                    vis_aligned_pred_label_list.append(f"{self.hparams.logging.logging_prefix}_aligned_pred_{i}")
                    if self.hparams.logging.save_npy:
                        npy_path = os.path.join(self.npy_save_dir, f"batch{batch_idx}_sample{i}_aligned_norm.npy")
                        np.save(npy_path, pred_norm_np_aligned) # Save normalized version

                    # Update SkillScore with NORMALIZED data
                    self.test_aligned_score.update(pred_seq_norm_aligned, target_seq_norm)

                    # Denormalize prediction for standard metrics
                    pred_seq_denorm_aligned = self._denormalize_tensor(pred_seq_norm_aligned, self.denorm_clip_value)

                    # Update standard metrics with DENORMALIZED data
                    self.test_aligned_mse(pred_seq_denorm_aligned, target_seq_denorm)
                    self.test_aligned_mae(pred_seq_denorm_aligned, target_seq_denorm)

                    # Prepare prediction for SSIM (rearrange denormalized, potentially scale)
                    if pred_seq_denorm_aligned.ndim == 5 and pred_seq_denorm_aligned.shape[-1] == 1: # NTHWC
                        pred_ssim_input = rearrange(pred_seq_denorm_aligned, 'b t h w c -> (b t) c h w')
                    elif pred_seq_denorm_aligned.ndim == 4: # Assume NTHW
                        pred_ssim_input = rearrange(pred_seq_denorm_aligned, 'b t h w -> (b t) 1 h w')
                    else: raise ValueError(f"Unsupported prediction shape for SSIM: {pred_seq_denorm_aligned.shape}")
                    # Scale prediction for SSIM if needed
                    if abs(self.ssim_data_range - self.denorm_clip_value) > 1e-6:
                         pred_ssim_input = pred_ssim_input / self.denorm_clip_value * self.ssim_data_range
                    # Update SSIM
                    self.test_aligned_ssim(pred_ssim_input, target_ssim_input)


                # --- Unaligned Prediction ---
                if self.hparams.eval.eval_unaligned:
                    pred_seq_norm_unaligned = self.sample(
                        cond=cond, batch_size=micro_batch_size, return_intermediates=False, verbose=False,
                    ).contiguous()

                    # Store normalized for visualization & optional saving
                    pred_norm_np_unaligned = pred_seq_norm_unaligned[0].detach().float().cpu().numpy()
                    vis_pred_seq_norm_list.append(pred_norm_np_unaligned)
                    vis_pred_label_list.append(f"{self.hparams.logging.logging_prefix}_pred_{i}")
                    if self.hparams.logging.save_npy:
                        npy_path = os.path.join(self.npy_save_dir, f"batch{batch_idx}_sample{i}_norm.npy")
                        np.save(npy_path, pred_norm_np_unaligned) # Save normalized version

                    # Update SkillScore with NORMALIZED data
                    self.test_score.update(pred_seq_norm_unaligned, target_seq_norm)

                    # Denormalize prediction for standard metrics
                    pred_seq_denorm_unaligned = self._denormalize_tensor(pred_seq_norm_unaligned, self.denorm_clip_value)

                    # Update standard metrics with DENORMALIZED data
                    self.test_mse(pred_seq_denorm_unaligned, target_seq_denorm)
                    self.test_mae(pred_seq_denorm_unaligned, target_seq_denorm)

                    # Prepare prediction for SSIM (rearrange denormalized, potentially scale)
                    if pred_seq_denorm_unaligned.ndim == 5 and pred_seq_denorm_unaligned.shape[-1] == 1: # NTHWC
                        pred_ssim_input = rearrange(pred_seq_denorm_unaligned, 'b t h w c -> (b t) c h w')
                    elif pred_seq_denorm_unaligned.ndim == 4: # Assume NTHW
                        pred_ssim_input = rearrange(pred_seq_denorm_unaligned, 'b t h w -> (b t) 1 h w')
                    else: raise ValueError(f"Unsupported prediction shape for SSIM: {pred_seq_denorm_unaligned.shape}")
                    # Scale prediction for SSIM if needed
                    if abs(self.ssim_data_range - self.denorm_clip_value) > 1e-6:
                         pred_ssim_input = pred_ssim_input / self.denorm_clip_value * self.ssim_data_range
                    # Update SSIM
                    self.test_ssim(pred_ssim_input, target_ssim_input)


            # Combine lists for visualization (all normalized numpy arrays)
            vis_pred_seq_list = vis_aligned_pred_seq_norm_list + vis_pred_seq_norm_list
            vis_label_list = vis_aligned_pred_label_list + vis_pred_label_list

            # Call visualization with NORMALIZED data
            self.save_vis_step_end(
                data_idx=data_idx,
                context_seq=context_seq_norm[0].detach().float().cpu().numpy(), # Normalized
                target_seq=target_seq_norm[0].detach().float().cpu().numpy(),   # Normalized
                pred_seq=vis_pred_seq_list, # List of normalized numpy arrays
                pred_label=vis_label_list,
                mode="test",
                suffix="",
            )

    def on_test_epoch_end(self):
        # Unaligned Metrics
        if self.hparams.eval.eval_unaligned:
            # Standard metrics were updated with denormalized data (or scaled for SSIM)
            test_mse = self.test_mse.compute()
            test_mae = self.test_mae.compute()
            test_ssim = self.test_ssim.compute()
            # Skill score was updated with normalized data (handled internally)
            test_score_dict = self.test_score.compute()

            # Log metrics
            self.log('test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True)
            self.log('test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True)
            self.log('test_ssim_epoch', test_ssim, prog_bar=True, on_step=False, on_epoch=True)
            self.log_score_epoch_end(score_dict=test_score_dict, prefix="test") # Log detailed skill scores
            # Reset metrics
            self.test_mse.reset()
            self.test_mae.reset()
            self.test_ssim.reset()
            self.test_score.reset()

        # Aligned Metrics
        if self.hparams.eval.eval_aligned:
            # Standard metrics were updated with denormalized data (or scaled for SSIM)
            test_aligned_mse = self.test_aligned_mse.compute()
            test_aligned_mae = self.test_aligned_mae.compute()
            test_aligned_ssim = self.test_aligned_ssim.compute()
            # Skill score was updated with normalized data (handled internally)
            test_aligned_score_dict = self.test_aligned_score.compute()

            # Log aligned metrics
            self.log('test_aligned_mse_epoch', test_aligned_mse, prog_bar=True, on_step=False, on_epoch=True)
            self.log('test_aligned_mae_epoch', test_aligned_mae, prog_bar=True, on_step=False, on_epoch=True)
            self.log('test_aligned_ssim_epoch', test_aligned_ssim, prog_bar=True, on_step=False, on_epoch=True)
            self.log_score_epoch_end(score_dict=test_aligned_score_dict, prefix="test_aligned") # Log detailed skill scores
            # Reset metrics
            self.test_aligned_mse.reset()
            self.test_aligned_mae.reset()
            self.test_aligned_ssim.reset()
            self.test_aligned_score.reset()

    def save_vis_step_end(self,
                          data_idx: int,
                          context_seq: np.ndarray, # Expected Normalized
                          target_seq: np.ndarray,  # Expected Normalized
                          pred_seq: Union[np.ndarray, Sequence[np.ndarray]], # Expected Normalized list/array
                          pred_label: Union[str, Sequence[str]] = None,
                          label_mode: str = "name", # Keep default, avg_int might be misleading if done on normalized
                          mode: str = "train",
                          prefix: str = "",
                          suffix: str = "",
                          ):
        if mode == "train": example_data_idx_list = self.train_example_data_idx_list
        elif mode == "val": example_data_idx_list = self.val_example_data_idx_list
        elif mode == "test": example_data_idx_list = self.test_example_data_idx_list
        else: raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")

        # --- Avg Intensity Labeling ---
        if self.hparams.eval.get("label_avg_int", False): # Check config for this behavior
             label_mode = "avg_int" # Trigger the internal logic

        if label_mode == "name":
            context_label = "context"
            target_label = "target"
            if isinstance(pred_seq, Sequence) and isinstance(pred_label, str):
                pred_label = [pred_label] * len(pred_seq) # Basic replication if single label provided
            elif isinstance(pred_seq, Sequence) and not isinstance(pred_label, Sequence):
                 pred_label = [f"pred_{i}" for i in range(len(pred_seq))] # Default labels
            elif not isinstance(pred_seq, Sequence) and isinstance(pred_label, Sequence):
                 pred_label = pred_label[0] if pred_label else "pred" # Take first label if seq provided
        elif label_mode == "avg_int":
            # Calculate mean on the normalized numpy arrays
            context_mean = np.nanmean(context_seq[np.isfinite(context_seq)])
            target_mean = np.nanmean(target_seq[np.isfinite(target_seq)])
            context_label = f"context\navg_norm={context_mean:.4f}"
            target_label = f"target\navg_norm={target_mean:.4f}"
            if isinstance(pred_label, Sequence):
                temp_labels = []
                for label, seq in zip(pred_label, pred_seq):
                     pred_mean = np.nanmean(seq[np.isfinite(seq)])
                     temp_labels.append(f"{label}\navg_norm={pred_mean:.4f}")
                pred_label = temp_labels
            elif isinstance(pred_label, str):
                 pred_mean = np.nanmean(pred_seq[np.isfinite(pred_seq)])
                 pred_label = f"{pred_label}\navg_norm={pred_mean:.4f}"
            else: raise TypeError(f"Wrong pred_label type {type(pred_label)}!")
        else:
             raise NotImplementedError(f"label_mode '{label_mode}' not implemented.")
        # --- End Avg Intensity Labeling ---


        if isinstance(pred_seq, Sequence):
            seq_list = [context_seq, target_seq] + list(pred_seq)
            label_list = [context_label, target_label] + list(pred_label) # Ensure label is list
        else:
            seq_list = [context_seq, target_seq, pred_seq]
            label_list = [context_label, target_label, pred_label] # Label should be string here


        if data_idx in example_data_idx_list:
            png_save_name = f"{prefix}{mode}_epoch_{self.current_epoch}_data_{data_idx}{suffix}.png"
            # Call vis_imerg_seq with correct denorm settings
            vis_imerg_seq(
                    save_path=os.path.join(self.example_save_dir, png_save_name),
                    seq=seq_list, # Pass normalized sequences
                    label=label_list,
                    denormalize=True, # Let visualization handle denormalization
                    denorm_clip_value=self.denorm_clip_value, # Pass clip value
                    interval_real_time=self.hparams.eval.get("interval_real_time", 30),
                    plot_stride=self.hparams.eval.get("plot_stride", 1),
                    fs=self.hparams.eval.get("fs", 12),
                    fig_title=f"{mode.capitalize()} Epc {self.current_epoch} Idx {data_idx}")

    def log_score_epoch_end(self, score_dict: Dict, prefix: str = "valid"):
        """
        Logs skill scores from the computed score dictionary.
        If mode='1' is used, it logs metrics for each timestep individually.
        """
        if not score_dict or not isinstance(score_dict, dict):
            warnings.warn(f"Received invalid score_dict for prefix '{prefix}'. Skipping logging.")
            return

        is_per_timestep = self.hparams.dataset.metrics_mode == "1"

        # Iterate through the metrics and thresholds defined in the config
        for metrics in self.hparams.dataset.metrics_list:
            # --- Log Per-Threshold Scores ---
            for thresh_str in map(str, self.hparams.dataset.threshold_list):
                thresh_metrics = score_dict.get(thresh_str, {})
                score_val = thresh_metrics.get(metrics)

                if score_val is not None:
                    if is_per_timestep:
                        # score_val is an array of shape (seq_len,)
                        for t in range(len(score_val)):
                            timestep_score = score_val[t]
                            # e.g. valid_csi_2.5_t1_epoch, valid_csi_2.5_t2_epoch ...
                            log_name = f"{prefix}_{metrics}_{thresh_str}_t{t+1}_epoch"
                            self.log(log_name, timestep_score,
                                    prog_bar=False, on_step=False, on_epoch=True)
                    else:
                        # score_val is a single float
                        # e.g. valid_csi_2.5_epoch
                        log_name = f"{prefix}_{metrics}_{thresh_str}_epoch"
                        self.log(log_name, score_val,
                                prog_bar=False, on_step=False, on_epoch=True)
                else:
                    warnings.warn(f"Metric '{metrics}' for threshold '{thresh_str}' not found in score_dict for prefix '{prefix}'.")

            # --- Log Average-Over-Thresholds Scores (from 'avg' key) ---
            # This is useful for a high-level view and the progress bar
            score_avg_dict = score_dict.get("avg", {})
            grand_avg_score = score_avg_dict.get(metrics)

            if grand_avg_score is not None:
                # Log the single, overall average value (averaged over thresholds and time)
                # This is good for the progress bar. e.g., valid_csi_avg_epoch
                self.log(f"{prefix}_{metrics}_avg_epoch", grand_avg_score,
                        prog_bar=True, on_step=False, on_epoch=True)


    def on_before_optimizer_step(self, optimizer):
        # Use self.hparams for configuration
        if self.hparams.logging.get("track_grad_norm", -1) != -1:
            norms = grad_norm(self.torch_nn_module, norm_type=self.hparams.logging.track_grad_norm)
            self.log_dict(norms)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_imerg', type=str)
    parser.add_argument('--cfg', default=None, type=str, required=True, help='Path to config file')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on IMERG.')
    # Add the device argument
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                        help='Device to use for training (cpu or gpu)')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    oc_from_file = OmegaConf.load(open(args.cfg, "r"))
    # Access config values via dot notation directly from oc_from_file
    total_batch_size = oc_from_file.optim.total_batch_size
    micro_batch_size = oc_from_file.optim.micro_batch_size
    max_epochs = oc_from_file.optim.max_epochs
    seed = oc_from_file.optim.seed
    # Use .get for potentially missing keys with a default
    matmul_precision = oc_from_file.optim.get("float32_matmul_precision", "medium")
    torch.set_float32_matmul_precision(matmul_precision)
    seed_everything(seed, workers=True)

    dm = PreDiffIMERGPLModule.get_imerg_datamodule(
        dataset_cfg=oc_from_file.dataset, # Pass the OmegaConf object directly
        micro_batch_size=micro_batch_size,
        num_workers=oc_from_file.trainer.get("num_workers", 8), # Get from trainer config or default
    )
    dm.prepare_data()
    dm.setup()

    # Ensure micro_batch_size divides total_batch_size
    if total_batch_size % micro_batch_size != 0:
        raise ValueError(f"total_batch_size ({total_batch_size}) must be divisible by micro_batch_size ({micro_batch_size})")
    accumulate_grad_batches = total_batch_size // micro_batch_size

    total_num_steps = PreDiffIMERGPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = PreDiffIMERGPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save, # Pass save_dir argument
        oc_file=args.cfg    # Pass config file path
    )
    trainer_kwargs = pl_module.set_trainer_kwargs(
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # --- Set device based on args ---
    if args.device == 'gpu':
        if torch.cuda.is_available():
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = 'auto' # Let PyTorch Lightning handle device selection
            print(f"Using GPU (accelerator='gpu', devices='auto').")
        else:
            print("GPU specified via --device but torch.cuda.is_available() is False. Falling back to CPU.")
            trainer_kwargs['accelerator'] = 'cpu'
            trainer_kwargs.pop('devices', None) # Remove devices if set
    elif args.device == 'cpu':
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs.pop('devices', None) # Remove devices if set
        print("Using CPU (accelerator='cpu').")
    # No else needed due to choices=['cpu', 'gpu'] in argparse

    trainer = Trainer(
        **trainer_kwargs, 
        # fast_dev_run=True
    )

    ckpt_path = None # Initialize ckpt_path
    if args.ckpt_name is not None:
        if args.ckpt_name.endswith('.pt'):
            print(f"Loading pretrained .pt weights for {'testing' if args.test else 'training'}")
            try:
                # Load to CPU first to avoid device mismatches before model is moved
                pretrained_weights = torch.load(args.ckpt_name, map_location=torch.device("cpu"))
                # Attempt to load into the diffusion model module
                pl_module.torch_nn_module.load_state_dict(pretrained_weights, strict=False)
                print(".pt weights loaded successfully into torch_nn_module.")
            except Exception as e:
                 warnings.warn(f"Could not load .pt weights directly: {e}. Check model structure.")
            # For .pt files, we don't use Lightning's ckpt_path mechanism for resuming trainer state
            ckpt_path = None
        else:
            # Assume it's a Lightning checkpoint name (relative to save_dir/checkpoints)
             potential_ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
             if os.path.exists(potential_ckpt_path):
                 ckpt_path = potential_ckpt_path
                 print(f"Found Lightning checkpoint: {ckpt_path}")
             else:
                  warnings.warn(f"Specified ckpt_name '{args.ckpt_name}' not found at {potential_ckpt_path}.")
                  ckpt_path = None # Ensure ckpt_path is None if file not found
    elif not args.test:
         # Check for last.ckpt if not testing and no specific ckpt provided
         last_ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", "last.ckpt")
         if os.path.exists(last_ckpt_path):
              ckpt_path = last_ckpt_path
              print(f"Resuming from last checkpoint: {ckpt_path}")


    if args.test:
        # If testing with a .pt file, weights are already loaded above.
        # If testing with a .ckpt file, ckpt_path is set.
        # If no ckpt_name provided for test, ckpt_path remains None, Trainer tests from scratch (unlikely desired).
        if ckpt_path is None and not (args.ckpt_name and args.ckpt_name.endswith('.pt')):
             warnings.warn("Testing without a valid checkpoint specified.")
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path if ckpt_path else "best") # Use 'best' if no specific ckpt
    else:
        # If training with a .pt file, weights loaded, ckpt_path is None. Start fresh training state.
        # If training with a .ckpt file, ckpt_path is set. Trainer resumes state.
        # If resuming from last.ckpt, ckpt_path is set.
        # If no checkpoint found, ckpt_path is None. Starts fresh.
        trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)

        # Save best model as PyTorch state dict after training completes
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path and os.path.exists(best_ckpt_path):
             print(f"Saving best model weights from {best_ckpt_path} to {pytorch_state_dict_name}")
             try:
                 pl_ckpt = pl_load(path_or_url=best_ckpt_path, map_location=torch.device("cpu"))
                 pl_state_dict = pl_ckpt.get('state_dict', pl_ckpt)

                 model_key = "torch_nn_module."
                 state_dict = OrderedDict()
                 for key, val in pl_state_dict.items():
                     if key.startswith(model_key):
                         state_dict[key.replace(model_key, "")] = val
                     elif key in pl_module.torch_nn_module.state_dict():
                          state_dict[key] = val

                 if state_dict: # Ensure we extracted something
                     torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
                     print(f"Saved state_dict to {os.path.join(pl_module.save_dir, 'checkpoints', pytorch_state_dict_name)}")

                     # Test the best model after saving
                     print("Testing the best model...")
                     trainer.test(ckpt_path="best", datamodule=dm)
                 else:
                      warnings.warn("Could not extract torch_nn_module state_dict from the best checkpoint.")

             except Exception as e:
                  warnings.warn(f"Failed to save best model state_dict or test: {e}")
        else:
            warnings.warn("Best checkpoint path not found after training. Cannot save PyTorch state dict or test.")


if __name__ == "__main__":
    main()