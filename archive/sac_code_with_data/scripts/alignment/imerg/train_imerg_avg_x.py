import warnings
from typing import Sequence, Union, Dict
from shutil import copyfile
import inspect
from collections import OrderedDict
import numpy as np
import torch
import torchmetrics
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import (
    Callback, LearningRateMonitor, DeviceStatsMonitor,
    EarlyStopping, ModelCheckpoint,
)
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf
import os
import argparse

from src.prediff.datasets.imerg.imerg_torch_wrap import IMERGLightningDataModule
from src.prediff.utils.pl_checkpoint import pl_load
from src.prediff.utils.optim import warmup_lambda
from src.prediff.utils.layout import layout_to_in_out_slice
from src.prediff.utils.path import (
    default_exps_dir,
)
from src.prediff.taming import AutoencoderKL
from src.prediff.diffusion.knowledge_alignment.alignment_pl import AlignmentPL
from src.prediff.diffusion.knowledge_alignment.imerg import IMERGAvgIntensityAlignment

pytorch_state_dict_name = "imerg_alignment_avgx.pt"


class IMERGAlignmentPLModule(AlignmentPL):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str,
                 save_dir: str = None):
        self.total_num_steps = total_num_steps
        oc = OmegaConf.load(open(oc_file, "r"))
        self.save_hyperparameters(oc)
        self.oc = oc

        knowledge_alignment_cfg = OmegaConf.to_object(oc.model.align)
        self.alignment_obj = IMERGAvgIntensityAlignment(
            alignment_type=knowledge_alignment_cfg["alignment_type"],
            model_type=knowledge_alignment_cfg["model_type"],
            model_args=knowledge_alignment_cfg["model_args"],
        )

        vae_cfg = OmegaConf.to_object(oc.model.vae)
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
            state_dict = torch.load(pretrained_ckpt_path, # Use path directly
                                    map_location=torch.device("cpu"))
            first_stage_model.load_state_dict(state_dict=state_dict)
        else:
            warnings.warn("Pretrained weights for `AutoencoderKL` not set. Run for sanity check only.")

        diffusion_cfg = OmegaConf.to_object(oc.model.diffusion)
        super(IMERGAlignmentPLModule, self).__init__(
            torch_nn_module=self.alignment_obj.model,
            target_fn=self.alignment_obj.model_objective,
            layout=oc.layout.layout,
            timesteps=diffusion_cfg["timesteps"],
            beta_schedule=diffusion_cfg["beta_schedule"],
            loss_type=oc.optim.loss_type,
            monitor=oc.optim.monitor,
            linear_start=diffusion_cfg["linear_start"],
            linear_end=diffusion_cfg["linear_end"],
            cosine_s=diffusion_cfg["cosine_s"],
            given_betas=diffusion_cfg["given_betas"],
            # latent diffusion
            first_stage_model=first_stage_model,
            cond_stage_model=diffusion_cfg["cond_stage_model"],
            num_timesteps_cond=diffusion_cfg["num_timesteps_cond"],
            cond_stage_trainable=diffusion_cfg["cond_stage_trainable"],
            cond_stage_forward=diffusion_cfg["cond_stage_forward"],
            scale_by_std=diffusion_cfg["scale_by_std"],
            scale_factor=diffusion_cfg["scale_factor"],
        )
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        if self.save_dir is not None:
            self.save_dir = os.path.join(default_exps_dir, self.save_dir)
            os.makedirs(self.save_dir, exist_ok=True)
            if cfg_file_path is not None:
                cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
                if (not os.path.exists(cfg_file_target_path)) or (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                    try:
                        copyfile(cfg_file_path, cfg_file_target_path)
                    except Exception as e:
                         warnings.warn(f"Could not copy config file {cfg_file_path} to {cfg_file_target_path}: {e}")
        else:
             self.example_save_dir = None # Explicitly set to None if save_dir is None

    def configure_optimizers(self):
        optim_cfg = self.oc.optim
        params = list(self.torch_nn_module.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params += list(self.cond_stage_model.parameters())

        if optim_cfg.method == "adamw":
            optimizer = torch.optim.AdamW(params, lr=optim_cfg.lr, betas=optim_cfg.betas)
        else:
            raise NotImplementedError(f"Optimization method {optim_cfg.method} not supported.")

        warmup_iter = int(np.round(optim_cfg.warmup_percentage * self.total_num_steps))
        if optim_cfg.lr_scheduler_mode == 'none':
            return {'optimizer': optimizer}
        else:
            if optim_cfg.lr_scheduler_mode == 'cosine':
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=optim_cfg.warmup_min_lr_ratio)
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=(self.total_num_steps - warmup_iter),
                    eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr
                )
                lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_iter]
                )
                lr_scheduler_config = {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            else:
                raise NotImplementedError
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    # Modified set_trainer_kwargs to accept device
    def set_trainer_kwargs(self, device: str, **kwargs):
        if self.oc.logging.profiler is None:
            profiler = None
        elif self.oc.logging.profiler == "pytorch":
            profiler = PyTorchProfiler(filename=f"{self.oc.logging.logging_prefix}_PyTorchProfiler.log")
        else:
            raise NotImplementedError
        checkpoint_callback = ModelCheckpoint(
            monitor=self.oc.optim.monitor,
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        # Ensure callbacks is a list before attempting to extend
        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        # Check for valid callback instances (optional but good practice)
        # for ele in callbacks:
        #     assert isinstance(ele, Callback)

        callbacks += [checkpoint_callback]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step')]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor()]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode)]
        logger = kwargs.pop("logger", [])
        # Ensure logger is a list before attempting to extend
        if not isinstance(logger, list):
            logger = [logger]

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix, save_dir=self.save_dir)
            logger += [wandb_logger]
        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()

        ret = dict(
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            profiler=profiler,
            default_root_dir=self.save_dir,
            # accelerator="cpu", # REMOVED HARDCODED CPU
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            precision=self.oc.trainer.precision,
            num_sanity_val_steps=self.oc.trainer.num_sanity_val_steps,
            inference_mode=False,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)

        # Set accelerator based on the device argument
        ret['accelerator'] = device
        if device == 'gpu':
            # Use 'auto' for automatic GPU detection/selection if not specified elsewhere
            if 'devices' not in ret and 'devices' not in kwargs:
                ret['devices'] = 'auto'
        else: # cpu
            # Remove 'devices' key if it exists for CPU
            ret.pop('devices', None)

        # Apply remaining explicit kwargs, potentially overriding config/device settings
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(cls, num_samples: int, total_batch_size: int, epoch: int):
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_imerg_datamodule(dataset_cfg, micro_batch_size: int = 1, num_workers: int = 0):
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
                layout=self.oc.layout.layout,
                in_len=self.oc.layout.in_len,
                out_len=self.oc.layout.out_len
            )
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.oc.layout.layout,
                in_len=self.oc.layout.in_len,
                out_len=self.oc.layout.out_len
            )
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    @property
    def intensity_avg_dims(self):
        if not hasattr(self, "_intensity_avg_dims"):
            # Calculate dims based on layout string directly from hparams/config
            layout = self.hparams.layout.layout # Access layout from hparams
            self._intensity_avg_dims = tuple(layout.find(dim) for dim in "HWC" if dim in layout)
            if not self._intensity_avg_dims:
                 warnings.warn(f"Could not find H, W, or C in layout '{layout}'. Intensity averaging might be incorrect.")
        return self._intensity_avg_dims


    @torch.no_grad()
    def get_input(self, batch, **kwargs):
        x, cond, extra = self._get_input_imerg(batch, return_verbose=kwargs.get("return_verbose", False))
        # Making contiguous might not be strictly necessary anymore unless specific ops require it.
        # If performance issues arise on CPU, consider re-adding x = x.contiguous()
        return x, cond, extra

    @torch.no_grad()
    def _get_input_imerg(self, batch, return_verbose=False):
        seq = batch
        in_seq = seq[self.in_slice]
        out_seq = seq[self.out_slice]
        if return_verbose:
            avg_dims = self.intensity_avg_dims # Use property to get dims
            # Ensure avg_dims is not empty before attempting mean calculation
            if avg_dims:
                avg_x_gt = torch.mean(out_seq, dim=avg_dims, keepdim=True) # Keep dim for consistency if needed later
            else:
                avg_x_gt = torch.zeros_like(out_seq[:, :1, ...]) # Placeholder if dims not found
                warnings.warn("Cannot calculate intensity average due to missing HWC dims.")
            return out_seq.contiguous(), {"y": in_seq.contiguous()}, {"avg_x_gt": avg_x_gt}
        else:
            return out_seq.contiguous(), {"y": in_seq.contiguous()}, {}

    def on_before_optimizer_step(self, optimizer):
        # Use self.hparams for consistency
        if self.hparams.logging.get("track_grad_norm", -1) != -1:
            norms = grad_norm(self.torch_nn_module, norm_type=self.hparams.logging.track_grad_norm)
            self.log_dict(norms)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_imerg_avg_x', type=str)
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on IMERG (.ckpt or .pt)')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'],
                        help='Device to use for training/testing (cpu or gpu)')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    oc_from_file = OmegaConf.load(open(args.cfg, "r"))
    # Convert relevant parts early if needed, or pass OmegaConf object
    dataset_cfg = oc_from_file.dataset # Keep as OmegaConf for datamodule if it handles it
    total_batch_size = oc_from_file.optim.total_batch_size
    micro_batch_size = oc_from_file.optim.micro_batch_size
    max_epochs = oc_from_file.optim.max_epochs
    seed = oc_from_file.optim.seed
    # Use .get for safer access
    float32_matmul_precision = oc_from_file.optim.get("float32_matmul_precision", "medium")
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed_everything(seed, workers=True)
    dm = IMERGAlignmentPLModule.get_imerg_datamodule(
        dataset_cfg=dataset_cfg, # Pass OmegaConf or dict
        micro_batch_size=micro_batch_size,
        num_workers=oc_from_file.trainer.get("num_workers", 0), # Get from trainer or default
    )
    dm.prepare_data()
    dm.setup()

    if total_batch_size % micro_batch_size != 0:
         raise ValueError(f"total_batch_size ({total_batch_size}) must be divisible by micro_batch_size ({micro_batch_size})")
    accumulate_grad_batches = total_batch_size // micro_batch_size

    total_num_steps = IMERGAlignmentPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = IMERGAlignmentPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg
    )

    # Pass the device argument to set_trainer_kwargs
    trainer_kwargs = pl_module.set_trainer_kwargs(
        device=args.device, # Pass the selected device
        accumulate_grad_batches=accumulate_grad_batches # Pass other kwargs as needed
    )
    print(f"Configuring Trainer with kwargs: {trainer_kwargs}") # Optional: print resolved trainer args
    trainer = Trainer(**trainer_kwargs)

    # Handle checkpoint loading logic
    ckpt_path = None # Initialize
    if args.ckpt_name:
        # Check if it's a file path or just a name relative to checkpoints dir
        if os.path.isfile(args.ckpt_name):
            potential_ckpt_path = args.ckpt_name
        elif pl_module.save_dir: # Only construct path if save_dir is valid
             potential_ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        else:
             potential_ckpt_path = None # Cannot construct path if save_dir is None

        if potential_ckpt_path and os.path.exists(potential_ckpt_path):
            if args.ckpt_name.endswith('.pt'):
                print(f"Loading pretrained .pt weights from {potential_ckpt_path}")
                try:
                    pretrained_weights = torch.load(potential_ckpt_path, map_location=torch.device("cpu"))
                    # Attempt to load into the alignment model module
                    pl_module.torch_nn_module.load_state_dict(pretrained_weights, strict=False)
                    print(".pt weights loaded successfully into torch_nn_module.")
                except Exception as e:
                    warnings.warn(f"Could not load .pt weights directly into torch_nn_module: {e}. Check model structure.")
                # For .pt files, we don't use Lightning's ckpt_path mechanism for resuming trainer state
                ckpt_path = None
            else: # Assume it's a Lightning .ckpt file
                ckpt_path = potential_ckpt_path
                print(f"Using Lightning checkpoint: {ckpt_path}")
        else:
            warnings.warn(f"Specified ckpt_name '{args.ckpt_name}' not found at expected path '{potential_ckpt_path}'.")
            ckpt_path = None # Ensure ckpt_path is None if file not found
    elif not args.test:
         # Check for last.ckpt if not testing and no specific ckpt provided
         if pl_module.save_dir and os.path.isdir(os.path.join(pl_module.save_dir, "checkpoints")):
             last_ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", "last.ckpt")
             if os.path.exists(last_ckpt_path):
                 ckpt_path = last_ckpt_path
                 print(f"Resuming from last checkpoint: {ckpt_path}")


    if args.test:
        if ckpt_path is None and not (args.ckpt_name and args.ckpt_name.endswith('.pt')):
            warnings.warn("Testing without a valid .ckpt checkpoint specified.")
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path) # Pass potentially None ckpt_path
    else:
        trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path) # ckpt_path handles resume/start

        # Ensure save_dir is valid before proceeding
        if pl_module.save_dir and os.path.exists(pl_module.save_dir) and trainer.checkpoint_callback:
            best_ckpt_path = trainer.checkpoint_callback.best_model_path
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                 print(f"Saving best model weights from {best_ckpt_path} to {pytorch_state_dict_name}")
                 try:
                    # Load best checkpoint from Lightning
                    pl_ckpt = pl_load(path_or_url=best_ckpt_path, map_location=torch.device("cpu"))
                    # Extract state_dict, handling potential differences in Lightning versions
                    pl_state_dict = pl_ckpt.get('state_dict', pl_ckpt)

                    # Extract only the state_dict for the torch_nn_module (the alignment model)
                    model_key = "torch_nn_module."
                    state_dict = OrderedDict()
                    for key, val in pl_state_dict.items():
                        if key.startswith(model_key):
                            state_dict[key.replace(model_key, "")] = val
                        elif key in pl_module.torch_nn_module.state_dict(): # Handle case without prefix
                             state_dict[key] = val

                    if state_dict: # Ensure we extracted something
                        save_path = os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name)
                        torch.save(state_dict, save_path)
                        print(f"Saved Alignment model state_dict to {save_path}")

                        # Test the best model after saving
                        print("Testing the best model...")
                        trainer.test(ckpt_path="best", datamodule=dm)
                    else:
                         warnings.warn("Could not extract torch_nn_module state_dict from the best checkpoint.")

                 except Exception as e:
                    warnings.warn(f"Failed to save best model state_dict or test: {e}")
            else:
                 warnings.warn("Best checkpoint path not found after training. Cannot save PyTorch state dict or test.")
        else:
             warnings.warn("Save directory not valid or checkpoint callback not found. Cannot save PyTorch state dict or test.")


if __name__ == "__main__":
    main()