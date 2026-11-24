import warnings
from shutil import copyfile
import inspect
from collections import OrderedDict
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange

from src.prediff.datasets.imerg.imerg_torch_wrap import IMERGLightningDataModule
from src.prediff.datasets.imerg.visualization import vis_imerg_seq
from src.prediff.taming import AutoencoderKL, LPIPSWithDiscriminator
from src.prediff.utils.optim import warmup_lambda
from src.prediff.utils.pl_checkpoint import pl_load
from src.prediff.utils.path import default_exps_dir

pytorch_state_dict_name = "imerg_vae.pt"
pytorch_loss_state_dict_name = "imerg_vae_loss.pt"

class VAEIMERGPLModule(pl.LightningModule):
    def __init__(self, total_num_steps: int, accumulate_grad_batches: int = 1, oc_file: str = None, save_dir: str = None):
        super().__init__()
        if oc_file is None:
            raise ValueError("Config file is required")
        self.oc = OmegaConf.load(oc_file)
        self.torch_nn_module = AutoencoderKL(
            down_block_types=self.oc.model.down_block_types,
            in_channels=self.oc.model.in_channels,
            block_out_channels=self.oc.model.block_out_channels,
            act_fn=self.oc.model.act_fn,
            latent_channels=self.oc.model.latent_channels,
            up_block_types=self.oc.model.up_block_types,
            norm_num_groups=self.oc.model.norm_num_groups,
            layers_per_block=self.oc.model.layers_per_block,
            out_channels=self.oc.model.out_channels,
        )
        self.loss = LPIPSWithDiscriminator(
            disc_start=self.oc.model.loss.disc_start,
            kl_weight=self.oc.model.loss.kl_weight,
            disc_weight=self.oc.model.loss.disc_weight,
            perceptual_weight=self.oc.model.loss.perceptual_weight,
            disc_in_channels=self.oc.model.loss.disc_in_channels,
        )
        self.total_num_steps = total_num_steps
        self.save_hyperparameters(self.oc)
        self.layout = self.oc.layout.layout
        self.channel_axis = self.layout.find("C")
        self.batch_axis = self.layout.find("N")
        self.t_axis = self.layout.find("T")
        self.h_axis = self.layout.find("H")
        self.w_axis = self.layout.find("W")
        self.channels = self.oc.model.data_channels
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_epochs = self.oc.optim.max_epochs
        self.optim_method = self.oc.optim.method
        self.lr = self.oc.optim.lr
        self.wd = self.oc.optim.wd
        self.save_dir = save_dir
        self.logging_prefix = self.oc.logging.logging_prefix
        self.train_example_data_idx_list = list(self.oc.eval.train_example_data_idx_list)
        self.val_example_data_idx_list = list(self.oc.eval.val_example_data_idx_list)
        self.test_example_data_idx_list = list(self.oc.eval.test_example_data_idx_list)
        self.eval_example_only = self.oc.eval.eval_example_only
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(default_exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def configure_optimizers(self):
        optim_cfg = self.oc.optim
        lr = optim_cfg.lr
        betas = optim_cfg.betas
        opt_ae = torch.optim.Adam(
            list(self.torch_nn_module.encoder.parameters()) +
            list(self.torch_nn_module.decoder.parameters()) +
            list(self.torch_nn_module.quant_conv.parameters()) +
            list(self.torch_nn_module.post_quant_conv.parameters()),
            lr=lr, betas=betas)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=betas)
        warmup_iter = int(np.round(optim_cfg.warmup_percentage * self.total_num_steps))
        if optim_cfg.lr_scheduler_mode == 'none':
            return [{"optimizer": opt_ae}, {"optimizer": opt_disc}]
        else:
            if optim_cfg.lr_scheduler_mode == 'cosine':
                warmup_scheduler_ae = LambdaLR(opt_ae, lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=optim_cfg.warmup_min_lr_ratio))
                cosine_scheduler_ae = CosineAnnealingLR(opt_ae, T_max=(self.total_num_steps - warmup_iter), eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr)
                lr_scheduler_ae = SequentialLR(opt_ae, schedulers=[warmup_scheduler_ae, cosine_scheduler_ae], milestones=[warmup_iter])
                lr_scheduler_config_ae = {'scheduler': lr_scheduler_ae, 'interval': 'step', 'frequency': 1}
                warmup_scheduler_disc = LambdaLR(opt_disc, lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=optim_cfg.warmup_min_lr_ratio))
                cosine_scheduler_disc = CosineAnnealingLR(opt_disc, T_max=(self.total_num_steps - warmup_iter), eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr)
                lr_scheduler_disc = SequentialLR(opt_disc, schedulers=[warmup_scheduler_disc, cosine_scheduler_disc], milestones=[warmup_iter])
                lr_scheduler_config_disc = {'scheduler': lr_scheduler_disc, 'interval': 'step', 'frequency': 1}
            else:
                raise NotImplementedError
            return [{"optimizer": opt_ae, "lr_scheduler": lr_scheduler_config_ae},
                    {"optimizer": opt_disc, "lr_scheduler": lr_scheduler_config_disc}]

    def set_trainer_kwargs(self, device: str, **kwargs):
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
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step')]
        if self.oc.logging.monitor_device and device == 'gpu':
            callbacks += [DeviceStatsMonitor()]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch", min_delta=0.0, patience=self.oc.optim.early_stop_patience, verbose=False, mode=self.oc.optim.early_stop_mode)]
        
        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix, save_dir=self.save_dir)
            logger += [wandb_logger]
        
        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()

        trainer_devices_arg = 1
        if device == 'gpu':
            trainer_devices_arg = -1 

        ret = dict(
            num_sanity_val_steps=self.oc.trainer.num_sanity_val_steps,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            default_root_dir=self.save_dir,
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            precision=self.oc.trainer.precision,
        )
        
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        filtered_oc_trainer_kwargs = {
            key: val for key, val in oc_trainer_kwargs.items() 
            if key in trainer_init_keys and key not in ["accelerator", "devices"]
        }
        ret.update(filtered_oc_trainer_kwargs)

        ret["accelerator"] = device
        ret["devices"] = trainer_devices_arg

        ret.update(kwargs)
        return ret


    @classmethod
    def get_total_num_steps(cls, num_samples: int, total_batch_size: int, epoch: int):
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_imerg_datamodule(dataset_cfg, micro_batch_size: int = 1):
        dm = IMERGLightningDataModule(
            input_seq_len=dataset_cfg["input_seq_len"],
            output_seq_len=dataset_cfg["output_seq_len"],
            data_dir=dataset_cfg["data_dir"],
            aug_mode=dataset_cfg["aug_mode"],
            ret_contiguous=dataset_cfg["ret_contiguous"],
            val_ratio=dataset_cfg["val_ratio"],
            batch_size=micro_batch_size,
            seed=dataset_cfg["seed"],
            downsample=dataset_cfg["downsample"],
        )
        return dm

    def get_last_layer(self):
        return self.torch_nn_module.decoder.conv_out.weight

    def get_input(self, batch):
        target_bchw = rearrange(batch, "b 1 h w c -> b c h w").contiguous()
        mask = None
        return target_bchw, mask

    def forward(self, target_bchw, sample_posterior=True):
        pred_bchw, posterior = self.torch_nn_module(sample=target_bchw, sample_posterior=sample_posterior, return_posterior=True)
        return pred_bchw, posterior

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()
        target_bchw, _ = self.get_input(batch=batch)
        pred_bchw, posterior = self(target_bchw)
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if self.current_epoch % self.oc.trainer.check_val_every_n_epoch == 0:
            self.save_vis_step_end(data_idx=data_idx, target=target_bchw.detach().float().cpu().numpy(), pred=pred_bchw.detach().float().cpu().numpy(), mode="train")
        aeloss, log_dict_ae = self.loss(target_bchw, pred_bchw, posterior, optimizer_idx=0, global_step=self.global_step, mask=None, last_layer=self.get_last_layer(), split="train")
        aeloss /= self.accumulate_grad_batches
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)
        self.manual_backward(aeloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(g_opt, gradient_clip_val=self.oc.optim.gradient_clip_val, gradient_clip_algorithm="norm")
            g_opt.step()
            g_sch.step()
            g_opt.zero_grad()
        discloss, log_dict_disc = self.loss(target_bchw, pred_bchw, posterior, optimizer_idx=1, global_step=self.global_step, mask=None, last_layer=self.get_last_layer(), split="train")
        discloss /= self.accumulate_grad_batches
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)
        self.manual_backward(discloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(d_opt, gradient_clip_val=self.oc.optim.gradient_clip_val, gradient_clip_algorithm="norm")
            d_opt.step()
            d_sch.step()
            d_opt.zero_grad()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            target_bchw, _ = self.get_input(batch=batch)
            pred_bchw, posterior = self(target_bchw)
            target_bchw = target_bchw.contiguous()
            pred_bchw = pred_bchw.contiguous()
            self.save_vis_step_end(data_idx=data_idx, target=target_bchw.detach().float().cpu().numpy(), pred=pred_bchw.detach().float().cpu().numpy(), mode="val")
            aeloss, log_dict_ae = self.loss(target_bchw, pred_bchw, posterior, 0, self.global_step, mask=None, last_layer=self.get_last_layer(), split="val")
            discloss, log_dict_disc = self.loss(target_bchw, pred_bchw, posterior, 1, self.global_step, mask=None, last_layer=self.get_last_layer(), split="val")
            self.log("val/rec_loss", log_dict_ae["val/rec_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.valid_mse.update(pred_bchw, target_bchw)
            self.valid_mae.update(pred_bchw, target_bchw)

    def on_validation_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            target_bchw, _ = self.get_input(batch=batch)
            pred_bchw, posterior = self(target_bchw)
            target_bchw = target_bchw.contiguous()
            pred_bchw = pred_bchw.contiguous()
            self.save_vis_step_end(data_idx=data_idx, target=target_bchw.detach().float().cpu().numpy(), pred=pred_bchw.detach().float().cpu().numpy(), mode="test")
            aeloss, log_dict_ae = self.loss(target_bchw, pred_bchw, posterior, 0, self.global_step, mask=None, last_layer=self.get_last_layer(), split="test")
            discloss, log_dict_disc = self.loss(target_bchw, pred_bchw, posterior, 1, self.global_step, mask=None, last_layer=self.get_last_layer(), split="test")
            self.log("test/rec_loss", log_dict_ae["test/rec_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.test_mse.update(pred_bchw, target_bchw)
            self.test_mae.update(pred_bchw, target_bchw)

    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        self.log('test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_mse.reset()
        self.test_mae.reset()

    def save_vis_step_end(self, data_idx: int, target: np.ndarray, pred: np.ndarray, mode: str = "train", prefix: str = ""):
        if mode == "train":
            example_data_idx_list = self.train_example_data_idx_list
        elif mode == "val":
            example_data_idx_list = self.val_example_data_idx_list
        elif mode == "test":
            example_data_idx_list = self.test_example_data_idx_list
        else:
            raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
        if data_idx in example_data_idx_list:
            save_name = f"{prefix}{mode}_epoch_{self.current_epoch}_data_{data_idx}.png"
            num_vis = min(target.shape[0], self.oc.eval.num_vis)
            seq_list = [target[:num_vis].squeeze(1), pred[:num_vis].squeeze(1)]
            label_list = ["Target", f"{self.oc.logging.logging_prefix}"]
            vis_imerg_seq(save_path=os.path.join(self.example_save_dir, save_name), seq=seq_list, label=label_list, plot_stride=1, fs=18, label_rotation=90)

    def on_before_optimizer_step(self, optimizer):
        if self.oc.logging.track_grad_norm != -1:
            norms = grad_norm(self.torch_nn_module, norm_type=self.oc.logging.track_grad_norm)
            self.log_dict(norms)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_vae_imerg', type=str)
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', 'gpu'], help='Device to use (cpu or gpu)')
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str, help='The model checkpoint trained on IMERG.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.cfg is None:
        raise ValueError("Config file is required")
    oc_from_file = OmegaConf.load(args.cfg)
    dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
    total_batch_size = oc_from_file.optim.total_batch_size
    micro_batch_size = oc_from_file.optim.micro_batch_size
    max_epochs = oc_from_file.optim.max_epochs
    seed = oc_from_file.optim.seed
    float32_matmul_precision = oc_from_file.optim.float32_matmul_precision
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed_everything(seed, workers=True)
    dm = VAEIMERGPLModule.get_imerg_datamodule(dataset_cfg=dataset_cfg, micro_batch_size=micro_batch_size)
    dm.prepare_data()
    dm.setup()

    num_devices_for_grad_accum = 1
    if args.device == 'gpu':
        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()
            if num_cuda_devices > 0:
                num_devices_for_grad_accum = num_cuda_devices
            else:
                warnings.warn("GPU selected, but no CUDA devices found. Accumulation batch size calculated assuming 1 device, but training may fail as PyTorch Lightning might not find any GPUs.")
        else: 
            warnings.warn("GPU selected, but CUDA not available. Accumulation batch size calculated assuming 1 device, but training may fail as PyTorch Lightning might not find any GPUs.")
    
    global_batch_per_step = micro_batch_size * num_devices_for_grad_accum
    if global_batch_per_step <= 0:
        warnings.warn(
            f"Calculated global batch per step ({global_batch_per_step}) is non-positive "
            f"(micro_batch_size: {micro_batch_size}, num_devices: {num_devices_for_grad_accum}). "
            f"Setting accumulate_grad_batches to 1."
        )
        accumulate_grad_batches = 1
    else:
        if total_batch_size % global_batch_per_step != 0:
            warnings.warn(
                f"Total batch size {total_batch_size} is not perfectly divisible by "
                f"global_batch_per_step ({global_batch_per_step} = micro_batch_size {micro_batch_size} * "
                f"num_devices {num_devices_for_grad_accum}). "
                f"Effective batch size might differ slightly due to integer division for accumulation."
            )
        accumulate_grad_batches = total_batch_size // global_batch_per_step

    if accumulate_grad_batches <= 0:
        warnings.warn(
            f"Calculated accumulate_grad_batches is {accumulate_grad_batches} (total_batch_size={total_batch_size}, "
            f"micro_batch_size={micro_batch_size}, num_devices={num_devices_for_grad_accum}). "
            f"Setting it to 1. This might indicate an issue with batch size configuration."
        )
        accumulate_grad_batches = 1

    total_num_steps = VAEIMERGPLModule.get_total_num_steps(num_samples=dm.num_train_samples,
                                                            total_batch_size=total_batch_size,
                                                            epoch=max_epochs)
    pl_module = VAEIMERGPLModule(total_num_steps=total_num_steps,
                                 accumulate_grad_batches=accumulate_grad_batches,
                                 save_dir=args.save,
                                 oc_file=args.cfg)

    trainer_kwargs = pl_module.set_trainer_kwargs(device=args.device)
    trainer = Trainer(**trainer_kwargs)

    if args.test:
        ckpt_path = None
        if args.ckpt_name is not None:
            if args.ckpt_name.endswith('.pt'):
                print("Loading pretrained .pt weights for testing")
                map_location = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
                pretrained_weights = torch.load(args.ckpt_name, map_location=map_location)
                model_dict = pl_module.torch_nn_module.state_dict()
                model_dict.update(pretrained_weights)
                pl_module.torch_nn_module.load_state_dict(model_dict, strict=False)
            else:
                ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
    else:
        ckpt_path = None
        if args.ckpt_name is not None:
            if args.ckpt_name.endswith('.pt'):
                print("Loading pretrained .pt weights for training")
                map_location = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
                pretrained_weights = torch.load(args.ckpt_name, map_location=map_location)
                model_dict = pl_module.torch_nn_module.state_dict()
                model_dict.update(pretrained_weights)
                pl_module.torch_nn_module.load_state_dict(model_dict, strict=False)
            else:
                ckpt_path = args.ckpt_name
                if not os.path.exists(ckpt_path):
                    warnings.warn(f"ckpt {ckpt_path} does not exist! Starting training from epoch 0.")
                    ckpt_path = None

        trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)

        pl_ckpt = pl_load(path_or_url=trainer.checkpoint_callback.best_model_path,
                          map_location=torch.device("cpu"))
        state_dict = pl_ckpt["state_dict"] if "state_dict" in pl_ckpt else pl_ckpt
        vae_key = "torch_nn_module."
        vae_state_dict = OrderedDict()
        loss_key = "loss."
        loss_state_dict = OrderedDict()
        unexpected_dict = OrderedDict()
        for key, val in state_dict.items():
            if key.startswith(vae_key):
                vae_state_dict[key[len(vae_key):]] = val
            elif key.startswith(loss_key):
                loss_state_dict[key[len(loss_key):]] = val
            else:
                unexpected_dict[key] = val
        torch.save(vae_state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        torch.save(loss_state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_loss_state_dict_name))
        if len(unexpected_dict) > 0:
            torch.save(unexpected_dict, os.path.join(pl_module.save_dir, "checkpoints", "unexpected_dict.pt"))
        trainer.test(ckpt_path="best", datamodule=dm)

if __name__ == "__main__":
    main()