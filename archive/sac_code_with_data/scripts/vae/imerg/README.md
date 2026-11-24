## Commands for training
Run the following command to train VAE on IMERG frames.
Note: vgg16-397923af.pth will be downloaded if ran for the first time 
Change the configurations in [corresponding cfg.yaml](cfg.yaml)
```bash
PYTHONPATH=$(pwd) python3 scripts/vae/imerg/train_vae_imerg.py --cfg scripts/vae/imerg/vae_imerg_server.yaml --ckpt_name pretrained/vae/pretrained_imerg_vae_8x8x64_v1.pt
```
Run the tensorboard command to visualize the experiment records
```bash
cd ROOT_DIR/PreDiff
tensorboard --logdir ./experiments/tmp_vae_imerg/lightning_logs
```