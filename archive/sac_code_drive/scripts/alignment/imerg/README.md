## Commands for training
Run the following command to train knowledge alignment network on IMERG dataset.
Change the configurations in [corresponding cfg.yaml](cfg.yaml)
```bash
PYTHONPATH=$(pwd) python3 scripts/alignment/imerg/train_imerg_avg_x.py --cfg scripts/alignment/imerg/alignment_imerg_server.yaml --ckpt_name pretrained/alignment/pretrained_imerg_alignment_avg_x_cuboid_v1.pt
```
Run the tensorboard command to visualize the experiment records
```bash
cd ROOT_DIR/PreDiff
tensorboard --logdir ./experiments/tmp_imerg_avg_x/lightning_logs
```