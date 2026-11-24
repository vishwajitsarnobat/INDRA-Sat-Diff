## Configurations for training and inference
Change the configurations in [corresponding cfg.yaml](cfg.yaml)
You might consider modifying the following configurations according to your specific requirements:
- `trainer.check_val_every_n_epoch`: Run validation every `n` training epochs. Set a larger value for it if you want to alleviate the time costs in validation.
- `vis.eval_example_only`: If `true`, only data with indices in the `example_data_idx_list` will be evaluated. Set it to `false` if you want to evaluate the whole val/test set.
- `vis.eval_aligned`: If `true`, PreDiff-KA will be evaluated.
- `vis.eval_unaligned`: If `true`, PreDiff without knowledge alignment will be evaluated.
- `vis.num_samples_per_context`: Generate `n` samples for each context sequence.
- `model.align.alignment_type`: `null` by default means not to load the knowledge alignment module. Setting it to `avg_x` for knowledge alignment with anticipated future average intensity.
- `model.align.model_ckpt_path`: Point it to your own pretrained checkpoint if you want a custom knowledge alignment network.
- `model.vae.pretrained_ckpt_path`: Point it to your own pretrained checkpoint if you want a custom vae.

## Commands for training and inference
Run the following command to train PreDiff on IMERG dataset.
```bash
PYTHONPATH=$(pwd) python3 scripts/prediff/imerg/train_imerg_prediff.py --cfg scripts/prediff/imerg/prediff_imerg_server.yaml --ckpt_name pretrained/earthformerunet/pretrained_imerg_earthformerunet_v1.pt
```
Run the tensorboard command to visualize the experiment records
```bash
cd ROOT_DIR/PreDiff
tensorboard --logdir ./experiments/tmp_imerg_prediff/lightning_logs
```
