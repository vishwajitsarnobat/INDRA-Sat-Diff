#!/bin/bash

# Run south region test on GPU 2 and log output
CUDA_VISIBLE_DEVICES=2 uv run -m scripts.test.prediff_test \
  --region south \
  --shapefile Shapefile/states_IND \
  --cfg scripts/prediff/imerg/prediff_imerg_south.yaml \
  --ckpt experiments/tmp_imerg_south/checkpoints/last-v10.ckpt \
  &> prediff_south.log &

# Run northeast region test on GPU 3 and log output
CUDA_VISIBLE_DEVICES=3 uv run -m scripts.test.prediff_test \
  --region ne \
  --shapefile Shapefile/states_IND\
  --cfg scripts/prediff/imerg/prediff_imerg_ne.yaml \
  --ckpt experiments/tmp_imerg_ne/checkpoints/last-v5.ckpt \
  &> prediff_ne.log &

# Wait for all background jobs to finish
wait

echo "Both processes finished."
