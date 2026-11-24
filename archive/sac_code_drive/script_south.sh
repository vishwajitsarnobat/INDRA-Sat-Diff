#!/bin/bash
set -e
exec > >(tee -a training_south_log.txt) 2>&1

NUM_RUNS=1

echo "Activating environment..."
cd ~/nowcasting_venv
source .venv/bin/activate

cd ~/data_67/Nowcasting/vishwajit/PreDiff_ISRO

for ((i=1; i<=NUM_RUNS; i++)); do
    echo "=== South Region Run #$i ==="

    echo "Starting prediff training..."
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$(pwd) uv run python3 scripts/prediff/imerg/train_imerg_prediff.py \
      --cfg scripts/prediff/imerg/prediff_imerg_south.yaml \
      --ckpt_name last-v10.ckpt \
      --device gpu --save tmp_imerg_south

    echo "Copying prediff checkpoint..."
    cp experiments/tmp_imerg_south/checkpoints/imerg_earthformerunet.pt models_south/imerg_earthformerunet.pt

    echo "=== Completed South Region Run #$i ==="
done

echo "All $NUM_RUNS south region training runs completed!"

