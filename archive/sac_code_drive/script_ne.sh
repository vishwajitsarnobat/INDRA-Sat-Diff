#!/bin/bash
set -e
exec > >(tee -a training_ne_log.txt) 2>&1

NUM_RUNS=1

echo "Activating environment..."
cd ~/nowcasting_venv
source .venv/bin/activate

cd ~/data_67/Nowcasting/vishwajit/PreDiff_ISRO

for ((i=1; i<=NUM_RUNS; i++)); do
    echo "=== NorthEast Region Run #$i ==="

    echo "Starting northeast alignment training..."
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$(pwd) uv run python3 scripts/alignment/imerg/train_imerg_avg_x.py \
      --cfg scripts/alignment/imerg/alignment_imerg_ne.yaml \
      --ckpt_name last.ckpt \
      --device gpu --save tmp_imerg_avg_x_ne

    echo "Copying alignment checkpoint..."
    cp experiments/tmp_imerg_avg_x_ne/checkpoints/imerg_alignment_avgx.pt models_ne/imerg_alignment_avgx.pt

    echo "Starting prediff training..."
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$(pwd) uv run python3 scripts/prediff/imerg/train_imerg_prediff.py \
      --cfg scripts/prediff/imerg/prediff_imerg_ne.yaml \
      --ckpt_name models_ne/imerg_earthformerunet.pt \
      --device gpu --save tmp_imerg_ne

    echo "Copying prediff checkpoint..."
    cp experiments/tmp_imerg_ne/checkpoints/imerg_earthformerunet.pt models_ne/imerg_earthformerunet.pt

    echo "=== Completed NorthEast Region Run #$i ==="
done

echo "All $NUM_RUNS northeast region training runs completed!"

