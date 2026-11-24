#!/bin/bash
set -e
exec > >(tee -a training_log.txt) 2>&1

NUM_RUNS=1

echo "Activating environment..."
cd ~/nowcasting_venv
source .venv/bin/activate

cd ~/data_67/Nowcasting/vishwajit/PreDiff_ISRO

for ((i=1; i<=NUM_RUNS; i++)); do
    echo "=== Run #$i ==="

    echo "Starting alignment training..."
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$(pwd) uv run python3 scripts/alignment/imerg/train_imerg_avg_x.py \
        --cfg scripts/alignment/imerg/alignment_imerg_server.yaml \
        --ckpt_name models/imerg_alignment_avgx.pt --device gpu

    echo "Copying alignment checkpoint..."
    cp experiments/tmp_imerg_avg_x/checkpoints/imerg_alignment_avgx.pt models/imerg_alignment_avgx.pt

    echo "Starting prediff training..."
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$(pwd) uv run python3 scripts/prediff/imerg/train_imerg_prediff.py \
        --cfg scripts/prediff/imerg/prediff_imerg_server.yaml \
        --ckpt_name models/imerg_earthformerunet.pt --device gpu

    echo "Copying prediff checkpoint..."
    cp experiments/tmp_imerg/checkpoints/imerg_earthformerunet.pt models/imerg_earthformerunet.pt

    echo "=== Completed run #$i ==="
done

echo "All $NUM_RUNS training runs completed!"

