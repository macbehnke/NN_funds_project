#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python train.py \
  --model lenet5 \
  --optimizer sgd \
  --lr-schedule lecun98 \
  --epochs 20 \
  --batch-size 128 \
  --num-workers 2 \
  --checkpoint-dir checkpoints/lecun98_schedule \
  --metrics-path outputs/lecun98_schedule_metrics.json

python evaluate.py \
  --model lenet5 \
  --checkpoint checkpoints/lecun98_schedule/lenet5_mnist_best.pt \
  --num-workers 2 \
  --output outputs/lecun98_schedule_evaluation.json

python make_figures.py \
  --metrics outputs/lecun98_schedule_metrics.json \
  --evaluation outputs/lecun98_schedule_evaluation.json \
  --checkpoint checkpoints/lecun98_schedule/lenet5_mnist_best.pt \
  --output-dir outputs/lecun98_schedule \
  --num-workers 2
