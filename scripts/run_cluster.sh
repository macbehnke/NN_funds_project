#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python train.py --model lenet5 --epochs 20 --batch-size 128 --num-workers 2
python evaluate.py --checkpoint checkpoints/lenet5_mnist_best.pt --num-workers 2
python make_figures.py --num-workers 2
python demo.py --sample-index 7

# Optional modern baseline for the presentation:
# python train.py --model resnet18 --epochs 5 --batch-size 128 --num-workers 2 --metrics-path outputs/resnet18_metrics.json
# python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_mnist_best.pt --num-workers 2 --output outputs/resnet18_evaluation.json
