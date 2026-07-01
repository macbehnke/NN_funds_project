#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python train.py --epochs 8 --batch-size 128 --num-workers 2
python evaluate.py --checkpoint checkpoints/lenet5_mnist_best.pt --num-workers 2
python make_figures.py --num-workers 2
python demo.py --sample-index 7
