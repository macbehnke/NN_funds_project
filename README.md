# LeNet-5 on MNIST

Class project for neural networks fundamentals.

## Problem

The goal is to reproduce a classic convolutional neural network baseline for handwritten digit recognition. The model receives a 28x28 grayscale image from the MNIST dataset and predicts one of ten digit classes: 0-9.

This is intentionally a small, reproducible project: the full model trains end-to-end on CPU or a modest GPU, uses a public dataset downloaded by `torchvision`, and can be demonstrated from a saved checkpoint.

## Approach

We implement a LeNet-5 style CNN:

- convolution 1: `1 -> 6`, kernel `5x5`
- `tanh`
- average pooling
- convolution 2: `6 -> 16`, kernel `5x5`
- `tanh`
- average pooling
- fully connected layers: `400 -> 120 -> 84 -> 10`

The original LeNet-5 used 32x32 inputs and an RBF-style output layer. This project keeps the classical architecture shape but uses a standard linear classifier and cross-entropy loss, which is the modern PyTorch convention for multi-class classification.

## Dataset

Dataset: MNIST

- 60,000 training images
- 10,000 test images
- grayscale, 28x28 pixels
- 10 digit classes

The training script splits the 60,000 training images into:

- 55,000 train examples
- 5,000 validation examples

## Repository layout

```text
.
├── train.py
├── evaluate.py
├── demo.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── scripts/
│   └── run_cluster.sh
├── report/
│   └── report_draft.md
├── slides/
│   └── slide_outline.md
├── checkpoints/
└── outputs/
```

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train

Default run:

```bash
python train.py
```

CPU-only run:

```bash
python train.py --cpu
```

Cluster/GPU-friendly run:

```bash
python train.py --epochs 8 --batch-size 128 --num-workers 2
```

Training saves:

- best model checkpoint: `checkpoints/lenet5_mnist_best.pt`
- metrics file: `outputs/metrics.json`

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/lenet5_mnist_best.pt
```

This saves detailed test metrics to:

```text
outputs/evaluation.json
```

## Make report figures

After training and evaluation:

```bash
python make_figures.py
```

This creates:

- `outputs/training_curves.png`
- `outputs/confusion_matrix.png`
- `outputs/misclassified_examples.png`

## Demo

Predict one MNIST test image:

```bash
python demo.py --sample-index 7
```

Predict a custom image:

```bash
python demo.py --image path/to/digit.png
```

The demo saves a visualization to:

```text
outputs/demo_prediction.png
```

## Expected result

With the default configuration, the model should normally reach around 98-99% test accuracy after several epochs. The exact value depends on hardware, PyTorch version, and deterministic behavior of the environment.

Use the final value from `outputs/metrics.json` in the report and presentation.

## Reproducibility

Default random seed: `42`

The training script sets Python, NumPy, and PyTorch seeds and uses a deterministic validation split. The raw MNIST data is downloaded automatically into `data/`, which is ignored by git.

## Course chapter links

This project mainly connects to:

- CNN architectures and convolution/pooling layers
- training neural networks with gradient descent
- loss functions for classification
- experiments, validation, and test evaluation
- PyTorch training loops

## Known limitations

- MNIST is clean and much easier than real handwriting from phones or scanned documents.
- The demo image preprocessing is simple and may fail on unusual backgrounds or off-center digits.
- The model is intentionally small and classical; it is not meant to be state of the art.
