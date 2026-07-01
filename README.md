# LeNet-5 on MNIST

Class project for neural networks fundamentals.

## Problem

The goal is to reproduce a classic convolutional neural network baseline for handwritten digit recognition. The model receives a 28x28 grayscale image from the MNIST dataset and predicts one of ten digit classes: 0-9.

This is intentionally a small, reproducible project: the full model trains end-to-end on CPU or a modest GPU, uses a public dataset downloaded by `torchvision`, and can be demonstrated from a saved checkpoint.

## Final result

Historical LeNet-5 final run:

| Metric | Value |
|---|---:|
| Model | Historical LeNet-5 with RBF output |
| Parameters | 60,000 |
| Epochs | TODO after cluster run |
| Device used | TODO after cluster run |
| Training time | TODO after cluster run |
| Best validation accuracy | TODO after cluster run |
| Test accuracy | TODO after cluster run |
| Test error | TODO after cluster run |

LeCun et al. reported about 0.8-0.95% test error for LeNet-5 on MNIST depending on the specific run and training setting. The historical model must be rerun after this architecture update before reporting the final comparison.

## Approach

We implement a LeNet-5 style CNN:

- convolution 1: `1 -> 6`, kernel `5x5`
- `tanh`
- average pooling
- convolution 2: `6 -> 16`, kernel `5x5`
- `tanh`
- average pooling
- C5 convolution with 120 maps, F6 with 84 units, and fixed RBF output prototypes

The implemented `lenet5` model is the historical LeNet-5 reproduction path: 32x32 padded inputs, scaled tanh activations, trainable average subsampling, partial C3 connectivity, convolutional C5, F6 with 84 units, and fixed Euclidean RBF output units with the paper's MAP-style discriminative loss.

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
├── app.py
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

## Train LeNet-5

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

## Train historical LeNet-5

The default `lenet5` model is now the historical reproduction:

- pads MNIST from 28x28 to 32x32;
- uses scaled tanh activations;
- uses trainable average subsampling layers;
- uses the partial C3 connectivity table;
- uses a convolutional C5 layer with 120 maps;
- uses fixed Euclidean RBF output units and MAP-style loss.

Run:

```bash
python train.py --model lenet5 --epochs 20 --batch-size 128 --num-workers 2 --metrics-path outputs/metrics.json
python evaluate.py --model lenet5 --checkpoint checkpoints/lenet5_mnist_best.pt --num-workers 2 --output outputs/evaluation.json
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/lenet5_mnist_best.pt
```

This saves detailed test metrics to:

```text
outputs/evaluation.json
```

## Optional ResNet-18 baseline

The project also includes a modern baseline path:

```bash
python train.py --model resnet18 --epochs 5 --batch-size 128 --num-workers 2 --metrics-path outputs/resnet18_metrics.json
python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_mnist_best.pt --num-workers 2 --output outputs/resnet18_evaluation.json
```

This adapts ResNet-18 to one-channel 28x28 MNIST images by replacing the first convolution and removing the initial max-pooling layer.

Final ResNet-18 baseline run:

| Metric | LeNet-5 | ResNet-18 |
|---|---:|---:|
| Parameters | 60,000 | 11,172,810 |
| Epochs | TODO | 5 |
| Device used | TODO | CPU |
| Training time | TODO | 726.28 s |
| Best validation accuracy | TODO | 98.88% |
| Test accuracy | TODO | 99.18% |
| Test error | TODO | 0.82% |
| Checkpoint size | TODO | 44.77 MB |

The modern ResNet-18 baseline is already trained. The historical LeNet-5 column should be filled from the next cluster run.

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

## Browser drawing demo

Run the local web app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Draw a digit in the browser and click `Classify`. The app sends the canvas image to the trained LeNet-5 checkpoint, preprocesses it into MNIST format, and returns the predicted digit plus class probabilities.

## Reproduce final result

The committed final run produced `outputs/metrics.json`, `outputs/evaluation.json`, figures in `outputs/`, and `checkpoints/lenet5_mnist_best.pt`. Re-running training should normally produce a result around 98-99% test accuracy, though exact values can differ slightly by environment.

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
- The browser drawing demo works best with MNIST-style white strokes on a black background.
- The model is intentionally small and classical; it is not meant to be state of the art.
