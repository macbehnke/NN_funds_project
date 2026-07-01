# LeNet-5 on MNIST: Reproducing a Classic CNN Baseline

## 1. Motivation

Handwritten digit recognition is one of the classical problems that made convolutional neural networks practical and visible. The goal of this project is to build a clean, reproducible LeNet-5 style classifier for MNIST. The emphasis is not novelty, but correct engineering: a small dataset, a small model, deterministic training, clear evaluation, and an honest explanation of what works and what does not.

MNIST is also a good teaching problem because the input is visual, the task is understandable to non-specialists, and the model can be trained quickly on CPU or a modest GPU.

## 2. Related Work

LeNet-5 was introduced by LeCun et al. for document recognition and became one of the reference architectures for early convolutional neural networks. Its key idea is to exploit local spatial structure in images: nearby pixels are processed by shared convolution filters, and pooling provides limited translation robustness.

Compared with fully connected neural networks, CNNs use far fewer parameters for image inputs and learn local visual features such as strokes, curves, and digit parts. Modern CNNs use different activations, batch normalization, residual connections, and larger datasets, but LeNet remains a compact baseline for explaining the core idea.

## 3. Dataset

Dataset: MNIST.

MNIST contains 70,000 grayscale images of handwritten digits:

- 60,000 official training images
- 10,000 official test images
- 10 classes, digits 0-9
- each image is 28x28 pixels

For model selection, the training set is split into:

- 55,000 training examples
- 5,000 validation examples

The test set is used only at the end. Images are normalized using the standard MNIST mean and standard deviation.

## 4. Model Architecture

The implemented model is a LeNet-5 style CNN:

```text
Input: 1 x 28 x 28
Conv2d: 1 -> 6, kernel 5, padding 2
Tanh
Average pooling: 2 x 2
Conv2d: 6 -> 16, kernel 5
Tanh
Average pooling: 2 x 2
Flatten
Linear: 400 -> 120
Tanh
Linear: 120 -> 84
Tanh
Linear: 84 -> 10
```

The model has approximately 61k trainable parameters.

The original 1998 LeNet-5 used 32x32 inputs, trainable subsampling, and a final RBF-style classifier. This project keeps the classical convolutional structure but uses a modern linear output layer with cross-entropy loss, which is the standard PyTorch approach for multi-class classification.

## 5. Training Details

Default configuration:

- optimizer: Adam
- learning rate: 0.001
- loss: cross-entropy
- batch size: 128
- epochs: 8
- seed: 42
- validation size: 5,000 images

The training script saves the checkpoint with the best validation accuracy. It also writes a `metrics.json` file containing training history, final test accuracy, and runtime.

The final reported run should be performed using:

```bash
python train.py --epochs 8 --batch-size 128 --num-workers 2
python evaluate.py --checkpoint checkpoints/lenet5_mnist_best.pt
```

If the cluster has no GPU available, use:

```bash
python train.py --cpu --epochs 8 --batch-size 128
```

## 6. Results

Replace this section after the final run with values from `outputs/metrics.json` and `outputs/evaluation.json`.

Suggested table:

| Metric | Value |
|---|---:|
| Parameters | TODO |
| Best validation accuracy | TODO |
| Test accuracy | TODO |
| Test loss | TODO |
| Training time | TODO |

Expected test accuracy is around 98-99% after several epochs.

## 7. Error Analysis

After running `evaluate.py`, inspect the confusion matrix in `outputs/evaluation.json`.

Common MNIST mistakes usually happen between visually similar digits:

- 4 and 9
- 3 and 5
- 7 and 9
- 1 and 7
- 8 and 3

The report should include a few examples of misclassified images. For each one, explain whether the mistake is understandable from the image shape, stroke thickness, centering, or ambiguity.

## 8. Demo

The demo script loads the saved checkpoint and predicts either:

- a selected MNIST test example, or
- a user-provided image file

Example:

```bash
python demo.py --sample-index 7
```

This produces `outputs/demo_prediction.png`, which can be used in the presentation.

## 9. Lessons Learned

Important points to discuss:

- CNNs are effective for images because convolution uses local spatial structure.
- Pooling makes the model less sensitive to small shifts.
- A small architecture can solve a clean benchmark very well.
- Validation accuracy is useful for checkpoint selection.
- MNIST performance does not guarantee robustness on real phone photos or unusual handwriting.

## 10. Limitations and Future Work

Limitations:

- The dataset is clean and centered.
- Custom image preprocessing is simple.
- The model is not tested on real handwritten samples collected by the team.

Possible extensions:

- collect team handwriting samples and test domain shift;
- compare `tanh` with `ReLU`;
- compare average pooling with max pooling;
- add an adversarial example demo using FGSM;
- build a small Streamlit or browser demo.

## References

- Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 1998.
- MNIST database of handwritten digits.
- PyTorch documentation for `torch`, `torchvision`, and `torch.nn`.
