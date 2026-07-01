# LeNet-5 on MNIST: Reproducing a Classic CNN Baseline

## 1. Motivation

Handwritten digit recognition is one of the classical problems that made convolutional neural networks practical and visible. The goal of this project is to build a clean, reproducible LeNet-5 style classifier for MNIST. The emphasis is not novelty, but correct engineering: a small dataset, a small model, deterministic training, clear evaluation, and an honest explanation of what works and what does not.

MNIST is also a good teaching problem because the input is visual, the task is understandable to non-specialists, and the model can be trained quickly on CPU or a modest GPU.

## 2. Related Work

LeNet-5 was introduced by LeCun et al. for document recognition and became one of the reference architectures for early convolutional neural networks. Its key idea is to exploit local spatial structure in images: nearby pixels are processed by shared convolution filters, and pooling provides limited translation robustness.

Compared with fully connected neural networks, CNNs use far fewer parameters for image inputs and learn local visual features such as strokes, curves, and digit parts. Modern CNNs use different activations, batch normalization, residual connections, and larger datasets, but LeNet remains a compact baseline for explaining the core idea.

The 1998 paper reports LeNet-5 at about 0.8% error on the MNIST test set. Our reproduction reaches 1.17% error. This is close for a short, simple run, but it does not meet the stricter wow target of matching within 0.2 percentage points.

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

The model has 61,706 trainable parameters.

The original 1998 LeNet-5 used 32x32 inputs, trainable average-pooling/subsampling, partial connectivity between some feature maps, and a final RBF-style classifier. This project keeps the visible layer sizes and the tanh plus average-pooling design, but uses a standard fully connected output layer with cross-entropy loss. That choice makes the experiment simpler, easier to reproduce in PyTorch, and easier to compare with a modern classifier.

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

| Metric | Value |
|---|---:|
| Parameters | 61,706 |
| Epochs | 8 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Device | CPU |
| Best validation accuracy | 98.68% |
| Test accuracy | 98.83% |
| Test error | 1.17% |
| Test loss | 0.0374 |
| Training time | 16.81 seconds |

The model fits the course feasibility constraint comfortably: it trains end-to-end in far less than one hour, even on CPU.

Comparison to the paper:

| System | Test error |
|---|---:|
| LeNet-5 reported by LeCun et al. | about 0.8% |
| Our LeNet-5 style reproduction | 1.17% |
| Gap | 0.37 percentage points |

This is a good practical reproduction, but it should not be presented as matching the paper within 0.2 percentage points. The likely reasons are the simplified output layer, no distortion-based data augmentation, only 8 epochs, and small implementation differences from the original training setup.

## 7. Error Analysis

Common MNIST mistakes usually happen between visually similar digits:

- 4 and 9
- 3 and 5
- 7 and 9
- 1 and 7
- 8 and 3

The generated figure `outputs/misclassified_examples.png` should be included in the final report or slides. The confusion matrix in `outputs/confusion_matrix.png` shows that most classes are recognized reliably; the remaining errors are sparse and visually plausible.

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

## 10. What We Changed and Why

The goal was to reproduce the spirit and main structure of LeNet-5 while keeping the project simple enough for a clean class submission.

What we kept from the 1998 design:

- two convolutional stages;
- tanh activations;
- average pooling rather than max pooling;
- fully connected layers of sizes 120 and 84;
- a very small parameter count compared with modern CNNs.

What we changed:

- We used 28x28 MNIST images directly instead of padding them to 32x32.
- We used a dense second convolution instead of the original partial feature-map connectivity.
- We used a standard linear output layer and cross-entropy loss instead of the original RBF-style output.
- We trained with Adam instead of the original older optimization setup.
- We did not use distortion-based data augmentation.

Why these changes are reasonable:

- Cross-entropy with logits is the standard modern formulation for multi-class classification.
- Dense convolution is simpler to explain and avoids custom connectivity code that would add little educational value.
- Adam gives stable convergence in a short run.
- Avoiding augmentation keeps the experiment focused on the baseline architecture.

Why these changes matter:

- They make the project easier to reproduce, but they also mean the result is not an exact historical reproduction.
- The 1.17% test error is close to the paper's 0.8%, but the gap is large enough that we should call this a LeNet-5 style reproduction, not a perfect replication.

## 11. Modern Baseline: ResNet-18

The repository includes an optional ResNet-18 baseline:

```bash
python train.py --model resnet18 --epochs 5 --batch-size 128 --num-workers 2 --metrics-path outputs/resnet18_metrics.json
python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_mnist_best.pt --num-workers 2 --output outputs/resnet18_evaluation.json
```

This baseline represents what we might try today: residual blocks, deeper features, and batch normalization. It is adapted for MNIST by changing the first convolution to accept one-channel images and by removing the initial max-pooling layer so the 28x28 images are not downsampled too aggressively.

## 12. Limitations and Future Work

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

## 13. References

- Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 1998.
- MNIST database of handwritten digits.
- PyTorch documentation for `torch`, `torchvision`, and `torch.nn`.
