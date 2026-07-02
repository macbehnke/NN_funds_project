# LeNet-5 on MNIST: Reproducing a Classic CNN Baseline

## 1. Motivation

Handwritten digit recognition is one of the classical problems that made convolutional neural networks practical and visible. The goal of this project is to build a clean, reproducible LeNet-5 style classifier for MNIST. The emphasis is not novelty, but correct engineering: a small dataset, a small model, deterministic training, clear evaluation, and an honest explanation of what works and what does not.

MNIST is also a good teaching problem because the input is visual, the task is understandable to non-specialists, and the model can be trained quickly on CPU or a modest GPU.

## 2. Related Work

LeNet-5 was introduced by LeCun et al. for document recognition and became one of the reference architectures for early convolutional neural networks. Its key idea is to exploit local spatial structure in images: nearby pixels are processed by shared convolution filters, and pooling provides limited translation robustness.

Compared with fully connected neural networks, CNNs use far fewer parameters for image inputs and learn local visual features such as strokes, curves, and digit parts. Modern CNNs use different activations, batch normalization, residual connections, and larger datasets, but LeNet remains a compact baseline for explaining the core idea.

The 1998 paper reports LeNet-5 at about 0.8-0.95% error on the MNIST test set depending on the run. Our historical reproduction reaches 1.07% error, which is close to the 0.95% result but not equal to the best reported 0.8% result.

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

The first implemented model is a LeNet-5 style CNN:

```text
Input: 1 x 32 x 32
C1: convolution, 1 -> 6 maps, kernel 5, scaled tanh
S2: trainable average subsampling, 6 maps
C3: convolution, 6 -> 16 maps, kernel 5, partial connectivity, scaled tanh
S4: trainable average subsampling, 16 maps
C5: convolution, 16 -> 120 maps, kernel 5, scaled tanh
F6: fully connected, 120 -> 84, scaled tanh
Output: fixed Euclidean RBF units, 84 -> 10 class penalties
```

The historical LeNet-5 model has exactly 60,000 trainable parameters, matching the count stated in the paper.

The implemented `lenet5` model follows the historical LeNet-5 architecture:

- MNIST is padded from 28x28 to 32x32.
- Activations use the scaled tanh form from the LeNet paper.
- S2 and S4 are trainable average-subsampling layers.
- C3 uses the historical partial feature-map connectivity table.
- C5 is implemented as a 5x5 convolution producing 120 feature maps.
- F6 has 84 units.
- The output layer uses fixed Euclidean RBF units with +1/-1 prototype vectors.
- Training uses the paper's MAP-style discriminative loss over class penalties.

## 5. Training Details

Default configuration:

- optimizer: SGD
- learning rate: 0.01
- loss: LeNet-5 MAP-style RBF penalty loss
- batch size: 128
- epochs: 20
- seed: 42
- validation size: 5,000 images

The training script saves the checkpoint with the best validation accuracy. It also writes a `metrics.json` file containing training history, final test accuracy, and runtime.

The final reported run should be performed using:

```bash
python train.py --model lenet5 --optimizer sgd --lr 0.01 --epochs 20 --batch-size 128 --num-workers 2
python evaluate.py --model lenet5 --checkpoint checkpoints/lenet5_mnist_best.pt --num-workers 2 --output outputs/evaluation.json
```

If the cluster has no GPU available, use:

```bash
python train.py --model lenet5 --optimizer sgd --lr 0.01 --epochs 20 --batch-size 128 --num-workers 2
```

## 6. Results

| Metric | Value |
|---|---:|
| Parameters | 60,000 |
| Epochs | 20 |
| Batch size | 128 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Device | CPU |
| Best validation accuracy | 98.94% |
| Test accuracy | 98.93% |
| Test error | 1.07% |
| Test loss | 0.6722 |
| Training time | 146.91 seconds |

The model fits the course feasibility constraint comfortably: it trains end-to-end in far less than one hour, even on CPU.

Comparison to the paper:

| System | Test error |
|---|---:|
| LeNet-5 reported by LeCun et al. | about 0.8-0.95% |
| Our historical LeNet-5 reproduction | 1.07% |
| Gap to 0.95% result | 0.12 percentage points |
| Gap to 0.8% result | 0.27 percentage points |

This is a defensible reproduction result: it is within 0.2 percentage points of the 0.95% historical run, but not within 0.2 percentage points of the strongest 0.8% result.

## 7. Error Analysis

Common MNIST mistakes usually happen between visually similar digits:

- 4 and 9
- 3 and 5
- 7 and 9
- 1 and 7
- 8 and 3

The generated figure `outputs/misclassified_examples.png` should be included in the final report or slides. The confusion matrix in `outputs/confusion_matrix.png` shows that most classes are recognized reliably; the remaining errors are sparse and visually plausible.

## 8. Demo

The project has two demos. The command-line demo loads the saved checkpoint and predicts either:

- a selected MNIST test example, or
- a user-provided image file

Example:

```bash
python demo.py --sample-index 7
```

This produces `outputs/demo_prediction.png`, which can be used in the presentation.

The stronger presentation demo is the local browser app:

```bash
python app.py
```

The app opens at `http://127.0.0.1:5000`. A user draws a digit in the browser canvas, clicks `Classify`, and sees the predicted digit with probability bars. This is the main wow component because it turns the trained model into an interactive artifact a non-specialist can try.

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

- We pad 28x28 MNIST images to 32x32 as in the historical architecture.
- We reproduce the partial C3 feature-map connectivity.
- We use fixed Euclidean RBF output units instead of a modern linear classifier.
- The completed run used SGD instead of Adam, which is closer to the older optimization setup than our first completed run.
- We did not use distortion-based data augmentation.

Why these changes are reasonable:

- The historical RBF output gives a closer reproduction than the modern softmax classifier.
- SGD keeps the training setup closer to the historical experiment while preserving the historical architecture and output formulation.
- Avoiding augmentation keeps the experiment focused on the baseline architecture.

Why these changes matter:

- The exact 1998 per-parameter/second-order optimizer and distortion training are still not reproduced.
- The 1.07% test error is close to the paper's 0.95% run, but not to the strongest 0.8% figure.

## 11. Modern Baseline: ResNet-18

The repository includes a ResNet-18 baseline:

```bash
python train.py --model resnet18 --epochs 5 --batch-size 128 --num-workers 2 --metrics-path outputs/resnet18_metrics.json
python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_mnist_best.pt --num-workers 2 --output outputs/resnet18_evaluation.json
```

This baseline represents what we might try today: residual blocks, deeper features, and batch normalization. It is adapted for MNIST by changing the first convolution to accept one-channel images and by removing the initial max-pooling layer so the 28x28 images are not downsampled too aggressively.

Final comparison:

| Metric | LeNet-5 | ResNet-18 |
|---|---:|---:|
| Parameters | 60,000 | 11,172,810 |
| Epochs | 20 | 5 |
| Device | CPU | CPU |
| Optimizer | SGD | Adam |
| Training time | 146.91 s | 726.28 s |
| Best validation accuracy | 98.94% | 98.88% |
| Test accuracy | 98.93% | 99.18% |
| Test error | 1.07% | 0.82% |
| Checkpoint size | 0.26 MB | 44.77 MB |

ResNet-18 improves test accuracy by 0.25 percentage points, but uses about 186 times more trainable parameters and takes about 4.9 times longer in our CPU run. This makes the comparison more interesting than simply saying that the modern model is better: historical LeNet-5 is almost as accurate on MNIST while being much smaller and easier to explain.

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
- improve the browser preprocessing to handle dark strokes on white paper photos.

## 13. References

- Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 1998.
- MNIST database of handwritten digits.
- PyTorch documentation for `torch`, `torchvision`, and `torch.nn`.
