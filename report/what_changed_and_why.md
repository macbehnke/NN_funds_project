# What We Changed and Why

The project topic asks for LeNet-5 on MNIST, faithful to LeCun 1998, plus a comparison with modern practice. Our implementation is a faithful educational reproduction of the main LeNet-5 structure, but not a byte-for-byte historical reconstruction.

## What stayed close to LeNet-5

- Two convolutional feature extraction stages.
- Small 5x5 convolution kernels.
- Tanh activations.
- Average pooling.
- Fully connected layers with 120 and 84 hidden units.
- A compact model: 60,000 trainable parameters.

These choices are the core of the historical architecture and make the model easy to explain from CNN fundamentals.

## What changed

1. Input size

   The original LeNet-5 used 32x32 normalized digit images. MNIST images are 28x28, so the current implementation pads each side by 2 pixels before the first convolution.

2. Connectivity

   The original second convolution used partial connectivity between feature maps. We used dense convolution, which is the normal PyTorch implementation and is easier to reproduce.

3. Output layer

   The original system used Euclidean RBF output units. The current `lenet5` implementation also uses fixed Euclidean RBF output units and the paper's MAP-style penalty loss.

4. Optimizer

   The completed run used Adam with learning rate 0.001. The original paper did not use Adam, so the code now also supports a stricter `--optimizer sgd` run. Plain SGD is closer to the historical training setup, although it is still not an exact implementation of the paper's older per-parameter/second-order training procedure.

5. Data augmentation

   We did not use distortion-based augmentation. This keeps the baseline simple, but it probably explains part of the remaining gap to the paper's reported error.

## Result

The previous LeNet run is superseded by the historical implementation. The final pulled run reports:

- best validation accuracy: 98.64%
- test accuracy: 98.87%
- test error: 1.13%
- training time: 125.94 seconds on CPU

LeCun et al. report about 0.8-0.95% error for LeNet-5 depending on the run. Our error is within 0.18 percentage points of 0.95%, but not within 0.2 percentage points of 0.8%.

## Modern baseline

The repository includes a ResNet-18 baseline command:

```bash
python train.py --model resnet18 --epochs 5 --batch-size 128 --num-workers 2 --metrics-path outputs/resnet18_metrics.json
python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_mnist_best.pt --num-workers 2 --output outputs/resnet18_evaluation.json
```

This represents what we would try today: a deeper residual network with batch normalization. For MNIST, ResNet-18 is probably overpowered, but it is useful for explaining the historical difference between compact early CNNs and modern deep residual architectures.

The ResNet-18 baseline reached 99.18% test accuracy with 11,172,810 parameters and 726.28 seconds of CPU training. Historical LeNet-5 reached 98.87% test accuracy with 60,000 parameters and 125.94 seconds of CPU training.

## Historical LeNet-5 experiment

The default `lenet5` architecture now supports a stricter historical reproduction. It pads MNIST to 32x32, uses scaled tanh activations, trainable average subsampling, partial C3 connectivity, convolutional C5, F6 with 84 units, fixed +1/-1 RBF target vectors, and MAP-style penalty loss.

This lets the final project compare two systems:

- historical LeNet-5 reproduction;
- ResNet-18 modern baseline.
