# What We Changed and Why

The project topic asks for LeNet-5 on MNIST, faithful to LeCun 1998, plus a comparison with modern practice. Our implementation is a faithful educational reproduction of the main LeNet-5 structure, but not a byte-for-byte historical reconstruction.

## What stayed close to LeNet-5

- Two convolutional feature extraction stages.
- Small 5x5 convolution kernels.
- Tanh activations.
- Average pooling.
- Fully connected layers with 120 and 84 hidden units.
- A compact model: 61,706 trainable parameters.

These choices are the core of the historical architecture and make the model easy to explain from CNN fundamentals.

## What changed

1. Input size

   The original LeNet-5 used 32x32 normalized digit images. MNIST images are 28x28. We used the 28x28 images directly and padded only in the first convolution to preserve a clean feature-map shape.

2. Connectivity

   The original second convolution used partial connectivity between feature maps. We used dense convolution, which is the normal PyTorch implementation and is easier to reproduce.

3. Output layer

   The original system used a Gaussian/RBF-style output. We used a linear classifier with cross-entropy loss. This is the modern default for multi-class neural classification.

4. Optimizer

   We used Adam with learning rate 0.001. The original paper used an older training setup. Adam gives stable convergence in a short class-project run.

5. Data augmentation

   We did not use distortion-based augmentation. This keeps the baseline simple, but it probably explains part of the remaining gap to the paper's reported error.

## Result

Our final run reached:

- best validation accuracy: 98.68%
- test accuracy: 98.83%
- test error: 1.17%
- training time: 16.81 seconds on CPU

LeCun et al. report about 0.8% error for LeNet-5. Our reproduction is close, but the gap is 0.37 percentage points, so we should not claim that we matched the paper within 0.2 percentage points.

## Modern baseline

The repository includes a ResNet-18 baseline command:

```bash
python train.py --model resnet18 --epochs 5 --batch-size 128 --num-workers 2 --metrics-path outputs/resnet18_metrics.json
python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_mnist_best.pt --num-workers 2 --output outputs/resnet18_evaluation.json
```

This represents what we would try today: a deeper residual network with batch normalization. For MNIST, ResNet-18 is probably overpowered, but it is useful for explaining the historical difference between compact early CNNs and modern deep residual architectures.

The final baseline reached 99.18% test accuracy, compared with 98.83% for LeNet-5. The cost is large: 11,172,810 parameters instead of 61,706, and 726.28 seconds of CPU training instead of 16.81 seconds. This makes the comparison more interesting than simply saying "ResNet is better": it is better on accuracy, but much less efficient for this task.
