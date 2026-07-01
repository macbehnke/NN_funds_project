# Slide Outline: LeNet-5 on MNIST

## 1. Title

LeNet-5 on MNIST: Reproducing a Classic CNN Baseline

Names, course, date.

## 2. Problem

- Input: 28x28 MNIST image padded to 32x32
- Output: digit class 0-9
- Goal: train a small CNN that is accurate, reproducible, and easy to demo

## 3. Why This Project

- Classic neural network benchmark
- Direct connection to CNN chapters
- Small enough to train under the course constraints
- Easy for non-specialists to understand

## 4. Dataset

- MNIST
- 60k train, 10k test
- 5k validation split from training data
- Show several example digits

## 5. Model Architecture

Show the pipeline:

```text
image -> conv -> tanh -> avg pool -> conv -> tanh -> avg pool -> FC -> FC -> prediction
```

Mention parameter count from final run.

## 6. Training Setup

- Adam optimizer
- LeNet-5 MAP-style RBF penalty loss
- batch size 128
- 8 epochs
- fixed random seed
- best checkpoint selected by validation accuracy

## 7. Results

Final metrics:

- parameters: 60,000
- best validation accuracy: 98.64%
- test accuracy: 98.87%
- test error: 1.13%
- training time: 125.94 seconds on CPU

Compare to LeNet-5 paper:

- reported LeNet-5 test error: about 0.8%
- our gap: 0.37 percentage points
- honest conclusion: close, but not within 0.2 percentage points

## 8. Error Analysis

Show 3-5 mistakes:

- true label
- predicted label
- confidence
- short explanation

## 9. Demo

Run:

```bash
python app.py
```

Open `http://127.0.0.1:5000`, draw a digit, and classify it live. Keep `python demo.py --sample-index 7` as the backup if the browser demo is not available.

## 10. What Worked

- Small CNN is enough for MNIST
- Convolutions learn useful stroke features
- Validation checkpointing was simple and reliable

## 11. What Did Not Work / Limitations

- MNIST is clean and centered
- Real photos need better preprocessing
- Accuracy alone does not prove robustness

## 12. Optional Wow Slide

Show one of:

- live browser drawing demo from `app.py`
- training curves from `outputs/training_curves.png`
- confusion matrix from `outputs/confusion_matrix.png`
- misclassified examples from `outputs/misclassified_examples.png`
- ResNet-18 baseline comparison: 99.18% test accuracy, but 11.17M parameters and 726.28 s CPU training time

## 13. Conclusion

- Reproduced a classic CNN baseline
- Built clean train/evaluate/demo pipeline
- Historical LeNet-5 has exactly 60,000 trainable parameters
- Final historical LeNet-5 test accuracy is 98.87%
- Explained differences from the 1998 LeNet-5 design

## 14. Q&A

Backup details:

- exact architecture
- confusion matrix
- training curve
- reproducibility settings
