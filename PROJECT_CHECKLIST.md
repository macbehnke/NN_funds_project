# Project Checklist

Use this before submitting.

## Code

- [x] `python train.py` runs from a clean environment.
- [x] `python evaluate.py --checkpoint checkpoints/lenet5_mnist_best.pt` runs.
- [x] `python demo.py --sample-index 7` runs.
- [x] `python app.py` runs and the browser drawing demo classifies a digit.
- [x] `outputs/metrics.json` contains the final reported metrics.
- [x] `outputs/evaluation.json` contains confusion matrix and class metrics.
- [x] Final checkpoint is under 100 MB.

## Report

- [x] Replace TODO metrics in `report/report_draft.md`.
- [x] Add a confusion matrix or summary.
- [x] Add examples of misclassified digits.
- [x] Mention limitations honestly.
- [x] Reference relevant course chapters.

## Slides

- [x] Add final accuracy and runtime.
- [x] Add architecture diagram.
- [x] Add demo screenshot.
- [x] Demonstrate the browser drawing app live or record it as a fallback.
- [x] Add 3-5 error examples.
- [ ] Rehearse 10-minute talk.

## GitHub / Cluster

- [x] Commit source code, README, report, and slides.
- [ ] Do not commit `data/`.
- [ ] Do not commit large temporary outputs.
- [x] Decide whether to commit the final checkpoint or attach it separately.
- [x] Clone repo on cluster.
- [x] Run `bash scripts/run_cluster.sh`.
