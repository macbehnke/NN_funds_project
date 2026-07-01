# Project Checklist

Use this before submitting.

## Code

- [ ] `python train.py` runs from a clean environment.
- [ ] `python evaluate.py --checkpoint checkpoints/lenet5_mnist_best.pt` runs.
- [ ] `python demo.py --sample-index 7` runs.
- [ ] `outputs/metrics.json` contains the final reported metrics.
- [ ] `outputs/evaluation.json` contains confusion matrix and class metrics.
- [ ] Final checkpoint is under 100 MB.

## Report

- [ ] Replace TODO metrics in `report/report_draft.md`.
- [ ] Add a confusion matrix or summary.
- [ ] Add examples of misclassified digits.
- [ ] Mention limitations honestly.
- [ ] Reference relevant course chapters.

## Slides

- [ ] Add final accuracy and runtime.
- [ ] Add architecture diagram.
- [ ] Add demo screenshot.
- [ ] Add 3-5 error examples.
- [ ] Rehearse 10-minute talk.

## GitHub / Cluster

- [ ] Commit source code, README, report, and slides.
- [ ] Do not commit `data/`.
- [ ] Do not commit large temporary outputs.
- [ ] Decide whether to commit the final checkpoint or attach it separately.
- [ ] Clone repo on cluster.
- [ ] Run `bash scripts/run_cluster.sh`.
