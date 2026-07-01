$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python train.py --epochs 8 --batch-size 128 --num-workers 0
python evaluate.py --checkpoint checkpoints/lenet5_mnist_best.pt --num-workers 0
python make_figures.py --num-workers 0
python demo.py --sample-index 7
