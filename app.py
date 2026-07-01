from __future__ import annotations

import argparse
from pathlib import Path

import torch
from flask import Flask, jsonify, request

from src.model import LeNet5
from src.preprocess import preprocess_canvas_data_url
from src.utils import get_device


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeNet-5 MNIST Demo</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #f5f5f2;
      color: #111;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    main {
      width: min(1040px, calc(100vw - 36px));
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 32px;
      align-items: center;
    }
    h1 {
      font-size: 44px;
      line-height: 1.05;
      margin: 0 0 14px;
      letter-spacing: 0;
    }
    p {
      color: #555;
      font-size: 18px;
      line-height: 1.45;
      margin: 0 0 22px;
    }
    .canvas-wrap {
      background: #111;
      padding: 14px;
      border: 1px solid #111;
    }
    canvas {
      display: block;
      width: 360px;
      height: 360px;
      background: #000;
      touch-action: none;
      cursor: crosshair;
    }
    .actions {
      display: flex;
      gap: 12px;
      margin-top: 16px;
    }
    button {
      border: 1px solid #111;
      background: #111;
      color: white;
      font-size: 17px;
      font-weight: 700;
      padding: 12px 18px;
      cursor: pointer;
    }
    button.secondary {
      background: white;
      color: #111;
    }
    .result {
      background: white;
      border: 1px solid #d6d6d2;
      padding: 26px;
      min-height: 260px;
    }
    .digit {
      font-size: 112px;
      font-weight: 800;
      line-height: 1;
      margin: 8px 0 8px;
    }
    .confidence {
      font-size: 24px;
      font-weight: 700;
      color: #2f7d55;
      margin-bottom: 22px;
    }
    .bars {
      display: grid;
      gap: 7px;
    }
    .bar {
      display: grid;
      grid-template-columns: 28px 1fr 54px;
      gap: 10px;
      align-items: center;
      font-size: 14px;
      color: #444;
    }
    .fill {
      height: 9px;
      background: #e4e4e0;
      position: relative;
    }
    .fill > span {
      display: block;
      height: 100%;
      width: 0%;
      background: #ff6b35;
    }
    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
        padding: 24px 0;
      }
      canvas {
        width: min(360px, calc(100vw - 68px));
        height: min(360px, calc(100vw - 68px));
      }
    }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Draw a digit. LeNet-5 classifies it.</h1>
      <p>Use the black canvas like MNIST: white digit on black background. The app crops and rescales your drawing before sending it to the trained checkpoint.</p>
      <div class="canvas-wrap">
        <canvas id="canvas" width="280" height="280" aria-label="Digit drawing canvas"></canvas>
      </div>
      <div class="actions">
        <button id="predict">Classify</button>
        <button id="clear" class="secondary">Clear</button>
      </div>
    </section>
    <section class="result">
      <p>Prediction</p>
      <div id="digit" class="digit">-</div>
      <div id="confidence" class="confidence">Draw a digit first</div>
      <div id="bars" class="bars"></div>
    </section>
  </main>
  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const predictButton = document.getElementById("predict");
    const clearButton = document.getElementById("clear");
    const digit = document.getElementById("digit");
    const confidence = document.getElementById("confidence");
    const bars = document.getElementById("bars");

    function resetCanvas() {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 22;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#fff";
    }

    function point(event) {
      const rect = canvas.getBoundingClientRect();
      const source = event.touches ? event.touches[0] : event;
      return {
        x: (source.clientX - rect.left) * (canvas.width / rect.width),
        y: (source.clientY - rect.top) * (canvas.height / rect.height),
      };
    }

    let drawing = false;
    canvas.addEventListener("pointerdown", (event) => {
      drawing = true;
      const p = point(event);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drawing) return;
      const p = point(event);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    });
    window.addEventListener("pointerup", () => { drawing = false; });

    clearButton.addEventListener("click", () => {
      resetCanvas();
      digit.textContent = "-";
      confidence.textContent = "Draw a digit first";
      bars.innerHTML = "";
    });

    predictButton.addEventListener("click", async () => {
      confidence.textContent = "Classifying...";
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: canvas.toDataURL("image/png") }),
      });
      const result = await response.json();
      digit.textContent = result.prediction;
      confidence.textContent = `${Math.round(result.confidence * 1000) / 10}% confidence`;
      bars.innerHTML = "";
      result.probabilities.forEach((value, index) => {
        const row = document.createElement("div");
        row.className = "bar";
        row.innerHTML = `<strong>${index}</strong><div class="fill"><span style="width:${value * 100}%"></span></div><span>${Math.round(value * 1000) / 10}%</span>`;
        bars.appendChild(row);
      });
    });

    resetCanvas();
  </script>
</body>
</html>
"""


def create_app(checkpoint_path: Path, cpu: bool = False) -> Flask:
    app = Flask(__name__)
    device = get_device(prefer_cuda=not cpu)
    model = LeNet5().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    @app.get("/")
    def index() -> str:
        return HTML

    @app.post("/predict")
    def predict():
        payload = request.get_json(force=True)
        image_data = payload.get("image", "")
        image_tensor = preprocess_canvas_data_url(image_data).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        prediction = int(probabilities.argmax().item())
        confidence = float(probabilities[prediction].item())
        return jsonify(
            {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": [float(value) for value in probabilities.tolist()],
            }
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local browser demo for LeNet-5 MNIST.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lenet5_mnist_best.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.checkpoint, cpu=args.cpu)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
