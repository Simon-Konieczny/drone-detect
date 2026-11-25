"""Latency depending on the scene: empty sky vs densest flock.

to run: `python scene_latency.py`

Reports, for each image: GT boxes, how many proposals reached the RoI head,
 raw detections from ONNX model, and Runtime p50 / p95 latency with 4 threads. (same as export.py)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

sys.path.insert(0, "src")
from model import load_checkpoint

THREADS, WARMUP, RUNS = 4, 10, 200

records = []
for split in ("valid", "test"):
    records += json.loads(Path(f"data/processed/{split}.json").read_text())["images"]
empty = next(r for r in records if not r["boxes"])
flock = max(records, key=lambda r: len(r["boxes"]))

def load(record):
    x = np.asarray(Image.open(record["file"]).convert("RGB"), dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1) / 255.0)

so = ort.SessionOptions()
so.intra_op_num_threads = THREADS
sess = ort.InferenceSession("models/model.onnx", so, providers=["CPUExecutionProvider"])

torch.set_num_threads(THREADS)
model, _ = load_checkpoint("models/model.pt", "cpu")
seen = {}

def count_proposals(module, inputs, output):
    seen["n"] = inputs[0].shape[0]

model.roi_heads.box_head.register_forward_hook(count_proposals)

def bench(x):
    for _ in range(WARMUP):
        sess.run(None, {"image": x})
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = sess.run(None, {"image": x})
        times.append((time.perf_counter() - t0) * 1000)
    return np.percentile(times, 50), np.percentile(times, 95), len(out[0])

print(f"{'image':14s} {'size':>9s} {'GT':>4s} {'proposals':>10s} {'dets':>5s} {'p50 ms':>8s} {'p95 ms':>8s}")
for name, record in (("empty sky", empty), ("densest flock", flock)):
    x = load(record)
    with torch.no_grad():
        model([torch.from_numpy(x)])
    p50, p95, n_det = bench(x)
    size = f"{x.shape[2]}x{x.shape[1]}"
    print(f"{name:14s} {size:>9s} {len(record['boxes']):4d} {seen['n']:10d} {n_det:5d} {p50:8.1f} {p95:8.1f}")
