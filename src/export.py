"""Export stage: ONNX export, accuracy parity check against PyTorch, CPU latency benchmark.

The exported model is self-describing: class names and the per-class score thresholds
chosen on validation by evaluate.py are stored in the ONNX metadata, so the inference
container needs nothing but the .onnx file.

Outputs
  models/model.onnx              DVC out, copied into the Docker image
  reports/export_metrics.json    DVC metrics: size, parity, latency
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from torchmetrics.detection import MeanAveragePrecision

from dataset import DetectionDataset
from model import ARCH, load_checkpoint

ONNX_PATH = Path("models/model.onnx")
INPUT = "image"
OUTPUTS = ["boxes", "labels", "scores"]


def export_onnx(model, sample: torch.Tensor, opset: int, metadata: dict[str, str]):
    """Input: one float32 image [3, H, W] in [0, 1], any H/W (the graph contains the
    model's own resize + normalisation). Outputs: boxes [N, 4] xyxy in input pixels,
    labels [N], scores [N]."""
    kwargs = dict(
        input_names=[INPUT], output_names=OUTPUTS, opset_version=opset,
        dynamic_axes={INPUT: {1: "height", 2: "width"},
                      **{o: {0: "num_detections"} for o in OUTPUTS}},
    )
    # The TorchScript-based exporter handles torchvision detection models (NMS, RoIAlign,
    # data-dependent output sizes); newer torch defaults to the dynamo exporter.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(model, ([sample],), str(ONNX_PATH), **kwargs)

    m = onnx.load(str(ONNX_PATH))
    for key, value in metadata.items():
        prop = m.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.checker.check_model(m)
    onnx.save(m, str(ONNX_PATH))


def make_session(threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])


def onnx_predict(sess: ort.InferenceSession, image: torch.Tensor) -> dict:
    boxes, labels, scores = sess.run(OUTPUTS, {INPUT: image.numpy()})
    return {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels).long(),
            "scores": torch.from_numpy(scores)}


@torch.no_grad()
def parity(model, sess, dataset, n: int) -> dict:
    """Compare mAP of PyTorch vs ONNX Runtime on the same images. Comparing raw boxes is
    brittle (tiny numeric differences can flip NMS decisions); comparing the metric is not."""
    m_torch = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    m_onnx = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    for i in range(min(n, len(dataset))):
        image, t = dataset[i]
        target = [{"boxes": t["boxes"], "labels": t["labels"]}]
        m_torch.update([model([image])[0]], target)
        m_onnx.update([onnx_predict(sess, image)], target)
    a, b = m_torch.compute(), m_onnx.compute()
    return {"parity_images": min(n, len(dataset)),
            "map_torch": float(a["map"]), "map_onnx": float(b["map"]),
            "map_50_torch": float(a["map_50"]), "map_50_onnx": float(b["map_50"])}


def bench(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def main():
    params = yaml.safe_load(open("params.yaml"))
    xp, prep, evp = params["export"], params["prepare"], params["evaluate"]
    torch.set_num_threads(xp["num_threads"])

    model, ckpt = load_checkpoint("models/model.pt", "cpu")
    dataset = DetectionDataset(Path(prep["out_dir"]) / "valid.json")
    sample = dataset[0][0]

    eval_path = Path("reports/eval_metrics.json")
    ops = json.loads(eval_path.read_text()).get("operating_points", {}) if eval_path.exists() else {}
    thresholds = {c: (ops.get(c) or {}).get("score_threshold", xp["fallback_score_threshold"])
                  for c in ckpt["classes"]}
    metadata = {
        "classes": json.dumps(ckpt["classes"]),             # label id i+1 -> classes[i]
        "score_thresholds": json.dumps(thresholds),          # chosen on validation
        "target_precision": str(evp["target_precision"]),
        "arch": ARCH,
        "checkpoint_epoch": str(ckpt["epoch"]),
        "input": "float32 RGB [3,H,W] in [0,1], any size",
    }

    print(f"[export] exporting {ARCH} to ONNX (opset {xp['opset']})")
    export_onnx(model, sample, xp["opset"], metadata)
    sess = make_session(xp["num_threads"])

    par = parity(model, sess, dataset, xp["parity_images"])
    par["map_abs_diff"] = abs(par["map_torch"] - par["map_onnx"])

    with torch.inference_mode():
        latency = {
            "torch_cpu": bench(lambda: model([sample]), xp["warmup_runs"], xp["latency_runs"]),
            "onnx_cpu": bench(lambda: onnx_predict(sess, sample), xp["warmup_runs"], xp["latency_runs"]),
        }

    results = {"onnx_size_mb": ONNX_PATH.stat().st_size / 1e6, "opset": xp["opset"],
               "num_threads": xp["num_threads"], "input_hw": list(sample.shape[1:]),
               **par, "latency": latency, "score_thresholds": thresholds}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/export_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    if par["map_abs_diff"] > xp["parity_tol"]:
        raise SystemExit(f"[export] parity check FAILED: |mAP diff| {par['map_abs_diff']:.4f} "
                         f"> tolerance {xp['parity_tol']}")
    print("[export] parity check passed")


if __name__ == "__main__":
    main()