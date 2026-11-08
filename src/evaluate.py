"""Reports
  - COCO metrics plus per-class AP (IoU .50:.95) and AP50 on clean test images
  - bootstrap confidence intervals (resampling images): rare classes have few boxes,
    so single point estimates would overstate how precisely we know them
  - operating points: per class, the score threshold that maximises recall at a target
    precision. Chosen on VALIDATION, then checked on test, so the threshold that ships
    with the model was never tuned on the numbers we report
  - robustness: metrics under fixed-severity corruptions and the drop versus clean

Outputs
  reports/eval_metrics.json   DVC metrics (export.py embeds the thresholds in the ONNX model)
  reports/pr_curves.png       DVC plot
  MLflow                      logged into the training run that produced the checkpoint
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from dataset import DetectionDataset, collate_fn
from model import load_checkpoint
from transforms import build_corruptions

SCALARS = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100")
RECALL_GRID = np.linspace(0.0, 1.0, 101)  # COCO's 101 recall thresholds


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect(model, dataset, device, batch_size: int, num_workers: int):
    """Run inference once and keep predictions on CPU, so metrics and the bootstrap
    can be recomputed without touching the model again."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)
    preds, targets = [], []
    for images, tgts in loader:
        out = model([img.to(device) for img in images])
        preds += [{k: v.detach().cpu() for k, v in o.items()} for o in out]
        targets += [{"boxes": t["boxes"], "labels": t["labels"]} for t in tgts]
    return preds, targets


def summarise(preds, targets, names: dict[int, str]):
    """COCO metrics + per-class AP/AP50, and the IoU-0.5 PR curve for each class.

    torchmetrics' extended summary exposes COCOeval's precision and score arrays,
    shaped [IoU thresholds, recall points, classes, area ranges, max dets]:
    index 0 = IoU 0.5, area 0 = all sizes, max-dets -1 = 100. -1 marks "undefined".
    """
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  class_metrics=True, extended_summary=True)
    metric.update(preds, targets)
    r = metric.compute()

    def nan_if_undefined(x: float) -> float:
        return math.nan if x < 0 else x

    summary = {k: nan_if_undefined(float(r[k])) for k in SCALARS}
    classes = r["classes"].reshape(-1).tolist()
    ap = r["map_per_class"].reshape(-1).tolist()
    per_class, curves = {}, {}
    for k, cid in enumerate(classes):
        prec = r["precision"][0, :, k, 0, -1].numpy()
        scores = r["scores"][0, :, k, 0, -1].numpy()
        name = names.get(int(cid), str(cid))
        has_gt = bool((prec > -1).any())
        per_class[name] = {"ap": nan_if_undefined(ap[k]),
                           "ap50": float(prec.mean()) if has_gt else math.nan}
        if has_gt:
            curves[name] = (prec, scores)
    summary["per_class"] = per_class
    return summary, curves


def operating_point(prec: np.ndarray, scores: np.ndarray, target: float) -> dict | None:
    """Highest recall whose (interpolated, monotone) precision still meets the target."""
    ok = np.nonzero(prec >= target)[0]
    if len(ok) == 0:
        return None
    i = int(ok.max())
    return {"score_threshold": float(scores[i]), "recall": float(RECALL_GRID[i]),
            "precision": float(prec[i])}


def point_at_threshold(prec: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Recall/precision reached on a curve when keeping detections scoring >= threshold."""
    idx = np.nonzero(scores >= threshold)[0]
    if len(idx) == 0:
        return {"recall": 0.0, "precision": math.nan}
    i = int(idx.max())
    return {"recall": float(RECALL_GRID[i]), "precision": float(prec[i])}


def bootstrap(preds, targets, names, iters: int, ci: float, seed: int) -> dict:
    """Percentile CIs from resampling test IMAGES with replacement (objects in one image
    are correlated, so images, not boxes, are the independent unit)."""
    rng = np.random.default_rng(seed)
    n = len(preds)
    samples: dict[str, list] = defaultdict(list)
    for b in range(iters):
        idx = rng.integers(0, n, n)
        s, _ = summarise([preds[i] for i in idx], [targets[i] for i in idx], names)
        samples["map"].append(s["map"])
        samples["map_50"].append(s["map_50"])
        for c, v in s["per_class"].items():
            samples[f"{c}.ap"].append(v["ap"])
            samples[f"{c}.ap50"].append(v["ap50"])
        if (b + 1) % 50 == 0:
            print(f"[eval] bootstrap {b + 1}/{iters}")
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    return {k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi)))
            for k, v in samples.items()}


def plot_pr(curves: dict, per_class: dict, ops: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (prec, _) in curves.items():
        line, = ax.plot(RECALL_GRID, np.clip(prec, 0, None),
                        label=f"{name} (AP50 {per_class[name]['ap50']:.2f})")
        op = ops.get(name)
        if op and op.get("test_recall") is not None:
            ax.scatter(op["test_recall"], op["test_precision"], color=line.get_color(),
                       edgecolor="black", zorder=3)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Test PR curves @ IoU 0.5 (dots: val-chosen operating points)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def flatten(d: dict, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) \
                and not (isinstance(v, float) and math.isnan(v)):
            out[f"{prefix}{k}"] = float(v)
    return out


def main():
    params = yaml.safe_load(open("params.yaml"))
    ep, tp, prep = params["evaluate"], params["train"], params["prepare"]
    data_dir = Path(prep["out_dir"])
    device = pick_device()
    model, ckpt = load_checkpoint("models/model.pt", device)
    names = {i + 1: c for i, c in enumerate(ckpt["classes"])}
    run = dict(device=device, batch_size=ep["batch_size"], num_workers=ep["num_workers"])

    # ---- thresholds chosen on validation
    val_preds, val_targets = collect(model, DetectionDataset(data_dir / "valid.json"), **run)
    _, val_curves = summarise(val_preds, val_targets, names)

    # ---- clean test metrics
    preds, targets = collect(model, DetectionDataset(data_dir / "test.json"), **run)
    clean, curves = summarise(preds, targets, names)

    ops = {}
    for name in names.values():
        op = operating_point(*val_curves[name], ep["target_precision"]) if name in val_curves else None
        if op is None:
            ops[name] = None  # target precision not reachable on validation
            continue
        test_pt = point_at_threshold(*curves[name], op["score_threshold"]) if name in curves \
            else {"recall": math.nan, "precision": math.nan}
        ops[name] = {"score_threshold": op["score_threshold"],
                     "val_recall": op["recall"], "val_precision": op["precision"],
                     "test_recall": test_pt["recall"], "test_precision": test_pt["precision"]}

    # ---- uncertainty
    print(f"[eval] bootstrapping {ep['bootstrap_iters']} resamples of {len(preds)} images")
    cis = bootstrap(preds, targets, names, ep["bootstrap_iters"], ep["ci"], ep["seed"])
    for key in ("map", "map_50"):
        clean[f"{key}_ci_low"], clean[f"{key}_ci_high"] = cis[key]
    for c, v in clean["per_class"].items():
        v["ap_ci_low"], v["ap_ci_high"] = cis.get(f"{c}.ap", (math.nan, math.nan))
        v["ap50_ci_low"], v["ap50_ci_high"] = cis.get(f"{c}.ap50", (math.nan, math.nan))

    # ---- robustness (photometric corruptions: boxes unchanged, so clean targets apply)
    robustness = {}
    for name in ep["corruptions"]:
        c_preds, _ = collect(model, DetectionDataset(data_dir / "test.json",
                                                     transforms=build_corruptions(name)), **run)
        s, _ = summarise(c_preds, targets, names)
        robustness[name] = {
            "map": s["map"], "map_50": s["map_50"],
            "map_50_drop": clean["map_50"] - s["map_50"],
            "per_class_ap50": {c: v["ap50"] for c, v in s["per_class"].items()},
        }
        print(f"[eval] {name:>12}: mAP50={s['map_50']:.3f} (drop {clean['map_50'] - s['map_50']:+.3f})")

    results = {"n_test_images": len(preds), "checkpoint_epoch": ckpt["epoch"],
               "target_precision": ep["target_precision"], "clean": clean,
               "operating_points": ops, "robustness": robustness}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/eval_metrics.json").write_text(json.dumps(jsonable(results), indent=2))
    plot_pr(curves, clean["per_class"], ops, Path("reports/pr_curves.png"))

    print(f"\n[eval] test mAP={clean['map']:.3f} [{clean['map_ci_low']:.3f}, {clean['map_ci_high']:.3f}]"
          f"  mAP50={clean['map_50']:.3f}")
    for c, v in clean["per_class"].items():
        print(f"  {c:>6}: AP50={v['ap50']:.3f} [{v['ap50_ci_low']:.3f}, {v['ap50_ci_high']:.3f}]"
              f"  AP={v['ap']:.3f}  operating point: {ops.get(c)}")

    # ---- MLflow: attach to the run that trained this checkpoint
    mlflow.set_tracking_uri(tp["mlflow_uri"])
    mlflow.set_experiment(tp["experiment"])
    run_id = ckpt.get("mlflow_run_id")
    with (mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name="evaluate")):
        mlflow.log_metrics(flatten({"clean": clean, "operating_points": {k: v for k, v in ops.items() if v},
                                    "robustness": robustness}, prefix="test/"))
        mlflow.log_artifact("reports/eval_metrics.json")
        mlflow.log_artifact("reports/pr_curves.png")


if __name__ == "__main__":
    main()