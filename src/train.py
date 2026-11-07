"""Train stage: fine-tune Faster R-CNN (MobileNetV3-FPN) with MLflow tracking.

Model selection uses the VALID split only, test is only used for evaluate.py.

Outputs:
    models/model.pt             best checkpoint by val mAP@[.5:.95] (DVC out)
    reports/train_metrics.json  best-epoch summary
    MLflow run                  params, per-epoch losses/mAP/lr, lineage tag, model
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import DetectionDataset, collate_fn
from engine import evaluate_map, train_one_epoch
from model import ARCH, build_model, load_checkpoint
from transforms import build_train_transforms

# ------------------------------------------------------------------
CKPT_PATH = Path("models/model.pt")
LOGGED_VAL_KEYS = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def seed_worker(_worker_id: int):
    s = torch.initial_seed() % 2**32 # distinct per worker
    np.random.seed(s)
    random.seed(s)

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ------------------------------------------------------------------
def git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"git_commit": commit, "git_dirty": str(dirty)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": "unknown"}

def data_fingerprint(*paths: Path) -> str:
    """md5 of the manifests: ties each MLflow run to the data version DVC produced."""
    h = hashlib.md5()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------
def warmup_cosine(total_iters: int, warmup_iters: int):
    """Linear warmup protects the freshly initialized head from large early gradients,
    then cosine decay to 0 over the planned run.
    """
    def fn(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return fn

def log_model_to_mlflow(model, registered_name: str | None):
    mlflow.pytorch.log_model( # type: ignore
        model,
        name="model",
        serialization_format="pickle",
        registered_model_name=registered_name
    )


def main():
    params = yaml.safe_load(open("params.yaml"))
    tp, prep = params["train"], params["prepare"]
    preset = params["augment"]["preset"]
    classes = prep["classes"]
    data_dir = Path(prep["out_dir"])

    seed_everything(tp["seed"])
    device = pick_device()
    use_amp = bool(tp["amp"]) and device.type == "cuda"
    print(f"[train] device={device} amp={use_amp} augment={preset}")

    #----- data
    train_ds = DetectionDataset(data_dir / "train.json", build_train_transforms(preset))
    val_ds = DetectionDataset(data_dir / "valid.json", transforms=None)
    g = torch.Generator().manual_seed(tp["seed"])
    loader_kw = dict(batch_size=tp["batch_size"], num_workers=tp["num_workers"],
                     collate_fn=collate_fn, pin_memory=device.type == "cuda",
                     persistent_workers=tp["num_workers"] > 0)
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, worker_init_fn=seed_worker,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    #----- model
    model = build_model(len(classes) + 1, tp["min_size"], tp["max_size"],
                        tp["trainable_backbone_layers"]).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=tp["lr"], momentum=tp["momentum"],
                                weight_decay=tp["weight_decay"])
    total_iters = tp["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_iters, min(tp["warmup_iters"], total_iters))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    #----- tracking
    mlflow.set_tracking_uri(tp.get("mlflow_uri", "sqlite://mlflow.db"))
    mlflow.set_experiment(tp.get("experiment", "drone-detection"))
    run_name = f"{preset}-lr{tp['lr']}-s{tp['min_size']}-tbl{tp['trainable_backbone_layers']}"

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_map, best_epoch, best_val, bad_epochs = -1.0, -1, {}, 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**tp, "augment_preset": preset, "classes": ",".join(classes),
                           "arch": ARCH, "n_train": len(train_ds), "n_val": len(val_ds)})
        mlflow.set_tags({**git_info(), "device": str(device),
                         "data_md5": data_fingerprint(data_dir / "train.json", data_dir / "valid.json")})
        for f in ("params.yaml", "reports/data_stats.json", "reports/aug_preview.jpg"):
            if Path(f).exists():
                mlflow.log_artifact(f)

        for epoch in range(1, tp["epochs"] + 1):
            t0 = time.time()
            losses = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                     device, use_amp, tp["grad_clip"])
            train_time = time.time() - t0
            val = evaluate_map(model, val_loader, device)

            metrics = {f"train/{k}": v for k, v in losses.items()}
            metrics.update({f"val/{k}": val[k] for k in LOGGED_VAL_KEYS if k in val})
            metrics.update({"lr": optimizer.param_groups[0]["lr"],
                            "train/images_per_sec": len(train_loader) * tp["batch_size"] / train_time})
            mlflow.log_metrics(metrics, step=epoch)
            print(f"[epoch {epoch:02d}] | loss={losses['loss']:.4f} "
                  f"val mAP={val['map']:.4f} mAP50={val['map_50']:.4f} "
                  f"mAP_small={val.get('map_small', -1):.4f} ({train_time:.0f}s)")

            if val['map'] > best_map:
                best_map, best_epoch, best_val, bad_epochs = val["map"], epoch, val, 0
                torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": ARCH,
                            "min_size": tp["min_size"], "max_size": tp["max_size"],
                            "epoch": epoch, "val_map": best_map}, CKPT_PATH)
            else:
                bad_epochs += 1
                if bad_epochs >= tp["patience"]:
                    print(f"[train] early stop: no cal mAP gain for {tp['patience']} epochs")
                    break
        
        summary = {"best_epoch": best_epoch, "epochs_run": epoch,
                **{f"best_val_{k}": best_val[k] for k in LOGGED_VAL_KEYS if k in best_val}}
        Path("reports").mkdir(exist_ok=True)
        Path("reports/train_metrics.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_artifact("reports/train_metrics.json")

        best_model, _ = load_checkpoint(str(CKPT_PATH), device="cpu") # CPU copy: portable artifact
        log_model_to_mlflow(best_model, tp.get("registered_model_name") if tp["register_model"] else None)
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()