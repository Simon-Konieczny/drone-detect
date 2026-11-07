"""One training epoch and mAP eval. evaluate.py reuses evalualte_map"""
from __future__ import annotations

import contextlib
from collections import defaultdict

import torch
from torchmetrics.detection import MeanAveragePrecision

def _to_device(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
    return images, targets

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    use_amp: bool, grad_clip: float | None) -> dict[str, float]:
    """Returns the epoch mean of the total loss and each component."""
    model.train()
    sums: dict[str, float] = defaultdict(float)
    n = 0
    for images, targets in loader:
        images, targets = _to_device(images, targets, device)
        amp_ctx = torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()
        with amp_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            parts = {k: round(v.item(), 4) for k, v in loss_dict.items()}
            raise FloatingPointError(f"Non-finite loss {loss.item()}: {parts}. Lower LR or check boxes.")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # per-iteration schedule (warmup + cosine)

        sums["loss"] += loss.item()
        for k, v in loss_dict.items():
            sums[k] += v.item()
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}

@torch.no_grad()
def evaluate_map(model, loader, device, class_metrics: bool = False) -> dict:
    """COCO-style mAP via torchmetrics
    
    Runs at default score threshhold (0.05)
    """
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=class_metrics)
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)
        metric.update(
            [{k: v.detach().cpu() for k, v in p.items()} for p in preds],
            [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets],
        )
        res = metric.compute()
        scalars = {k: float(v) for k, v in res.items() if v.numel() == 1}
        if class_metrics:
            scalars["per_class"] = {
                int(c): {"map": float(m), "mar_100": float(r)}
                for c, m, r in zip(res["classes"], res["map_per_class"], res["mar_100_per_class"])
            }
        return scalars