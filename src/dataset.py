"""Dataset that reads the mainfests written by prepare.py.

Return (image, target) in the format torchvision detection models epxect:
    image: float tensor [3, H, W] in [0, 1]. Not normalized

    target: {"boxes: FloatTensor[N, 4] xyxy, "labels": Int64Tensor[N], "image_id": int}
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    def __init__(self, manifest_path: str, transforms=None):
        with open(manifest_path) as f:
            self.records = json.load(f)["images"]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        image = np.array(Image.open(r["file"]).convert("RGB"))
        boxes = np.asarray(r["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(r["labels"], dtype=np.int64)

        if self.transforms is not None:
            out = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = out["image"]
            boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1,4)
            labels = np.asarray(out["labels"], dtype=np.int64)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255)
        target = {
            "boxes": torch.from_numpy(boxes), # shape: (0, 4) for empty images
            "labels": torch.from_numpy(labels),
            "image_id": idx
        }

        return image_t, target

def collate_fn(batch):
    """Detection batches are lists: images differ in size and box couns differ."""
    return tuple(zip(*batch))