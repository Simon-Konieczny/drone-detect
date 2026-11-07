"""Prepare stage: validate, clean, and index the raw COCO export.

Outputs
    data/processed/{train,valid,test}.json
    data/processed/label_map.json
    reports/data_stats.json
    reports/samples/*.jpg                   (visual sanity check)
"""
from __future__ import annotations

import yaml
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import imagehash
from PIL import Image, ImageDraw

SPLITS = ("train", "valid", "test")
ANN_FILE = "_annotations.coco.json"
SMALL, MEDIUM = 32**2, 96**2 # COCO size buckets
COLORS = [(255,255,255), (0,200,255), (255,60,60), (60,255,60)]

def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["prepare"]

def build_category_map(categories: list[dict], classes: list[str]) -> dict[int, int]:
    """Map COCO category ids -> contiguous label ids 1...K, in the order given in params.

    Label 0 is background.
    Anything not in classes is ignored.
    """
    wanted = {name.lower(): i + 1 for i, name in enumerate(classes)}
    found = {c["name"].lower() for c in categories}
    missing = set(wanted) - found

    if missing:
        raise ValueError(f"Classes {sorted(missing)} not in export. Found: {sorted(found)}")

    return {c["id"]: wanted[c["name"].lower()] for c in categories if c["name"].lower() in wanted}

def load_split(split_dir: Path, classes: list[str], min_size: float) -> tuple[list[dict], Counter]:
    """Read one split, convert boxes COCO xywh -> xyxy, clip to image, drop degenerate boxes."""
    coco = json.loads((split_dir / ANN_FILE).read_text())
    cat_map = build_category_map(coco["categories"], classes)

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    drops: Counter = Counter()
    records = []
    for img in coco["images"]:
        path = split_dir / img["file_name"]
        if not path.exists():
            drops["missing_image"] += 1
            continue
        W, H = img["width"], img["height"]
        boxes, labels = [], []
        for a in anns_by_img[img["id"]]:
            if a["category_id"] not in cat_map:
                drops["unknown_category"] += 1
                continue
            x, y, w, h = a["bbox"] # COCO format: top-left x, y, width, height
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(W), x + w), min(float(H), y + h)
            if (x2 - x1 < min_size or (y2 - y1) < min_size):
                drops["degenerate_box"] += 1
                continue
            boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
            labels.append(cat_map[a["category_id"]])
        records.append({"file":path.as_posix(), "width": W, "height": H,
                        "boxes": boxes, "labels": labels})

    return records, drops

def phash_all(records: list[dict], hash_size: int) -> np.ndarray:
    """PHash per image, packed into an (N, hash_size**2 / 8) uint8 array."""
    hashes = []
    for r in records:
        with Image.open(r["file"]) as im:
            h = imagehash.phash(im, hash_size=hash_size)
        hashes.append(np.packbits(h.hash.flatten()))
    return np.stack(hashes) if hashes else np.zeros((0, hash_size**2 // 8), np.uint8)

def leakage_mask(reference: np.ndarray, hashes: np.ndarray, max_hamming: int) -> np.ndarray:
    """True for images that are not near-duplicates of anything in reference.
    
    Prevent inflated metrics from frames cut from same video in different splits.
    """
    keep = np.ones(len(hashes), dtype=bool)
    if len(reference) == 0:
        return keep

    for i, h in enumerate(hashes):
        hamming = np.unpackbits(np.bitwise_xor(reference, h), axis=1).sum(axis=1)
        keep[i] = hamming.min() > max_hamming
    return keep

def split_stats(records: list[dict], names: dict[int, str]) -> dict:
    per_class: Counter = Counter()
    sizes: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        for (x1, y1, x2, y2), label in zip(r["boxes"], r["labels"]):
            name = names[label]
            per_class[name] += 1
            area = (x2 - x1) * (y2 - y1)
            sizes[name]["small" if area < SMALL else "medium" if area < MEDIUM else "large"] += 1
    
    return {
        "images": len(records),
        "empty_images": sum(not r["boxes"] for r in records),
        "boxes_per_class": dict(per_class),
        "size_buckets": {k: dict(v) for k, v in sizes.items()},
    }

def draw_samples(records: list[dict], names: dict[int, str], out_dir: Path, n: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = [r for r in records if r["boxes"]] or records
    for i, r in enumerate(random.Random(seed).sample(pool, min(n, len(pool)))):
        with Image.open(r["file"]) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            for box, label in zip(r["boxes"], r["labels"]):
                color = COLORS[label % len(COLORS)]
                draw.rectangle(box, outline=color, width=2)
                draw.text((box[0], max(0, box[1] - 11)), names[label], fill=color)
            im.save(out_dir / f"sample_{i:02d}.jpg")

def main():
    p = load_params()
    raw, out, reports = Path(p["raw_dir"]), Path(p["out_dir"]), Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    names = {0: "__background__", **{i + 1: c for i, c in enumerate(p["classes"])}}

    splits, drops = {}, {}
    for s in SPLITS:
        if not (raw / s / ANN_FILE).exists():
            if s in ("train", "valid"):
                raise FileNotFoundError(f"Missing required split: {raw / s / ANN_FILE}")
            print(f"[prepare] no '{s}' split found, skipping")
            continue
        splits[s], drops[s] = load_split(raw / s, p["classes"], p["min_box_size"])

    leakage = {}
    if p["dedupe"]["enabled"]:
        hs, mh = p["dedupe"]["hash_size"], p["dedupe"]["max_hamming"]
        reference = phash_all(splits["train"], hs)
        leakage["train_internal_duplicates"] = int(len(reference) - len(np.unique(reference, axis=0)))
        for s in ("valid", "test"): # test is checked against train AND the kept valid images
            if s not in splits:
                continue
            hashes = phash_all(splits[s], hs)
            keep = leakage_mask(reference, hashes, mh)
            leakage[f"{s}_removed_near_duplicates"] = int((~keep).sum())
            splits[s] = [r for r, k in zip(splits[s], keep) if k]
            reference = np.concatenate([reference, hashes[keep]])

    for s, records in splits.items():
        (out / f"{s}.json").write_text(json.dumps({"images": records}))
    (out / "label_map.json").write_text(json.dumps(names, indent=2))

    stats = {
        "splits": {s: split_stats(r, names) for s, r in splits.items()},
        "dropped": {s: dict(d) for s, d in drops.items()},
        "leakage": leakage
    }
    (reports / "data_stats.json").write_text(json.dumps(stats, indent=2))
    draw_samples(splits["train"], names, reports / "samples", p["n_samples"], p["seed"])
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()