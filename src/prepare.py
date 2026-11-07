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
from sklearn.model_selection import StratifiedGroupKFold

SPLITS = ("train", "valid", "test")
ANN_FILE = "_annotations.coco.json"
SMALL, MEDIUM = 32**2, 96**2 # COCO size buckets
COLORS = [(255,255,255), (0,200,255), (255,60,60), (60,255,60)]
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

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

def load_split(split_dir: Path, classes: list[str], min_size: float,
               unknown_policy: str) -> tuple[list[dict], Counter]:
    """Read one split, convert boxes COCO xywh -> xyxy, clip to image, drop degenerate boxes."""
    coco = json.loads((split_dir / ANN_FILE).read_text())
    cat_map = build_category_map(coco["categories"], classes)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
 
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
        boxes, labels, has_unknown = [], [], False
        for a in anns_by_img[img["id"]]:
            if a["category_id"] not in cat_map:
                drops[f"unknown_category_box:{cat_names.get(a['category_id'], a['category_id'])}"] += 1
                has_unknown = True
                continue
            x, y, w, h = a["bbox"]  # COCO format: top-left x, y, width, height
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(W), x + w), min(float(H), y + h)
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                drops["degenerate_box"] += 1
                continue
            boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
            labels.append(cat_map[a["category_id"]])
        if has_unknown and unknown_policy == "drop_image":
            drops["image_with_unknown_category"] += 1
            continue
        # Images with no boxes are kept on purpose: empty sky is a useful hard negative.
        records.append({"file": path.as_posix(), "width": W, "height": H,
                        "boxes": boxes, "labels": labels, "source_split": split_dir.name})
    return records, drops

def phash_all(records: list[dict], hash_size: int) -> np.ndarray:
    """PHash per image, packed into an (N, hash_size**2 / 8) uint8 array."""
    hashes = []
    for r in records:
        with Image.open(r["file"]) as im:
            h = imagehash.phash(im, hash_size=hash_size)
        hashes.append(np.packbits(h.hash.flatten()))
    return np.stack(hashes) if hashes else np.zeros((0, hash_size**2 // 8), np.uint8)

def hamming(h: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Hamming distance from one packed hash to many, via a byte popcount table."""
    return POPCOUNT[np.bitwise_xor(others, h)].sum(axis=1)

def leakage_mask(reference: np.ndarray, hashes: np.ndarray, max_hamming: int) -> np.ndarray:
    """True for images that are NOT near-duplicates of anything in `reference`."""
    keep = np.ones(len(hashes), dtype=bool)
    if len(reference):
        for i, h in enumerate(hashes):
            keep[i] = hamming(h, reference).min() > max_hamming
    return keep

def near_duplicate_groups(hashes: np.ndarray, max_hamming: int) -> np.ndarray:
    """Union-find over all pairs within max_hamming: returns a group id per image.
 
    Transitive by design: if frame A~B and B~C, all three share a group even if A and C
    differ more.
    """
    n = len(hashes)
    parent = np.arange(n)
 
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
 
    for i in range(n - 1):
        close = np.nonzero(hamming(hashes[i], hashes[i + 1:]) <= max_hamming)[0] + i + 1
        for j in close:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])

def stratum(record: dict, names: dict[int, str], priority: list[str],
            flock_class: str, flock_min: int) -> str:
    """Image type used for stratification: the rarest class present wins, dense flocks
    get their own stratum (they dominate box counts), and empty images are their own type."""
    present = Counter(names[label] for label in record["labels"])
    for cls in priority:
        if cls == flock_class and present[cls] >= flock_min:
            return f"{cls}_flock"
        if present[cls]:
            return cls
    return "empty"
 
 
def grouped_stratified_split(strata: list[str], groups: np.ndarray, n_folds: int,
                             seed: int) -> np.ndarray:
    """One fold -> test, one fold -> valid, the rest -> train (80/10/10 with 10 folds)."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in sgkf.split(np.zeros(len(strata)), strata, groups)]
    assign = np.array(["train"] * len(strata), dtype=object)
    assign[folds[0]] = "test"
    assign[folds[1]] = "valid"
    return assign

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
    hs, mh = p["dedupe"]["hash_size"], p["dedupe"]["max_hamming"]
 
    splits, drops = {}, {}
    for s in SPLITS:
        if not (raw / s / ANN_FILE).exists():
            if s == "train":
                raise FileNotFoundError(f"Missing required split: {raw / s / ANN_FILE}")
            print(f"[prepare] no '{s}' split found, skipping")
            continue
        splits[s], drops[s] = load_split(raw / s, p["classes"], p["min_box_size"],
                                         p["unknown_category_policy"])
 
    leakage: dict = {}
    rs = p["resplit"]
    if rs["enabled"]:
        uncovered = set(p["classes"]) - set(rs["priority"])
        if uncovered:  # otherwise images with only those classes would be stratified as "empty"
            raise ValueError(f"resplit.priority must list every class; missing {sorted(uncovered)}")
        pool = [r for s in SPLITS for r in splits.get(s, [])]
        hashes = phash_all(pool, hs)
        groups = near_duplicate_groups(hashes, mh)
        sizes = Counter(groups.tolist())
        strata = [stratum(r, names, rs["priority"], rs["flock_class"], rs["flock_min_boxes"])
                  for r in pool]
        assign = grouped_stratified_split(strata, groups, rs["n_folds"], p["seed"])
        splits = {s: [r for r, a in zip(pool, assign) if a == s] for s in SPLITS}
 
        # Verification: by construction no near-duplicate pair should cross splits.
        hash_by_split = {s: hashes[assign == s] for s in SPLITS}
        cross = int((~leakage_mask(hash_by_split["train"], hash_by_split["valid"], mh)).sum()
                    + (~leakage_mask(np.concatenate([hash_by_split["train"], hash_by_split["valid"]]),
                                     hash_by_split["test"], mh)).sum())
        leakage = {
            "strategy": "grouped_stratified_resplit",
            "pooled_images": len(pool),
            "duplicate_groups": sum(1 for c in sizes.values() if c > 1),
            "images_in_duplicate_groups": sum(c for c in sizes.values() if c > 1),
            "largest_group": max(sizes.values()) if sizes else 0,
            "cross_split_near_duplicates": cross,  # should be 0
        }
        strata_counts = {s: dict(Counter(st for st, a in zip(strata, assign) if a == s))
                         for s in SPLITS}
    else:
        strata_counts = {}
        if p["dedupe"]["enabled"]:
            reference = phash_all(splits["train"], hs)
            leakage = {"strategy": "remove_eval_near_duplicates",
                       "train_internal_duplicates": int(len(reference)
                                                        - len(np.unique(reference, axis=0)))}
            for s in ("valid", "test"):  # test is checked against train AND the kept valid images
                if s not in splits:
                    continue
                hashes = phash_all(splits[s], hs)
                keep = leakage_mask(reference, hashes, mh)
                leakage[f"{s}_removed_near_duplicates"] = int((~keep).sum())
                splits[s] = [r for r, k in zip(splits[s], keep) if k]
                reference = np.concatenate([reference, hashes[keep]])
 
    for s in SPLITS:
        if not splits.get(s):
            raise ValueError(f"Split '{s}' is empty after preparation.")
        (out / f"{s}.json").write_text(json.dumps({"images": splits[s]}))
    (out / "label_map.json").write_text(json.dumps(names, indent=2))
 
    stats = {
        "splits": {s: split_stats(r, names) for s, r in splits.items()},
        "strata": strata_counts,
        "dropped": {s: dict(d) for s, d in drops.items()},
        "leakage": leakage,
    }
    (reports / "data_stats.json").write_text(json.dumps(stats, indent=2))
    draw_samples(splits["train"], names, reports / "samples", p["n_samples"], p["seed"])
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()