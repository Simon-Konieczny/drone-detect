"""Run the exported ONNX detector on an image or a folder of images.

Needs only onnxruntime, numpy and Pillow (no PyTorch), which keeps the Docker image small.
Class names and per-class score thresholds are read from the model's own metadata.

  python src/predict.py --model models/model.onnx path/to/img_or_dir --out predictions/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {"drone": (255, 60, 60), "bird": (0, 200, 255), "plane": (60, 255, 60),
          "helicopter": (255, 200, 0), "insect": (200, 100, 255)}


def load_model(path: str):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    classes = json.loads(meta["classes"])
    thresholds = json.loads(meta.get("score_thresholds", "{}"))
    return sess, classes, thresholds


def detect(sess, classes, thresholds, image: Image.Image, override: float | None) -> list[dict]:
    x = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0)
    boxes, labels, scores = sess.run(["boxes", "labels", "scores"], {"image": x})
    out = []
    for box, label, score in zip(boxes, labels, scores):
        name = classes[int(label) - 1]  # label 0 is background
        threshold = override if override is not None else thresholds.get(name, 0.5)
        if score >= threshold:
            out.append({"class": name, "score": round(float(score), 4),
                        "box_xyxy": [round(float(v), 1) for v in box]})
    return out


def draw(image: Image.Image, detections: list[dict]) -> Image.Image:
    image = image.convert("RGB")
    d = ImageDraw.Draw(image)
    for det in detections:
        color = COLORS.get(det["class"], (255, 255, 255))
        d.rectangle(det["box_xyxy"], outline=color, width=2)
        d.text((det["box_xyxy"][0], max(0, det["box_xyxy"][1] - 11)),
               f"{det['class']} {det['score']:.2f}", fill=color)
    return image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or directory of images")
    ap.add_argument("--model", default="models/model.onnx")
    ap.add_argument("--out", help="directory for annotated images + predictions.json")
    ap.add_argument("--score-thresh", type=float, default=None,
                    help="override the per-class thresholds stored in the model")
    args = ap.parse_args()

    src = Path(args.input)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    if not paths:
        sys.exit(f"No images found at {src}")

    sess, classes, thresholds = load_model(args.model)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for p in paths:
        with Image.open(p) as im:
            dets = detect(sess, classes, thresholds, im, args.score_thresh)
            if out_dir:
                draw(im, dets).save(out_dir / f"{p.stem}_pred.jpg")
        results[p.name] = dets
        print(json.dumps({"image": p.name, "detections": dets}))
    if out_dir:
        (out_dir / "predictions.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()