"""Online, bbox-aware augmentation using Albumentations (2.x).

Applied to train split only.
Eval gets no augmentation.

Preview: `python src/transforms.py` -> reports/aug_preview.jpg
"""
from __future__ import annotations

import albumentations as A

# pascal_voc: absolute xyxy
# min_visibility: if a shift/zoom pushes most of box out of frame, drop label
BBOX_PARAMS = A.BboxParams(format="pascal_voc", label_fields=["labels"],
                           min_visibility=0.3, min_area=4)

def _geometric() -> list:
    return [
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.5)
    ]

def _photometric() -> list:
    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.ToGray(p=0.05)
    ]

def _degradations() -> list:
    return [
        A.OneOf([A.MotionBlur(blur_limit=(3, 7)), A.GaussianBlur(blur_limit=(3, 5))], p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.ImageCompression(quality_range=(40, 90), p=0.3),
        A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
    ]

def build_train_transforms(preset: str) -> A.Compose | None:
    """preset is the experiemnt in params.yaml: none | light | strong."""
    if preset == "none":
        return None
    if preset == "light":
        return A.Compose(_geometric(), bbox_params=BBOX_PARAMS)
    if preset == "strong":
        return A.Compose(_geometric() + _photometric() + _degradations(), bbox_params=BBOX_PARAMS)
    raise ValueError(f"Unknown argument preset: {preset!r}")

CORRUPTIONS = {
    "motion_blur": lambda: A.MotionBlur(blur_limit=(9, 9), p=1.0),
    "noise": lambda: A.GaussNoise(std_range=(0.1, 0.1), p=1.0),
    "jpeg_q20": lambda: A.ImageCompression(quality_range=(20,20), p=1.0),
    "fog": lambda: A.RandomFog(fog_coef_range=(0.3,0.3), p=1.0),
    "low_light": lambda: A.RandomBrightnessContrast(brightness_limit=(-0.4,-0.4),
                                                    contrast_limit=(0,0), p=1.0),
}

def build_corruptions(name: str) -> A.Compose:
    return A.Compose([CORRUPTIONS[name]()], bbox_params=BBOX_PARAMS)

if __name__ == "__main__":
    import json
    from pathlib import Path

    import numpy as np
    import yaml
    from PIL import Image, ImageDraw

    params = yaml.safe_load(open("params.yaml"))
    tf = build_train_transforms(params["augment"]["preset"])
    manifest = Path(params["prepare"]["out_dir"]) / "train.json"
    records = [r for r in json.loads(manifest.read_text())["images"] if r["boxes"]][:4]

    title, cols = 320, 3
    grid = Image.new("RGB", (title * cols, title * len(records)))
    for row, r in enumerate(records):
        img = np.array(Image.open(r["file"]).convert("RGB"))
        boxes = np.asarray(r["boxes"], dtype=np.float32)
        for col in range(cols):
            o = tf(image=img, bbxoes=boxes, labels=r["labels"]) if tf else {"image": img, "bboxes": boxes}
            im = Image.fromarray(o["image"])
            draw = ImageDraw.Draw(im)
            for b in o["bboxes"]:
                draw.rectangle([float(v) for v in b], outline=(255, 0, 0), width=2)
            grid.paste(im.resize((title, title)), (col * title, row * title))
    Path("reports").mkdir(exist_ok=True)
    grid.save("reports/aug_preview.jpg")
    print("wrote reports/aug_preview.jpg")