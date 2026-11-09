# Drone Detection Pipeline

A computer vision pipeline that detects drones in ground-to-air images and separates them from birds, planes, helicopters and insects. Built to work through a full ML workflow on a counter-UAS problem: versioned data, a reproducible pipeline, tracked experiments, leak-free evaluation and a model that can run on edge hardware.

The model itself is kept simple (a pretrained Faster R-CNN with a MobileNetV3 backbone). Most of the work is in the data preparation, evaluation and deployment around it.

**Stack:** PyTorch, torchvision, Albumentations, DVC, MLflow, torchmetrics, ONNX Runtime, Docker

## Pipeline

```
data/raw ──► prepare ──► train ──► evaluate ──► export ──► Docker image
 (DVC)                  (MLflow)                (ONNX)
```

Each stage is defined in `dvc.yaml` with its code, data and parameters as dependencies. `dvc repro` only reruns the stages affected by a change, and `dvc.lock` records the exact versions behind every result. All parameters live in `params.yaml`.

## Quickstart

```bash
pip install -r requirements-lock.txt
dvc pull                                   # or download the dataset from Roboflow using `fetch_data.py`
dvc repro                                  # prepare -> train -> evaluate -> export

dvc exp run -S augment.preset=none         # experiments without editing files
dvc exp show
mlflow ui --backend-store-uri sqlite:///mlflow.db

docker build -t drone-detector .
docker run --rm -v "$PWD/images:/data" drone-detector /data --out /data/preds
```

## Repository structure

```
src/
  prepare.py     cleaning, de-duplication and re-splitting of the raw COCO export
  transforms.py  training augmentations and fixed corruptions for robustness tests
  dataset.py     PyTorch Dataset over the prepared manifests
  model.py       model construction and checkpoint loading
  engine.py      training epoch and mAP evaluation
  train.py       training loop, MLflow tracking, checkpointing
  evaluate.py    test metrics, confidence intervals, operating points, robustness
  export.py      ONNX export, parity check, latency benchmark
  predict.py     standalone ONNX inference (used in the Docker image)
params.yaml      all parameters, tracked by DVC
dvc.yaml         pipeline definition
reports/         data statistics, plots and metrics
```

## Data

The data is the *flying-objects* dataset from Roboflow Universe (version 7, COCO format). I use five classes: bird, drone, plane, helicopter and insect. Helicopters and insects are kept as their own classes because they are common sources of false alarms when looking for drones, so the model should learn to tell them apart.

### Problems with the original splits

The table below shows the original Roboflow splits after cleaning and after removing near-duplicates of training images from validation and test.

| Split | Images | Empty | Bird | Drone | Plane | Helicopter | Insect | Boxes per image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7,101 | 129 | 3,571 | 2,676 | 2,180 | 1,520 | 743 | 1.5 |
| valid | 424 | 14 | 4,673 | 93 | 92 | 0 | 0 | 11.5 |
| test | 181 | 5 | 1,660 | 48 | 25 | 0 | 0 | 9.6 |

Four problems came out of this:

1. **Leakage between splits.** 148 of 572 validation images (25.9%) and 109 of 290 test images (37.6%) were near-duplicates of training images, most likely adjacent frames from the same videos. Keeping them would have inflated every test metric.
2. **Missing classes.** Helicopters and insects only appear in the training split, so they could not be evaluated at all.
3. **Different class mix.** Validation and test are about 96% birds with around ten objects per image (flock scenes), while training averages 1.5 objects per image. Only 93 drone boxes remained in validation and 48 in test.
4. **Unlabelled objects.** A first version of the pipeline used only bird, drone and plane. This left 1,405 training images that looked empty but contained helicopters or insects, which the model would have learned as background. With all five classes, 129 truly empty images remain.

### Preparation steps

`prepare.py` is deterministic and does no augmentation. It:

1. Fixes the label ids in `params.yaml` (bird=1, drone=2, plane=3, helicopter=4, insect=5, with 0 as background), so they do not depend on the order of categories in the export.
2. Converts boxes from COCO `[x, y, w, h]` to `[x1, y1, x2, y2]`, clips them to the image and drops boxes under 2 pixels wide or high (4 in the whole dataset).
3. Removes any image that contains an annotation outside the five classes (126 annotations in train, 7 in validation, 5 in test), so no unlabelled object is used as background.
4. Computes a 64-bit perceptual hash for every image and groups images that are within 4 bits of each other. Groups are transitive, so a sequence of video frames ends up in one group.
5. Pools all images and re-splits them 80/10/10 with scikit-learn's `StratifiedGroupKFold`. Each group stays in a single split, and the splits are stratified by image type: the first class present in the order drone, insect, helicopter, plane, bird, with images of five or more birds as a separate "bird flock" type and empty images as their own type.
6. Checks that no near-duplicate pair crosses splits after re-splitting (the count should be 0), then writes the statistics to `reports/data_stats.json` and a few images with their boxes drawn to `reports/samples/`.

### Remaining limitations of the data

Drones in this dataset are mostly close to the camera: 53.6% of training drone boxes are large and only 13.7% are small (under 32x32 pixels). Detecting drones at long range is the harder real-world case, and it is under-represented here. There are also few empty sky images (129 in the original training split), so false alarms on empty frames are only partly tested.

## Augmentation

Augmentation is applied on the fly to the training split only, so every epoch sees new variants and validation and test stay untouched. Each transform corresponds to a real condition for a camera looking at the sky:

| Transform | Condition |
|---|---|
| Horizontal flip | Left and right have no meaning in the sky. Vertical flips are not used because the sky above and ground below is a useful cue. |
| Scale 0.7 to 1.3, shift up to 10% | Target size changes with distance. |
| Rotation up to 10 degrees | Platform roll. Kept small because rotating a box makes it larger than the object. |
| Brightness, contrast, hue and saturation | Exposure, time of day, glare. |
| Greyscale (5%) | Monochrome cameras. |
| Motion or Gaussian blur | Fast targets, vibration, focus at range. |
| Noise | Low light. |
| JPEG compression | Compressed video links. |
| Fog | Weather and haze. |

Boxes that are less than 30% visible after a transform are dropped. The `augment.preset` parameter (`none`, `light` or `strong`) is the main experiment setting.

## Training

- Faster R-CNN with a MobileNetV3-Large FPN backbone, pretrained on COCO, with a new prediction head for the five classes. COCO already contains birds and planes, so the pretrained features transfer well.
- SGD with a linear warmup and cosine decay, gradient clipping and early stopping on validation mAP.
- Mixed precision in bfloat16 on CUDA. The first runs in float16 produced NaN losses, because float16 overflows above about 65,000. bfloat16 has the same range as float32, so it avoids this without loss scaling.
- Every run is tracked in MLflow: all parameters, the four Faster R-CNN losses per epoch, validation mAP by object size, the git commit and an md5 hash of the data manifests. The best model is registered in the MLflow Model Registry.
- The test split is never used during training or model selection.

## Evaluation

`evaluate.py` is the only stage that reads the test split. It reports:

- **Per-class AP and AP50,** since overall mAP averages the five classes equally and hides how the drone class itself performs.
- **95% confidence intervals** from 200 bootstrap resamples. Images are resampled rather than boxes, because objects in the same image are not independent.
- **Operating points.** For each class, the score threshold that gives the highest recall at 90% precision is chosen on validation and then measured on test. These thresholds are stored in the exported model.
- **Robustness.** The test set is run again under motion blur, noise, heavy JPEG compression, fog and low light, and the drop in mAP is reported for each.
- **Precision-recall curves** at IoU 0.5 in `reports/pr_curves.png`.

Results are written to `reports/eval_metrics.json` and logged to the MLflow run that trained the checkpoint.

## Export and deployment

`export.py` converts the model to ONNX and stores the class names and score thresholds in the model's metadata, so the ONNX file is all that is needed for inference. It then:

- runs 100 validation images through both PyTorch and ONNX Runtime and fails the stage if their mAP differs by more than 0.01;
- measures single-image CPU latency (median and 95th percentile) for both, with the same number of threads.

The Docker image contains only ONNX Runtime, NumPy, Pillow, `predict.py` and the ONNX model, with no PyTorch, training code or data. The `.dockerignore` file is an allow-list, and the container runs as a non-root user.

```bash
docker run --rm -v "$PWD/images:/data" drone-detector /data --out /data/preds
```

This writes the images with predicted boxes and a `predictions.json` file to `/data/preds`.

## Results

Training is in progress. This section will include the comparison between training with and without augmentation, per-class AP with confidence intervals on the test set, the operating points, the robustness results and the ONNX latency figures.