"""Model contruction and checkpoint loading."""
from __future__ import annotations

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ARCH = "fasterrcnn_mobilenet_v3_large_fpn"

def build_model(num_classes: int, min_size: int, max_size: int,
                trainable_backbone_layers: int | None = 3, pretrained: bool = True):
    """COCO-pretrained Faster R-CNN with a fresh box head for classes.
    
    num_classes includes background. COCO already contains 'bird' and 'airplane'
    min/max_size set the model's internal resize, upscaling helps with small object at cost of compute
    """
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
        min_size=min_size,
        max_size=max_size
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_checkpoint(path: str, device: str | torch.device = "cpu"):
    """Rebuild the model from a checkpoint written by train.py"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ckpt["classes"]) + 1, ckpt["min_size"], ckpt["max_size"],
                        pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt