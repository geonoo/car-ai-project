from typing import Optional

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn_resnet50_fpn(
    num_classes: int,
    *,
    pretrained: bool = True,
    weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
) -> torch.nn.Module:
    """
    Faster R-CNN (ResNet50-FPN).

    Args:
        num_classes: 배경을 포함한 전체 클래스 수(예: 전경 클래스 K개면 K+1).
        pretrained: True이면 기본 사전학습 가중치 사용.
        weights: 명시 시 해당 가중치 로드(pretrained보다 우선).
    """
    if weights is not None:
        w = weights
    elif pretrained:
        w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        w = None

    model = fasterrcnn_resnet50_fpn(weights=w)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
