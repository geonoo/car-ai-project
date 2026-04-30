"""검증 단계 지표 (torchmetrics 기반 mAP 등)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_detection_map(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    label_offset: int = 0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Faster R-CNN 추론 결과로 COCO 스타일 mAP(박스)를 근사 계산합니다.
    torchmetrics.detection.MeanAveragePrecision 사용 (pycocotools 불필요).
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as e:
        raise ImportError(
            "mAP 검증에 torchmetrics 가 필요합니다. pip install torchmetrics"
        ) from e

    metric = MeanAveragePrecision(box_format="xyxy")
    model.eval()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds: List[Dict[str, torch.Tensor]] = []
        for o in outputs:
            preds.append(
                {
                    "boxes": o["boxes"].detach().cpu(),
                    "scores": o["scores"].detach().cpu(),
                    "labels": o["labels"].detach().cpu(),
                }
            )

        tgts: List[Dict[str, torch.Tensor]] = []
        for t in targets:
            labels = t["labels"].clone()
            if label_offset != 0 and labels.numel() > 0:
                labels = labels + label_offset
            tgts.append(
                {
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": labels.detach().cpu(),
                }
            )

        metric.update(preds, tgts)

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    raw = metric.compute()
    out: Dict[str, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if torch.is_tensor(v):
                out[str(k)] = float(v.detach().cpu().item())
            else:
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
    elif torch.is_tensor(raw):
        out["map"] = float(raw.detach().cpu().item())
    return out
