import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = os.environ.get(
    "DATA_ROOT", "/Users/geonoo/13.상용 자율주행차 주간 도심도로 데이터"
)
IMAGE_DIR = os.path.join(
    DATA_ROOT,
    "3.개방데이터",
    "1.데이터",
    "Training",
    "01.원천데이터",
    "TS",
    "1.맑음",
    "images",
)
LABEL_DIR = os.path.join(
    DATA_ROOT,
    "3.개방데이터",
    "1.데이터",
    "Training",
    "02.라벨링데이터",
    "TL",
    "1.맑음",
    "01.관계데이터",
)


def default_label_path(image_path: str, image_dir: str, label_dir: str) -> str:
    rel = os.path.relpath(image_path, image_dir)
    base, _ = os.path.splitext(rel)
    return os.path.join(label_dir, base + ".json")


def resolve_label_path(image_path: str, image_dir: str, label_dir: str) -> Optional[str]:
    """
    이미지 경로에 대응하는 라벨 JSON 경로를 찾습니다.
    기본: <same_rel_path>.json
    예외: 데이터셋에 따라 <same_rel_path>_u1.json 같은 접미사가 붙는 경우가 있어 후보를 확장합니다.
    """
    rel = os.path.relpath(image_path, image_dir)
    base, _ = os.path.splitext(rel)

    candidates = [
        os.path.join(label_dir, base + ".json"),
        os.path.join(label_dir, base + "_u1.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    # 마지막 fallback: base + "_u*.json" 중 첫 번째를 사용
    parent = os.path.join(label_dir, os.path.dirname(base))
    stem = os.path.basename(base)
    if os.path.isdir(parent):
        try:
            for fn in os.listdir(parent):
                if fn.startswith(stem + "_u") and fn.endswith(".json"):
                    p = os.path.join(parent, fn)
                    if os.path.isfile(p):
                        return p
        except OSError:
            pass
    return None


def polygon_to_bbox(polygon: List[float]) -> List[float]:
    """
    [x1, y1, x2, y2, ... , xn, yn] 형태의 다각형 좌표를
    [xmin, ymin, xmax, ymax] 형태의 Bounding Box로 변환합니다.
    """
    x_coords = polygon[0::2]
    y_coords = polygon[1::2]

    xmin = float(min(x_coords))
    ymin = float(min(y_coords))
    xmax = float(max(x_coords))
    ymax = float(max(y_coords))

    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    return [xmin, ymin, xmax, ymax]


def _normalize_windows_path(p: str) -> str:
    # 라벨 JSON에 "\\02.원천데이터\\..." 같은 형태가 들어오는 경우가 있어 정규화
    return p.replace("\\", "/")


def _coerce_polygon(value: Any) -> Optional[List[float]]:
    # 지원 케이스:
    # - [x1,y1,x2,y2,...] (flat)
    # - [[x1,y1,x2,y2,...]] (nested 1-level)
    # - [[x1,y1],[x2,y2],...] (pairs)
    if isinstance(value, (list, tuple)) and len(value) >= 6:
        if all(isinstance(v, (int, float)) for v in value):
            poly = [float(v) for v in value]
            if len(poly) % 2 == 0:
                return poly
            return None

        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return _coerce_polygon(value[0])

        if all(
            isinstance(v, (list, tuple))
            and len(v) == 2
            and all(isinstance(x, (int, float)) for x in v)
            for v in value
        ):
            flat: List[float] = []
            for x, y in value:
                flat.extend([float(x), float(y)])
            return flat if len(flat) >= 6 else None

    return None


def _bbox_xyxy_from_ann(ann: Dict[str, Any]) -> Optional[List[float]]:
    if "bbox" in ann and isinstance(ann["bbox"], (list, tuple)) and len(ann["bbox"]) == 4:
        x, y, w, h = (float(ann["bbox"][i]) for i in range(4))
        if w <= 0 or h <= 0:
            return None
        return [x, y, x + w, y + h]

    polygon = ann.get("polygon") or ann.get("segmentation") or ann.get("polyline")
    poly = _coerce_polygon(polygon)
    if poly is not None:
        return polygon_to_bbox(poly)

    box = ann.get("box") or ann.get("bndbox")
    if isinstance(box, dict):
        keys = (
            ("xmin", "ymin", "xmax", "ymax"),
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
        )
        for a, b, c, d in keys:
            if all(k in box for k in (a, b, c, d)):
                return [float(box[a]), float(box[b]), float(box[c]), float(box[d])]
    return None


def _category_id_from_ann(
    ann: Dict[str, Any],
    category_name_to_id: Optional[Dict[str, int]],
) -> int:
    # 1) 숫자 ID가 있으면 우선 사용
    for key in ("category_id", "categoryId", "class_id", "label_id"):
        if key in ann:
            try:
                return int(ann[key])
            except (TypeError, ValueError):
                pass

    # 2) 텍스트 클래스명 (건우님 데이터: label 키 사용)
    name = ann.get("label") or ann.get("category") or ann.get("class") or ann.get("name")
    if isinstance(name, str) and category_name_to_id is not None:
        key = name.lower().strip()
        return int(category_name_to_id.get(key, 0))
    return 0


def _flatten_object_entries(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    anns: List[Dict[str, Any]] = []

    if isinstance(raw.get("annotations"), list):
        for item in raw["annotations"]:
            if isinstance(item, dict):
                if "object" in item and isinstance(item["object"], dict):
                    anns.append(item["object"])
                else:
                    anns.append(item)

    ann_block = raw.get("annotation")
    if isinstance(ann_block, dict):
        objs = ann_block.get("objects")
        if isinstance(objs, list):
            for item in objs:
                if not isinstance(item, dict):
                    continue
                if "object" in item and isinstance(item["object"], dict):
                    anns.append(item["object"])
                else:
                    anns.append(item)

    if isinstance(raw.get("objects"), list):
        for item in raw["objects"]:
            if not isinstance(item, dict):
                continue
            if "object" in item and isinstance(item["object"], dict):
                anns.append(item["object"])
            else:
                anns.append(item)

    return anns


def scale_target_boxes(
    target: Dict[str, torch.Tensor], sx: float, sy: float
) -> Dict[str, torch.Tensor]:
    """원본 픽셀 좌표 박스를 (sx, sy)만큼 스케일합니다."""
    if target["boxes"].numel() == 0:
        return target
    b = target["boxes"].clone()
    b[:, 0] *= sx
    b[:, 2] *= sx
    b[:, 1] *= sy
    b[:, 3] *= sy
    return {"boxes": b, "labels": target["labels"]}


class AutonomousDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transform=None,
        category_name_to_id: Optional[Dict[str, int]] = None,
        require_label_file: bool = True,
        resize_hw: Optional[Tuple[int, int]] = None,
    ):
        self.image_dir = os.path.abspath(image_dir)
        self.label_dir = os.path.abspath(label_dir)
        self.transform = transform
        self.category_name_to_id = category_name_to_id
        self.require_label_file = require_label_file
        self.resize_hw = resize_hw

        self.image_paths: List[str] = []
        for root, _dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(".jpg"):
                    self.image_paths.append(os.path.join(root, file))
        self.image_paths.sort()

        if self.require_label_file:
            filtered: List[str] = []
            for p in self.image_paths:
                lp = resolve_label_path(p, self.image_dir, self.label_dir)
                if lp is not None:
                    filtered.append(p)
            self.image_paths = filtered

    def __len__(self) -> int:
        return len(self.image_paths)

    def _process_label(
        self, label_data: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        entries = _flatten_object_entries(label_data)
        boxes: List[List[float]] = []
        labels: List[int] = []

        for ann in entries:
            xyxy = _bbox_xyxy_from_ann(ann)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(_category_id_from_ann(ann, self.category_name_to_id))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        return {"boxes": boxes_t, "labels": labels_t}

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        label_path = resolve_label_path(img_path, self.image_dir, self.label_dir)
        label_data: Dict[str, Any] = {}
        if label_path is not None and os.path.isfile(label_path):
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    label_data = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                label_data = {}
        elif not self.require_label_file:
            label_data = {}

        # 라벨 JSON에 윈도우 경로가 섞인 경우 정규화(매칭/디버깅 편의)
        if isinstance(label_data, dict):
            for k in ("image_path", "imgname", "filename"):
                v = label_data.get(k)
                if isinstance(v, str) and "\\" in v:
                    label_data[k] = _normalize_windows_path(v)

        target = self._process_label(label_data)

        if self.resize_hw is not None:
            w0, h0 = image.size
            new_w, new_h = self.resize_hw
            if w0 > 0 and h0 > 0:
                sx = new_w / float(w0)
                sy = new_h / float(h0)
                target = scale_target_boxes(target, sx, sy)
            image = image.resize((new_w, new_h), Image.BILINEAR)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def detection_collate_fn(
    batch: List[Tuple[Any, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Faster R-CNN은 배치 이미지를 List[Tensor[C,H,W]] 형태로 받습니다."""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def visualize_sample(
    dataset: AutonomousDataset,
    index: int = 0,
    id_to_name: Optional[Dict[int, str]] = None,
    figsize: Tuple[float, float] = (12.0, 8.0),
) -> None:
    if dataset.transform is not None:
        raise ValueError(
            "시각화는 PIL 이미지가 필요합니다. "
            "AutonomousDataset(..., transform=None)으로 만든 뒤 호출하세요."
        )

    image, target = dataset[index]
    rgb = np.array(image, dtype=np.uint8)
    vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    boxes = target["boxes"].detach().cpu().numpy()
    labels = target["labels"].detach().cpu().numpy()

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cid = int(labels[i]) if labels.size else 0
        name = (
            id_to_name.get(cid, str(cid))
            if id_to_name is not None
            else str(cid)
        )
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_bgr,
            name,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(vis_rgb)
    plt.axis("off")
    plt.title(f"sample index={index}, num_boxes={boxes.shape[0]}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from torchvision import transforms

    ds_viz = AutonomousDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        transform=None,
        require_label_file=True,
    )
    if len(ds_viz) > 0:
        visualize_sample(ds_viz, index=0, id_to_name=None)

    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = AutonomousDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        transform=transform,
        require_label_file=True,
    )
    loader = DataLoader(
        ds_train,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )
    for batch_images, batch_targets in loader:
        print(len(batch_images), batch_images[0].shape, len(batch_targets))
        break
