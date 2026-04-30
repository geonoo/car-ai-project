"""
데이터 학습 전 체크리스트
------------------------
1) 클래스 맵 생성 (필수 권장)
   문자열 라벨을 쓰는 경우 category_name_to_id 없이 학습하면 대부분 0번으로만 들어갈 수 있습니다.

       python tools/extract_classes.py
       # 생성 확인: checkpoints/class_map.json

2) train.py 실행 (환경 변수로 조절)

       BATCH_SIZE=2 NUM_EPOCHS=10 NUM_WORKERS=4 python train.py

   고해상도 메모리 절약: TRAIN_RESIZE=640 (정사각) 또는 TRAIN_RESIZE=800,1333

3) 검증 mAP (선택): VAL_IMAGE_DIR, VAL_LABEL_DIR 를 설정하면 에폭마다 torchmetrics mAP 를 기록합니다.

       VAL_IMAGE_DIR=... VAL_LABEL_DIR=... VAL_MAX_BATCHES=50 python train.py

M1/M2/M3: pick_device() 가 mps 를 우선 사용합니다. 대용량이면 DATA_ROOT·경로를 SSD/CUDA 서버에 맞게 바꾸세요.
Loss NaN·MPS 런타임 오류 시 에러 로그를 알려 주세요.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import (
    IMAGE_DIR,
    LABEL_DIR,
    AutonomousDataset,
    detection_collate_fn,
    visualize_sample,
)
from metrics import evaluate_detection_map
from model import build_fasterrcnn_resnet50_fpn

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DEFAULT_CLASS_MAP_PATH = CHECKPOINT_DIR / "class_map.json"


def parse_resize_hw(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if "," in s:
        a, b = s.split(",", 1)
        return int(a.strip()), int(b.strip())
    v = int(s)
    return v, v


def build_dataloader_kwargs(num_workers: int) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"num_workers": num_workers}
    if num_workers > 0:
        kw["persistent_workers"] = True
    if torch.cuda.is_available():
        kw["pin_memory"] = True
    return kw


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_label_offset(
    targets: List[Dict[str, torch.Tensor]], offset: int
) -> List[Dict[str, torch.Tensor]]:
    if offset == 0:
        return targets
    out: List[Dict[str, torch.Tensor]] = []
    for t in targets:
        t2 = {k: v.clone() if k == "labels" else v for k, v in t.items()}
        if t2["labels"].numel() > 0:
            t2["labels"] = t2["labels"] + offset
        out.append(t2)
    return out


def save_checkpoint(
    path: Path,
    *,
    last_epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    avg_loss: float,
    best_loss: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "last_epoch": last_epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "avg_loss": avg_loss,
        "best_loss": best_loss,
    }
    if extra:
        payload["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


def load_class_map(path: Path) -> Tuple[Dict[str, int], Dict[int, str], int, int]:
    with open(path, "r", encoding="utf-8") as f:
        cm = json.load(f)
    category_name_to_id: Dict[str, int] = {
        str(k).lower(): int(v) for k, v in cm["category_name_to_id"].items()
    }
    id_to_name: Dict[int, str] = {
        int(k): str(v) for k, v in cm.get("id_to_name", {}).items()
    }
    num_classes = int(cm["num_classes"])
    label_offset = int(cm.get("label_offset", 0))
    return category_name_to_id, id_to_name, num_classes, label_offset


def train_one_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    label_offset: int = 0,
    max_batches: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
    epoch: int = 0,
) -> Tuple[float, int]:
    model.train()
    running = 0.0
    n = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = apply_label_offset(targets, label_offset)

        optimizer.zero_grad(set_to_none=True)
        loss_dict: Dict[str, torch.Tensor] = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        running += loss_val
        n += 1

        if writer is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", float(v.detach().cpu()), global_step)
            writer.add_scalar("train/total", loss_val, global_step)
            global_step += 1

        print(
            f"epoch {epoch}  batch {batch_idx}  "
            + "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            + f"  total={loss_val:.4f}"
        )

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    avg = running / max(n, 1)
    if writer is not None:
        writer.add_scalar("epoch/avg_loss", avg, epoch)
        writer.flush()
    return avg, global_step


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    CLASS_MAP_PATH = Path(
        os.environ.get("CLASS_MAP_PATH", str(DEFAULT_CLASS_MAP_PATH))
    )
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
    LR = float(os.environ.get("LR", "0.005"))
    MOMENTUM = float(os.environ.get("MOMENTUM", "0.9"))
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0005"))
    STEP_SIZE = int(os.environ.get("STEP_SIZE", "10"))
    GAMMA = float(os.environ.get("GAMMA", "0.1"))
    SEED = int(os.environ.get("SEED", "42"))
    RUN_VISUALIZE = os.environ.get("RUN_VISUALIZE", "1") == "1"
    RESUME_PATH = os.environ.get("RESUME", "").strip()
    LOG_DIR = os.environ.get("LOG_DIR", str(ROOT / "runs" / "train"))
    MAX_BATCHES = os.environ.get("MAX_BATCHES")
    max_batches = int(MAX_BATCHES) if MAX_BATCHES else None
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
    TRAIN_RESIZE = parse_resize_hw(os.environ.get("TRAIN_RESIZE", "").strip() or None)
    VAL_IMAGE_DIR = os.environ.get("VAL_IMAGE_DIR", "").strip()
    VAL_LABEL_DIR = os.environ.get("VAL_LABEL_DIR", "").strip()
    VAL_MAX_BATCHES = os.environ.get("VAL_MAX_BATCHES")
    val_max_batches = int(VAL_MAX_BATCHES) if VAL_MAX_BATCHES else None

    category_name_to_id: Optional[Dict[str, int]] = None
    id_to_name: Optional[Dict[int, str]] = None
    if CLASS_MAP_PATH.is_file():
        category_name_to_id, id_to_name, NUM_CLASSES, LABEL_OFFSET = load_class_map(
            CLASS_MAP_PATH
        )
        print(f"class_map 로드: {CLASS_MAP_PATH}  NUM_CLASSES={NUM_CLASSES}")
    else:
        NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "2"))
        LABEL_OFFSET = int(os.environ.get("LABEL_OFFSET", "1"))
        print(
            f"[경고] class_map 없음 → NUM_CLASSES={NUM_CLASSES}, LABEL_OFFSET={LABEL_OFFSET}\n"
            "  문자열 라벨을 쓰는 데이터면 먼저 실행: python tools/extract_classes.py\n"
            "  → checkpoints/class_map.json 생성 후 다시 학습하세요.",
            file=sys.stderr,
        )

    set_seed(SEED)
    device = pick_device()
    print("device:", device)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_RESIZE is not None:
        print(f"TRAIN_RESIZE={TRAIN_RESIZE} (박스 좌표 동일 비율 스케일)")

    if RUN_VISUALIZE:
        ds_viz = AutonomousDataset(
            image_dir=IMAGE_DIR,
            label_dir=LABEL_DIR,
            transform=None,
            category_name_to_id=category_name_to_id,
            require_label_file=True,
            resize_hw=TRAIN_RESIZE,
        )
        if len(ds_viz) > 0:
            visualize_sample(ds_viz, index=0, id_to_name=id_to_name)
        else:
            print("시각화 스킵: 데이터셋이 비어 있습니다.")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = AutonomousDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        transform=transform,
        category_name_to_id=category_name_to_id,
        require_label_file=True,
        resize_hw=TRAIN_RESIZE,
    )

    loader_kw = build_dataloader_kwargs(NUM_WORKERS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=detection_collate_fn,
        **loader_kw,
    )

    val_loader: Optional[DataLoader] = None
    if VAL_IMAGE_DIR and VAL_LABEL_DIR:
        val_dataset = AutonomousDataset(
            image_dir=VAL_IMAGE_DIR,
            label_dir=VAL_LABEL_DIR,
            transform=transform,
            category_name_to_id=category_name_to_id,
            require_label_file=True,
            resize_hw=TRAIN_RESIZE,
        )
        if len(val_dataset) == 0:
            print("VAL 경로가 비어 있어 검증을 건너뜁니다.")
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=detection_collate_fn,
                **loader_kw,
            )
            print(f"검증 로더: {len(val_dataset)} 샘플")
    print(f"DataLoader num_workers={NUM_WORKERS}")

    model = build_fasterrcnn_resnet50_fpn(NUM_CLASSES, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

    last_epoch = -1
    global_step = 0
    best_loss = float("inf")

    if RESUME_PATH:
        ckpt = load_checkpoint(
            Path(RESUME_PATH), model, optimizer, scheduler, device
        )
        last_epoch = int(ckpt.get("last_epoch", -1))
        global_step = int(ckpt.get("global_step", 0))
        best_loss = float(ckpt.get("best_loss", float("inf")))
        print(f"체크포인트 재개: {RESUME_PATH}  last_epoch={last_epoch}  best_loss={best_loss:.4f}")

    writer = SummaryWriter(log_dir=LOG_DIR)
    writer.add_text("config/num_classes", str(NUM_CLASSES), 0)
    writer.add_text("config/label_offset", str(LABEL_OFFSET), 0)
    if CLASS_MAP_PATH.is_file():
        writer.add_text("config/class_map", str(CLASS_MAP_PATH.resolve()), 0)

    try:
        for epoch in range(last_epoch + 1, NUM_EPOCHS):
            avg_loss, global_step = train_one_epoch(
                train_loader,
                model,
                optimizer,
                device,
                label_offset=LABEL_OFFSET,
                max_batches=max_batches,
                writer=writer,
                global_step=global_step,
                epoch=epoch,
            )
            scheduler.step()

            epoch_tag = epoch + 1
            save_checkpoint(
                CHECKPOINT_DIR / f"epoch_{epoch_tag:04d}.pth",
                last_epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                avg_loss=avg_loss,
                best_loss=best_loss,
                extra={
                    "num_classes": NUM_CLASSES,
                    "label_offset": LABEL_OFFSET,
                },
            )
            print(f"epoch {epoch_tag} mean total loss: {avg_loss:.4f}  (저장: epoch_{epoch_tag:04d}.pth)")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    CHECKPOINT_DIR / "best_model.pth",
                    last_epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    avg_loss=avg_loss,
                    best_loss=best_loss,
                    extra={
                        "num_classes": NUM_CLASSES,
                        "label_offset": LABEL_OFFSET,
                    },
                )
                print(f"  best 갱신 → best_model.pth (best_loss={best_loss:.4f})")

            if val_loader is not None:
                try:
                    val_metrics = evaluate_detection_map(
                        model,
                        val_loader,
                        device,
                        label_offset=LABEL_OFFSET,
                        max_batches=val_max_batches,
                    )
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)
                    writer.flush()
                    main = val_metrics.get(
                        "map", next(iter(val_metrics.values()), 0.0)
                    )
                    print(f"  val metrics: {val_metrics}  (대표 map={main:.4f})")
                except Exception as exc:
                    print(f"  val/mAP 스킵: {exc}")
    finally:
        writer.close()
