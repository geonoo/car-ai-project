# car-ai-project (Faster R-CNN 객체 탐지)

이 프로젝트는 **`torchvision.models.detection.fasterrcnn_resnet50_fpn`** 기반으로 자율주행 이미지 데이터의 객체를 탐지하는 학습 파이프라인입니다.  
코드는 기능별로 `dataset.py`, `model.py`, `train.py`로 분리되어 있습니다.

## 데이터 경로 (기본값)

- **`DATA_ROOT` 기본값**: `/Users/geonoo/13.상용 자율주행차 주간 도심도로 데이터`
- **이미지 폴더**: `3.개방데이터/1.데이터/Training/01.원천데이터/TS/1.맑음/images`
- **라벨 폴더**: `3.개방데이터/1.데이터/Training/02.라벨링데이터/TL/1.맑음/01.관계데이터`

모든 실행은 **환경 변수 `DATA_ROOT`로 덮어쓰기** 가능합니다.

## 실행 전 필수 체크리스트

### 1) 클래스 맵 생성 (중요)

라벨 JSON이 문자열 클래스명(`label`, `category`, `class`, `name` 등)을 사용하면, **문자열→정수 매핑**을 먼저 만들어야 합니다.

```bash
python3 tools/extract_classes.py
ls -la checkpoints/class_map.json
```

`train.py`는 기본으로 `checkpoints/class_map.json`을 읽어 **`NUM_CLASSES`를 자동 설정**합니다.

### 2) 바운딩박스 파싱 확인(권장)

이 데이터는 `polygon`/`segmentation` 형태가 있을 수 있어, `dataset.py`에서 **polygon→bbox(xyxy)** 변환을 지원합니다.  
학습 전에 시각화로 박스가 제대로 올라오는지 확인하세요.

## 환경 준비 (macOS / MPS)

### 가상환경 생성

```bash
python3 -m venv venv
./venv/bin/python -m pip install -U pip
./venv/bin/pip install -r requirements.txt
```

### MPS 사용 확인

```bash
./venv/bin/python -c "import torch; print('mps', torch.backends.mps.is_available())"
```

## 학습 실행

### 기본 학습 (MPS 자동 선택)

```bash
PYTHONUNBUFFERED=1 \
DATA_ROOT="/Users/geonoo/13.상용 자율주행차 주간 도심도로 데이터" \
NUM_EPOCHS=10 \
TRAIN_RESIZE=640 \
BATCH_SIZE=2 \
RUN_VISUALIZE=0 \
./venv/bin/python train.py
```

- **`TRAIN_RESIZE=640`**: 이미지와 박스를 `(640, 640)`으로 리사이즈(박스 좌표도 동일 비율 스케일)
- **`RUN_VISUALIZE=0`**: `matplotlib plt.show()`로 멈추는 것을 방지(학습 로그만 보고 싶을 때)
- **Out of Memory / Killed**: `BATCH_SIZE=1`로 낮춰 재시도

### DataLoader 병렬 로딩 (속도 개선)

```bash
NUM_WORKERS=4 ./venv/bin/python train.py
```

## 체크포인트 저장 / 재개

학습 중 매 에폭마다 `checkpoints/epoch_XXXX.pth` 저장, best 갱신 시 `checkpoints/best_model.pth` 저장합니다.  
재개:

```bash
RESUME="checkpoints/epoch_0003.pth" ./venv/bin/python train.py
```

## TensorBoard 로그

학습 로스(`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`, `total`)가 기록됩니다.

```bash
tensorboard --logdir runs/
```

## 참고: 이미지-라벨 매칭 규칙

라벨 파일명이 이미지와 1:1로 동일하지 않고 `*_u1.json` 같은 접미사가 붙는 경우가 있어, `dataset.py`는 아래 우선순위로 라벨을 찾습니다.

1) `<image_rel>.json`  
2) `<image_rel>_u1.json`  
3) `<image_rel>_u*.json` 중 첫 번째

---

문제(예: Loss NaN, MPS 런타임 에러, 경로 매칭 실패)가 나면 `train.py` 실행 로그를 첨부해 주세요.

