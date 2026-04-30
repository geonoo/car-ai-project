"""
LABEL_DIR 아래 모든 JSON을 읽어 문자열 클래스명을 전수 추출하고,
train.py가 바로 사용할 수 있는 class_map.json 포맷으로 저장합니다.

실행:
  python tools/extract_classes.py
  # 생성 확인: checkpoints/class_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import LABEL_DIR, _flatten_object_entries  # noqa: E402


def _names_from_ann(ann: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("label", "category", "class", "name", "subclass"):
        v = ann.get(key)
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
    return out


def _primary_category_name(ann: Dict[str, Any]) -> Optional[str]:
    for key in ("label", "category", "class", "name", "subclass"):
        v = ann.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def scan_labels(label_dir: str) -> Tuple[Set[str], Dict[str, int], int]:
    """
    Returns:
      - all_names: 등장한 모든 문자열 클래스명(원문)
      - primary_counts_lc: 대표 클래스(소문자)별 객체 수
      - num_files: 스캔한 json 파일 수
    """
    all_names: Set[str] = set()
    primary_counts_lc: Dict[str, int] = {}
    num_files = 0
    label_dir = os.path.abspath(label_dir)
    for root, _dirs, files in os.walk(label_dir):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(root, fn)
            num_files += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            for ann in _flatten_object_entries(data):
                for n in _names_from_ann(ann):
                    all_names.add(n)
                prim = _primary_category_name(ann)
                if prim:
                    lc = prim.lower().strip()
                    primary_counts_lc[lc] = primary_counts_lc.get(lc, 0) + 1
    return all_names, primary_counts_lc, num_files


def build_class_map(
    names: Set[str], primary_counts_lc: Dict[str, int]
) -> Dict[str, Any]:
    unique_sorted = sorted(names, key=lambda x: (x.lower(), x))
    category_name_to_id: Dict[str, int] = {
        n.lower().strip(): i + 1 for i, n in enumerate(unique_sorted)
    }
    id_to_name: Dict[str, str] = {
        str(i + 1): unique_sorted[i] for i in range(len(unique_sorted))
    }
    class_counts: Dict[str, int] = {
        n: int(primary_counts_lc.get(n.lower().strip(), 0)) for n in unique_sorted
    }
    num_foreground = len(unique_sorted)
    num_classes = num_foreground + 1  # + background
    return {
        "num_foreground_classes": num_foreground,
        "num_classes": num_classes,
        "label_offset": 0,
        "category_name_to_id": category_name_to_id,
        "id_to_name": id_to_name,
        "class_counts": class_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="라벨 JSON에서 문자열 클래스명을 전수 추출합니다."
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default=LABEL_DIR,
        help="라벨 JSON 루트 경로 (기본: dataset.LABEL_DIR)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "checkpoints" / "class_map.json"),
        help="class_map.json 저장 경로",
    )
    args = parser.parse_args()

    names, primary_lc, num_files = scan_labels(args.label_dir)
    class_map = build_class_map(names, primary_lc)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    print(f"label_dir: {args.label_dir}")
    print(f"스캔한 JSON 파일 수: {num_files}")
    print(f"고유 클래스명 개수(전경): {class_map['num_foreground_classes']}")
    print(f"num_classes (배경 포함): {class_map['num_classes']}")
    print(f"저장: {out_path.resolve()}")

    if class_map["num_foreground_classes"] > 0:
        print("\n클래스별 객체 수 (대표 필드 기준, 내림차순):")
        cc = class_map["class_counts"]
        for n in sorted(cc.keys(), key=lambda k: (-cc[k], k.lower())):
            print(f"  {n}: {cc[n]}")

    if class_map["num_foreground_classes"] == 0:
        print(
            "경고: 문자열 label/category/class/name/subclass 가 한 건도 없습니다. "
            "JSON에 category_id만 있는 경우 수동으로 class_map을 작성하세요."
        )


if __name__ == "__main__":
    main()