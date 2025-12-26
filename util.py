"""
Utility helpers to convert COCO-style datasets into Pascal VOC flat files.

The main entry point, ``convert_split``, reads ``annotations.json`` from a
dataset split (``hw3_dataset/<split>``) and mirrors every annotation into a text
file that ``extern/Object-Detection-Metrics`` can read.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Number = float | int


def _format_number(value: Number) -> str:
    """Format numbers with up to 6 decimals, trimming trailing zeros."""
    as_float = float(value)
    if abs(as_float - round(as_float)) < 1e-6:
        return str(int(round(as_float)))
    text = f"{as_float:.6f}"
    return text.rstrip("0").rstrip(".")


def _bbox_xyxy(bbox: Sequence[Number]) -> Tuple[float, float, float, float]:
    """Convert [x, y, w, h] to [left, top, right, bottom]."""
    x, y, w, h = map(float, bbox)
    return x, y, x + w, y + h


def _bbox_xywh(bbox: Sequence[Number]) -> Tuple[float, float, float, float]:
    """Keep COCO bbox as [left, top, width, height]."""
    return tuple(map(float, bbox))  # type: ignore[return-value]


def _convert_bbox(bbox: Sequence[Number],
                  box_format: str) -> Tuple[float, ...]:
    if box_format == "xyxy":
        return _bbox_xyxy(bbox)
    if box_format == "xywh":
        return _bbox_xywh(bbox)
    raise ValueError(f"Unsupported box_format '{box_format}'")


def _write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(rows)
    path.write_text(text)


def _synthesize_coco_from_detections(
        split_dir: Path, detections: Sequence[dict]) -> dict:
    """
    Build a minimal COCO-style structure using detections only.

    The synthesized dataset includes:
        * ``images``: derived from ``split_dir / "images"`` matching stems
          against detection ``image_id`` values (e.g. ``101 -> 101.jpg``).
        * ``categories``: numeric IDs mapped to their own string name.
        * ``annotations``: empty list.
    """
    images_dir = split_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    stem_to_name = {
        path.stem: path.name
        for path in images_dir.iterdir() if path.is_file()
    }
    image_entries = []
    for image_id in sorted({det["image_id"] for det in detections}):
        stem = str(image_id)
        file_name = stem_to_name.get(stem)
        if not file_name:
            raise FileNotFoundError(
                f"No image file matches id '{image_id}' in {images_dir}")
        image_entries.append({"id": image_id, "file_name": file_name})

    category_entries = [{
        "id": cat_id,
        "name": str(cat_id),
    } for cat_id in sorted({det["category_id"] for det in detections})]

    return {
        "images": image_entries,
        "annotations": [],
        "categories": category_entries,
    }


def convert_split(split_name: str,
                  dataset_root: str,
                  output_root: str | None = None,
                  detections_json: str | None = None,
                  box_format: str = "xyxy") -> None:
    """
    Convert a dataset split into Pascal VOC style TXT files.

    Args:
        split_name: One of ``train``, ``val`` or ``test``.
        dataset_root: Path to ``hw3_dataset``.
        output_root: Optional root for converted files. Defaults to
            ``<dataset_root>/<split>/pascalvoc``.
        detections_json: Optional path to COCO-style detection results JSON.
            When ``annotations.json`` is missing (e.g. ``test`` split), the
            detections are used to synthesize the required image metadata using
            ``split/images/<image_id>.jpg``.
        box_format: Either ``xyxy`` (default) or ``xywh``.
    """
    split_dir = Path(dataset_root) / split_name
    ann_path = split_dir / "annotations.json"

    output_base = (Path(output_root) if output_root else split_dir /
                   "pascalvoc")
    gt_dir = output_base / "groundtruths"
    det_dir = output_base / "detections"

    detections_data: List[dict] = []
    detections_by_image: Dict[int, List[dict]] = defaultdict(list)
    if detections_json:
        det_path = Path(detections_json)
        if not det_path.exists():
            raise FileNotFoundError(f"Missing detections JSON: {det_path}")
        detections_data = json.loads(det_path.read_text())
        for det in detections_data:
            image_id = det["image_id"]
            score = det.get("score", det.get("confidence"))
            if score is None:
                raise ValueError(
                    "Detection entry missing 'score'/'confidence'")
            detections_by_image[image_id].append({
                "category_id":
                det["category_id"],
                "score":
                float(score),
                "bbox":
                det["bbox"],
            })

    if ann_path.exists():
        coco = json.loads(ann_path.read_text())
    else:
        if not detections_data:
            raise FileNotFoundError(
                f"Missing annotations JSON: {ann_path}. Provide detections to "
                "synthesize image metadata.")
        coco = _synthesize_coco_from_detections(split_dir, detections_data)

    categories: Dict[int, str] = {
        item["id"]: item["name"]
        for item in coco.get("categories", [])
    }

    annotations = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        annotations[ann["image_id"]].append(ann)

    for image in coco.get("images", []):
        image_id = image["id"]
        file_stem = Path(image["file_name"]).stem

        gt_lines = []
        for ann in annotations.get(image_id, []):
            cat_id = ann["category_id"]
            label = categories.get(cat_id, str(cat_id))
            bbox = _convert_bbox(ann["bbox"], box_format)
            coords = " ".join(_format_number(val) for val in bbox)
            gt_lines.append(f"{label} {coords}")
        _write_lines(gt_dir / f"{file_stem}.txt", gt_lines)

        if detections_json:
            det_lines = []
            for det in detections_by_image.get(image_id, []):
                label = categories.get(det["category_id"],
                                       str(det["category_id"]))
                bbox = _convert_bbox(det["bbox"], box_format)
                coords = " ".join(_format_number(val) for val in bbox)
                det_lines.append(
                    f"{label} {_format_number(det['score'])} {coords}")
            _write_lines(det_dir / f"{file_stem}.txt", det_lines)


def main():
    """CLI entry point to run ``convert_split`` directly."""
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to Pascal VOC text files.")
    parser.add_argument("--split_name",
                        help="Dataset split to convert (train/val/test).")
    parser.add_argument("--dataset_root",
                        help="Path to the hw3_dataset directory.")
    parser.add_argument("--output_root",
                        help="Optional destination for VOC files.")
    parser.add_argument("--detections_json",
                        help="Optional COCO-style detections JSON to export.")
    parser.add_argument("--box_format",
                        choices=("xyxy", "xywh"),
                        default="xyxy",
                        help="Bounding box representation for the output.")
    args = parser.parse_args()
    convert_split(split_name=args.split_name,
                  dataset_root=args.dataset_root,
                  output_root=args.output_root,
                  detections_json=args.detections_json,
                  box_format=args.box_format)


if __name__ == "__main__":
    main()
