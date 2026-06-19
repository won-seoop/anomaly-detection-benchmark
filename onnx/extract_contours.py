#!/usr/bin/env python3
"""Save raw PiDiNet and FastSAM contours with ONNX models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import FastSAM


ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def pidinet_input(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = rgb.transpose(2, 0, 1)[None]
    return np.ascontiguousarray((tensor - IMAGENET_MEAN) / IMAGENET_STD)


def run_pidinet(session: ort.InferenceSession, image: np.ndarray) -> tuple[np.ndarray, dict]:
    edge = session.run(["edge"], {"images": pidinet_input(image)})[0][0, 0]
    edge = (edge * 255).clip(0, 255).astype(np.uint8)
    _, binary = cv2.threshold(edge, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 20]

    output = image.copy()
    area = 0.0
    box = None
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        box = cv2.boundingRect(contour)
        cv2.drawContours(output, [contour], -1, (0, 0, 255), 1)
    return output, {"contours": len(contours), "largest_area": area, "box": box}


def run_fastsam(model: FastSAM, image: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    upscaled = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    result = model(
        upscaled,
        device="cpu",
        retina_masks=True,
        imgsz=1024,
        conf=0.05,
        iou=0.7,
        verbose=False,
    )[0]

    combined = np.zeros((height, width), np.uint8)
    raw_contours = 0
    if result.masks is not None:
        for mask in result.masks.data.cpu().numpy():
            mask = (mask > 0.5).astype(np.uint8) * 255
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                _, _, box_width, box_height = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                full_frame = box_width >= width - 2 and box_height >= height - 2 and area > 0.65 * width * height
                if area > 150 and box_width > 40 and 6 < box_height < 140 and not full_frame:
                    raw_contours += 1
                    cv2.drawContours(combined, [contour], -1, 255, -1)

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
        iterations=2,
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > 500 and cv2.boundingRect(contour)[2] > 80
    ]

    output = image.copy()
    area = 0.0
    box = None
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        box = cv2.boundingRect(contour)
        cv2.drawContours(output, [contour], -1, (0, 0, 255), 1)
    return output, {
        "raw_contours": raw_contours,
        "contours": len(contours),
        "largest_area": area,
        "box": box,
    }


def image_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "2" / "cut")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result" / "onnx" / "max")
    parser.add_argument("--pidinet-model", type=Path, default=ROOT / "weights" / "onnx" / "PiDiNet.onnx")
    parser.add_argument("--fastsam-model", type=Path, default=ROOT / "weights" / "onnx" / "FastSAM-s.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(args.input_dir.glob("*.png"), key=image_sort_key)
    if not image_paths:
        raise RuntimeError(f"no PNG images found: {args.input_dir}")

    pidinet_dir = args.output_dir / "pidinet"
    sam_dir = args.output_dir / "sam"
    pidinet_dir.mkdir(parents=True, exist_ok=True)
    sam_dir.mkdir(parents=True, exist_ok=True)

    pidinet = ort.InferenceSession(str(args.pidinet_model), providers=["CPUExecutionProvider"])
    fastsam = FastSAM(str(args.fastsam_model))
    rows = []
    for path in image_paths:
        image = imread(path)
        pidinet_image, pidinet_info = run_pidinet(pidinet, image)
        sam_image, sam_info = run_fastsam(fastsam, image)
        imwrite(pidinet_dir / path.name, pidinet_image)
        imwrite(sam_dir / path.name, sam_image)
        rows.append({
            "image": path.name,
            "pidinet_contours": pidinet_info["contours"],
            "pidinet_largest_area": f'{pidinet_info["largest_area"]:.1f}',
            "pidinet_box": pidinet_info["box"],
            "sam_raw_contours": sam_info["raw_contours"],
            "sam_contours": sam_info["contours"],
            "sam_largest_area": f'{sam_info["largest_area"]:.1f}',
            "sam_box": sam_info["box"],
        })
        print(
            f"{path.name}: PiDiNet raw={pidinet_info['contours']} contour(s), "
            f"FastSAM={sam_info['contours']} contour(s)"
        )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
