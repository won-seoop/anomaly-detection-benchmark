from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT.parent
DATA_DIR = TASK_ROOT / "data" / "2" / "cut"
OUT_DIR = TASK_ROOT / "result" / "angles" / "2" / "contour" / "sam"
TEMP_DIR = ROOT / "_batch_tmp"


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def imwrite(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))


def fastsam_contour(model, img, stem):
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    temp_path = TEMP_DIR / f"sam_contour_{stem}.png"
    imwrite(temp_path, upscaled)

    result = model(str(temp_path), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)[0]
    if result.masks is None:
        return None

    candidates = []
    for mask in result.masks.data.cpu().numpy():
        m = ((mask > 0.5).astype(np.uint8) * 255)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            touches_any = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
            too_big = (bw > w * 0.80 and bh > h * 0.70) or area > w * h * 0.45
            shape_bad = bh > h * 0.60 and bw > w * 0.45
            too_small = area < 150 or bw < 40 or bh < 6
            if touches_any or too_big or shape_bad or too_small:
                continue
            candidates.append(c)

    if not candidates:
        return None
    return max(candidates, key=cv2.contourArea)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    model = FastSAM(str(ROOT / "FastSAM-s.pt"))

    for path in sorted(DATA_DIR.glob("*.png"), key=lambda p: int(p.stem)):
        img = imread(path)
        contour = fastsam_contour(model, img, path.stem)
        out = img.copy()

        if contour is not None:
            cv2.drawContours(out, [contour], -1, (0, 0, 255), 1)

        imwrite(OUT_DIR / f"{path.stem}_contour.png", out)
        print(f"{path.name} done")


if __name__ == "__main__":
    main()
