from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(str(path))


root = Path(__file__).resolve().parent
image_path = root / "1.png"
upscaled_path = root / "fastsam_s_upscaled_input.png"
out_path = root / "fastsam_s_result.png"
mask_path = root / "fastsam_s_mask.png"

model = FastSAM("FastSAM-s.pt")

img = imread_unicode(image_path)
h, w = img.shape[:2]
upscaled = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
imwrite_unicode(upscaled_path, upscaled)

results = model(str(upscaled_path), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7)

combined = np.zeros((h, w), np.uint8)
kept = []

masks = results[0].masks
if masks is not None:
    for mask in masks.data.cpu().numpy():
        m = (mask > 0.5).astype(np.uint8) * 255
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
        # Keep crimp/barrel-like chunks, drop tiny border/background noise.
        if area > 150 and bw > 40 and 6 < bh < 95:
                kept.append(c)
                cv2.drawContours(combined, [c], -1, 255, -1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)

final_contours = []
bands = [(15, 80), (80, 125)]
for y1, y2 in bands:
    band = np.zeros_like(combined)
    band[y1:y2, :] = combined[y1:y2, :]
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if cv2.contourArea(c) > 500 and bw > 80 and 8 < bh < 70:
            final_contours.append(c)

out = img.copy()
cv2.drawContours(out, final_contours, -1, (0, 0, 255), 1)

imwrite_unicode(out_path, out)
imwrite_unicode(mask_path, combined)
print(f"raw_kept={len(kept)} final={len(final_contours)}")
print(out_path)
