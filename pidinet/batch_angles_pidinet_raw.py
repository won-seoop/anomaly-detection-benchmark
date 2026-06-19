import csv
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import largestinteriorrectangle as lir
import numpy as np
import torch

ROOT = Path(r'C:\Users\lims\Desktop\dev\업무\비전\inspecter\task\성능비교\pidinet')
TASK_ROOT = ROOT.parent
DATA_DIR = TASK_ROOT / "data" / "2" / "cut"
OUT_DIR = TASK_ROOT / "result" / "angles" / "2" / "pidinet_raw"
REPO = ROOT / "pidinet_repo"
sys.path.insert(0, str(REPO))

import models
from models.convert_pidinet import convert_pidinet


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def imwrite(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))


def normalize_angle(angle):
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle


def load_pidinet():
    args = Namespace(config="carv4", sa=True, dil=True)
    model = torch.nn.DataParallel(models.pidinet_converted(args)).cuda().eval()
    ckpt = torch.load(REPO / "trained_models" / "table5_pidinet.pth", map_location="cpu")
    model.load_state_dict(convert_pidinet(ckpt["state_dict"], "carv4"))
    return model


def run_pidinet(model, img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = ((x - mean) / std).cuda()
    with torch.no_grad():
        result = model(x)
    edge = (torch.squeeze(result[-1]).detach().cpu().numpy() * 255).astype(np.uint8)
    return edge


def min_area_angle(contour):
    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), raw_angle = rect
    angle = raw_angle + 90 if w < h else raw_angle
    return normalize_angle(angle), rect


def rotated_inscribed_angle(mask, step=0.5, angle_min=-15.0, angle_max=15.0):
    h, w = mask.shape
    center = (w / 2.0, h / 2.0)
    best = None
    for angle in np.arange(angle_min, angle_max + step * 0.5, step):
        matrix = cv2.getRotationMatrix2D(center, -float(angle), 1.0)
        rotated = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        rx, ry, rw, rh = [int(v) for v in lir.lir(rotated.astype(bool))]
        area = rw * rh
        if area <= 0:
            continue
        if best is None or area > best["area"]:
            best = {"angle": float(angle), "rect": (rx, ry, rw, rh), "area": area, "matrix": matrix}
    return best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_pidinet()
    rows = []

    for path in sorted(DATA_DIR.glob("*.png"), key=lambda p: int(p.stem)):
        img = imread(path)
        edge = run_pidinet(model, img)

        _, binary = cv2.threshold(edge, 130, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 20]

        out = img.copy()
        row = {"image": path.name}

        if contours:
            contour = max(contours, key=cv2.contourArea)
            min_angle, min_rect = min_area_angle(contour)
            min_box = cv2.boxPoints(min_rect).astype(np.int32)

            mask = np.zeros(binary.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            best = rotated_inscribed_angle(mask)

            cv2.drawContours(out, [contour], -1, (0, 0, 255), 1)
            cv2.drawContours(out, [min_box], 0, (0, 255, 255), 2)

            inscribed_angle = None
            inscribed_area = None
            if best is not None:
                rx, ry, rw, rh = best["rect"]
                pts = np.array([[rx, ry], [rx+rw, ry], [rx+rw, ry+rh], [rx, ry+rh]], dtype=np.float32).reshape(-1, 1, 2)
                inv = cv2.invertAffineTransform(best["matrix"])
                pts = cv2.transform(pts, inv).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                cv2.drawContours(out, [pts], 0, (0, 255, 0), 2)
                inscribed_angle = best["angle"]
                inscribed_area = best["area"]

            cv2.putText(out, f"max={min_angle:.2f}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            if inscribed_angle is not None:
                cv2.putText(out, f"min={inscribed_angle:.2f}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            row.update({
                "min_outer_angle_deg": f"{min_angle:.2f}",
                "min_outer_area": f"{cv2.contourArea(min_box):.1f}",
                "max_inner_angle_deg": "" if inscribed_angle is None else f"{inscribed_angle:.2f}",
                "max_inner_area": "" if inscribed_area is None else str(inscribed_area),
                "contour_area": f"{cv2.contourArea(contour):.1f}",
            })
        else:
            row.update({"min_outer_angle_deg": "", "min_outer_area": "", "max_inner_angle_deg": "", "max_inner_area": "", "contour_area": ""})

        imwrite(OUT_DIR / f"{path.stem}_angles.png", out)
        rows.append(row)
        print(row)

    with (OUT_DIR / "angles_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "min_outer_angle_deg", "min_outer_area", "max_inner_angle_deg", "max_inner_area", "contour_area"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
