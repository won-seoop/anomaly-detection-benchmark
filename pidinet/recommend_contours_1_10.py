import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import FastSAM


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT.parent
DATA_DIR = TASK_ROOT / "data"
OUT_DIR = TASK_ROOT / "result" / "recommend"
REPO = ROOT / "pidinet_repo"

sys.path.insert(0, str(REPO))
import models  # noqa: E402
from models.convert_pidinet import convert_pidinet  # noqa: E402


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def imwrite(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))


def tight_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row = gray.mean(axis=1)
    rows = row > max(80, float(np.percentile(row, 45)))
    rows[:5] = False
    rows[-5:] = False
    ys = np.where(rows)[0]
    y1 = max(0, int(ys.min()) - 3) if len(ys) else 0
    y2 = min(h, int(ys.max()) + 4) if len(ys) else h

    col = gray[y1:y2].mean(axis=0)
    cols = col > max(50, float(np.percentile(col, 15)))
    xs = np.where(cols)[0]
    x1 = max(0, int(xs.min()) - 2) if len(xs) else 0
    x2 = min(w, int(xs.max()) + 3) if len(xs) else w
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def pidinet_input(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).cuda()


def load_pidinet():
    args = Namespace(config="carv4", sa=True, dil=True)
    model = torch.nn.DataParallel(models.pidinet_converted(args)).cuda().eval()
    ckpt = torch.load(REPO / "trained_models" / "table5_pidinet.pth", map_location="cpu")
    model.load_state_dict(convert_pidinet(ckpt["state_dict"], "carv4"))
    return model


def run_pidinet(model, img):
    crop, (x1, y1, _, _) = tight_crop(img)
    with torch.no_grad():
        res = model(pidinet_input(crop))
    edge = (torch.squeeze(res[-1]).detach().cpu().numpy() * 255).astype(np.uint8)
    _, binary = cv2.threshold(edge, 130, 255, cv2.THRESH_BINARY)
    margin = 8
    if binary.shape[0] > margin * 2 and binary.shape[1] > margin * 2:
        binary[:margin, :] = 0
        binary[-margin:, :] = 0
        binary[:, :margin] = 0
        binary[:, -margin:] = 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    out = img.copy()
    box = None
    area = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        c = c + np.array([[[x1, y1]]], dtype=c.dtype)
        box = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, (0, 0, 255), 1)
    return out, box, area, len(contours)


def run_fastsam(model, img, stem):
    h, w = img.shape[:2]
    up_path = ROOT / "_batch_tmp" / f"recommend_{stem}.png"
    up = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    imwrite(up_path, up)
    r = model(str(up_path), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)[0]
    candidates = []
    if r.masks is not None:
        for mask in r.masks.data.cpu().numpy():
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
    out = img.copy()
    box = None
    area = 0.0
    if candidates:
        c = max(candidates, key=cv2.contourArea)
        area = cv2.contourArea(c)
        box = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, (0, 0, 255), 1)
    return out, box, area, len(candidates)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (80, 18), (255, 255, 255), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_sheet(paths, out_path):
    cells = []
    for path in paths:
        img = imread(path)
        img = cv2.resize(img, (256, 160), interpolation=cv2.INTER_AREA)
        cells.append(img)
    rows = []
    for i in range(0, len(cells), 5):
        rows.append(np.hstack(cells[i:i + 5]))
    imwrite(out_path, np.vstack(rows))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_p = load_pidinet()
    model_f = FastSAM(str(ROOT / "FastSAM-s.pt"))
    summary = []
    p_paths = []
    f_paths = []

    for path in sorted(DATA_DIR.glob("*.png"), key=lambda p: int(p.stem)):
        img = imread(path)
        p_img, p_box, p_area, p_n = run_pidinet(model_p, img)
        f_img, f_box, f_area, f_n = run_fastsam(model_f, img, path.stem)
        p_img = label(p_img, f"{path.stem} PiDi")
        f_img = label(f_img, f"{path.stem} FSAM")
        p_out = OUT_DIR / f"{path.stem}_pidinet.png"
        f_out = OUT_DIR / f"{path.stem}_fastsam_s.png"
        imwrite(p_out, p_img)
        imwrite(f_out, f_img)
        p_paths.append(p_out)
        f_paths.append(f_out)
        summary.append(f"{path.name}: pidinet n={p_n} box={p_box} area={p_area:.1f} | fastsam n={f_n} box={f_box} area={f_area:.1f}")
        print(summary[-1])

    make_sheet(p_paths, OUT_DIR / "pidinet_sheet.png")
    make_sheet(f_paths, OUT_DIR / "fastsam_s_sheet.png")
    (OUT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
