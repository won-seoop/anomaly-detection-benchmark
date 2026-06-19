import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT.parent
DATA_DIR = TASK_ROOT / "data" / "2" / "cut"
OUT_DIR = TASK_ROOT / "result" / "angles" / "2" / "contour" / "pidinet"
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
    return img[y1:y2, x1:x2], (x1, y1)


def load_pidinet():
    args = Namespace(config="carv4", sa=True, dil=True)
    model = torch.nn.DataParallel(models.pidinet_converted(args)).cuda().eval()
    ckpt = torch.load(REPO / "trained_models" / "table5_pidinet.pth", map_location="cpu")
    model.load_state_dict(convert_pidinet(ckpt["state_dict"], "carv4"))
    return model


def pidinet_contour(model, img):
    crop, offset = tight_crop(img)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = ((x - mean) / std).cuda()

    with torch.no_grad():
        result = model(x)

    edge = (torch.squeeze(result[-1]).detach().cpu().numpy() * 255).astype(np.uint8)
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
    if not contours:
        return None, offset
    return max(contours, key=cv2.contourArea), offset


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_pidinet()

    for path in sorted(DATA_DIR.glob("*.png"), key=lambda p: int(p.stem)):
        img = imread(path)
        contour, (ox, oy) = pidinet_contour(model, img)
        out = img.copy()

        if contour is not None:
            contour_full = contour + np.array([[[ox, oy]]], dtype=contour.dtype)
            cv2.drawContours(out, [contour_full], -1, (0, 0, 255), 1)

        imwrite(OUT_DIR / f"{path.stem}_contour.png", out)
        print(f"{path.name} done")


if __name__ == "__main__":
    main()
