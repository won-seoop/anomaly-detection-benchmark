import sys
from argparse import Namespace
from pathlib import Path

import cv2
import largestinteriorrectangle as lir
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT.parent
REPO = ROOT / "pidinet_repo"
sys.path.insert(0, str(REPO))

import models  # noqa: E402
from models.convert_pidinet import convert_pidinet  # noqa: E402


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def imwrite(path, img):
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))


def make_pidinet_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    row = gray.mean(axis=1)
    rows = row > max(80, float(np.percentile(row, 45)))
    rows[:5] = False
    rows[-5:] = False
    ys = np.where(rows)[0]
    y1 = max(0, int(ys.min()) - 3)
    y2 = min(H, int(ys.max()) + 4)

    col = gray[y1:y2].mean(axis=0)
    cols = col > max(50, float(np.percentile(col, 15)))
    xs = np.where(cols)[0]
    x1 = max(0, int(xs.min()) - 2)
    x2 = min(W, int(xs.max()) + 3)
    crop = img[y1:y2, x1:x2]

    args = Namespace(config="carv4", sa=True, dil=True)
    model = torch.nn.DataParallel(models.pidinet_converted(args)).cuda().eval()
    ckpt = torch.load(REPO / "trained_models" / "table5_pidinet.pth", map_location="cpu")
    model.load_state_dict(convert_pidinet(ckpt["state_dict"], "carv4"))

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
    binary[:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, :margin] = 0
    binary[:, -margin:] = 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    if not contours:
        return None, None, None

    contour = max(contours, key=cv2.contourArea)
    return contour, binary.shape, (x1, y1)


def main():
    img_path = TASK_ROOT / "data" / "8.png"
    out_path = TASK_ROOT / "result" / "max" / "8_pidinet_rotated_inscribed_rect.png"
    img = imread(img_path)

    contour, shape, offset = make_pidinet_contour(img)
    if contour is None:
        raise RuntimeError("No PiDiNet contour found")

    h, w = shape
    x1, y1 = offset
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    center = (w / 2.0, h / 2.0)
    best = None
    for angle in np.arange(-15.0, 15.0001, 0.5):
        # Rotate mask by -angle so a rectangle at this angle becomes axis-aligned.
        matrix = cv2.getRotationMatrix2D(center, -float(angle), 1.0)
        rotated = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        rx, ry, rw, rh = [int(v) for v in lir.lir(rotated.astype(bool))]
        area = rw * rh
        if area <= 0:
            continue
        if best is None or area > best["area"]:
            best = {"angle": float(angle), "rect": (rx, ry, rw, rh), "area": area, "matrix": matrix}

    out = img.copy()
    full_contour = contour + np.array([[[x1, y1]]], dtype=contour.dtype)
    cv2.drawContours(out, [full_contour], -1, (0, 0, 255), 1)

    if best is None:
        raise RuntimeError("No inscribed rectangle found")

    rx, ry, rw, rh = best["rect"]
    rect_pts = np.array(
        [[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    inverse = cv2.invertAffineTransform(best["matrix"])
    pts = cv2.transform(rect_pts, inverse).reshape(-1, 2)
    pts += np.array([x1, y1], dtype=np.float32)
    pts = np.round(pts).astype(np.int32)

    cv2.drawContours(out, [pts], 0, (0, 255, 0), 2)
    cv2.putText(
        out,
        f"angle={best['angle']:.2f} deg",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    imwrite(out_path, out)

    print(f"best_angle={best['angle']:.2f}")
    print(f"best_rect_rotated={best['rect']}")
    print(f"area={best['area']}")
    print(f"points={pts.tolist()}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
