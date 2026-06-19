import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(r'C:\Users\lims\Desktop\dev\업무\비전\inspecter\task\성능비교\pidinet')
TASK_ROOT = ROOT.parent
DATA_DIR = TASK_ROOT / "data" / "2" / "cut"
OUT_DIR = TASK_ROOT / "result" / "angles" / "2" / "contour" / "pidinet_raw"
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_pidinet()

    for path in sorted(DATA_DIR.glob("*.png"), key=lambda p: int(p.stem)):
        img = imread(path)
        edge = run_pidinet(model, img)

        _, binary = cv2.threshold(edge, 130, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 20]

        out = img.copy()
        if contours:
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(out, [c], -1, (0, 0, 255), 1)

        imwrite(OUT_DIR / f"{path.stem}_contour_raw.png", out)
        print(f"{path.name} done")


if __name__ == "__main__":
    main()
