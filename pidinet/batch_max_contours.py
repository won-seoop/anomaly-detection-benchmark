import csv
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import FastSAM


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT.parent
DATA_DIR = TASK_ROOT / "data"
OUT_ROOT = TASK_ROOT / "result" / "max"
REPO = ROOT / "pidinet_repo"
TEMP_DIR = ROOT / "_batch_tmp"

sys.path.insert(0, str(REPO))
import models  # noqa: E402
from models.convert_pidinet import convert_pidinet  # noqa: E402


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(str(path))


def tight_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row_mean = gray.mean(axis=1)
    rows = row_mean > max(80, float(np.percentile(row_mean, 45)))
    rows[:5] = False
    rows[-5:] = False
    ys = np.where(rows)[0]
    y1 = max(0, int(ys.min()) - 3) if len(ys) else 0
    y2 = min(h, int(ys.max()) + 4) if len(ys) else h

    col_mean = gray[y1:y2].mean(axis=0)
    cols = col_mean > max(50, float(np.percentile(col_mean, 15)))
    xs = np.where(cols)[0]
    x1 = max(0, int(xs.min()) - 2) if len(xs) else 0
    x2 = min(w, int(xs.max()) + 3) if len(xs) else w
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def pidinet_input(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def load_pidinet():
    args = Namespace(config="carv4", sa=True, dil=True)
    model = models.pidinet_converted(args)
    model = torch.nn.DataParallel(model).cuda().eval()
    ckpt = torch.load(REPO / "trained_models" / "table5_pidinet.pth", map_location="cpu")
    model.load_state_dict(convert_pidinet(ckpt["state_dict"], "carv4"))
    return model


def cuda_time_ms(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def run_pidinet(model, img):
    crop, roi = tight_crop(img)
    x1, y1, _, _ = roi
    x = pidinet_input(crop).cuda()

    with torch.no_grad():
        for _ in range(3):
            model(x)
        result, infer_ms = cuda_time_ms(lambda: model(x))

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

    out = img.copy()
    area = 0.0
    box = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        c_full = c + np.array([[[x1, y1]]], dtype=c.dtype)
        box = cv2.boundingRect(c_full)
        cv2.drawContours(out, [c_full], -1, (0, 0, 255), 1)

    return out, binary, {
        "infer_ms": infer_ms,
        "roi": roi,
        "contours": len(contours),
        "largest_area": area,
        "box": box,
    }


def run_fastsam(model, img, stem):
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    temp_path = TEMP_DIR / f"{stem}_fastsam_input.png"
    imwrite_unicode(temp_path, upscaled)

    for _ in range(2):
        model(str(temp_path), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    result = model(str(temp_path), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)[0]
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0

    combined = np.zeros((h, w), np.uint8)
    raw = []
    if result.masks is not None:
        for mask in result.masks.data.cpu().numpy():
            m = ((mask > 0.5).astype(np.uint8) * 255)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                is_full_frame = bw >= w - 2 and bh >= h - 2 and area > 0.65 * w * h
                if area > 150 and bw > 40 and 6 < bh < 140 and not is_full_frame:
                    raw.append(c)
                    cv2.drawContours(combined, [c], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 500 and cv2.boundingRect(c)[2] > 80]

    out = img.copy()
    area = 0.0
    box = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        box = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, (0, 0, 255), 1)

    speed = result.speed or {}
    return out, combined, {
        "total_ms": total_ms,
        "preprocess_ms": float(speed.get("preprocess", 0.0)),
        "inference_ms": float(speed.get("inference", 0.0)),
        "postprocess_ms": float(speed.get("postprocess", 0.0)),
        "raw": len(raw),
        "contours": len(contours),
        "largest_area": area,
        "box": box,
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATA_DIR.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    pidinet = load_pidinet()
    fastsam = FastSAM("FastSAM-s.pt")

    rows = []
    for path in image_paths:
        img = imread_unicode(path)
        if img is None:
            print(f"skip unreadable: {path}")
            continue

        out_dir = OUT_ROOT / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        p_img, p_mask, p_info = run_pidinet(pidinet, img)
        f_img, f_mask, f_info = run_fastsam(fastsam, img, path.stem)

        imwrite_unicode(out_dir / "pidinet_largest.png", p_img)
        imwrite_unicode(out_dir / "pidinet_binary.png", p_mask)
        imwrite_unicode(out_dir / "fastsam_s_largest.png", f_img)
        imwrite_unicode(out_dir / "fastsam_s_mask.png", f_mask)

        info = f"""image: {path}

PiDiNet
- pure_forward_ms: {p_info['infer_ms']:.3f}
- roi: {p_info['roi']}
- contours: {p_info['contours']}
- largest_area: {p_info['largest_area']:.1f}
- box: {p_info['box']}
- output: pidinet_largest.png

FastSAM small
- loaded_predict_total_ms: {f_info['total_ms']:.3f}
- ultralytics_preprocess_ms: {f_info['preprocess_ms']:.3f}
- ultralytics_inference_ms: {f_info['inference_ms']:.3f}
- ultralytics_postprocess_ms: {f_info['postprocess_ms']:.3f}
- raw_masks_kept: {f_info['raw']}
- final_contours: {f_info['contours']}
- largest_area: {f_info['largest_area']:.1f}
- box: {f_info['box']}
- output: fastsam_s_largest.png
"""
        (out_dir / "info.txt").write_text(info, encoding="utf-8")

        rows.append({
            "image": path.name,
            "pidinet_forward_ms": f"{p_info['infer_ms']:.3f}",
            "pidinet_contours": p_info["contours"],
            "pidinet_largest_area": f"{p_info['largest_area']:.1f}",
            "pidinet_box": p_info["box"],
            "fastsam_total_ms": f"{f_info['total_ms']:.3f}",
            "fastsam_inference_ms": f"{f_info['inference_ms']:.3f}",
            "fastsam_contours": f_info["contours"],
            "fastsam_largest_area": f"{f_info['largest_area']:.1f}",
            "fastsam_box": f_info["box"],
        })
        print(f"{path.name}: pidinet {p_info['infer_ms']:.2f}ms/{p_info['contours']}c, fastsam {f_info['inference_ms']:.2f}ms/{f_info['contours']}c")

    with (OUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
