import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import FastSAM


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "pidinet_repo"
sys.path.insert(0, str(REPO))

import models  # noqa: E402
from models.convert_pidinet import convert_pidinet  # noqa: E402


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(str(path))


def make_tight_crop(img):
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


def pidinet_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def time_cuda_forward(fn, repeats=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def run_pidinet(img, out_dir):
    crop, roi = make_tight_crop(img)
    args = Namespace(config="carv4", sa=True, dil=True)
    model = models.pidinet_converted(args)
    model = torch.nn.DataParallel(model).cuda().eval()
    ckpt = torch.load(REPO / "trained_models" / "table5_pidinet.pth", map_location="cpu")
    model.load_state_dict(convert_pidinet(ckpt["state_dict"], "carv4"))

    x = pidinet_tensor(crop).cuda()

    with torch.no_grad():
        infer_ms = time_cuda_forward(lambda: model(x), repeats=50, warmup=10)
        result = torch.squeeze(model(x)[-1]).detach().cpu().numpy()

    edge = (result * 255).astype(np.uint8)
    _, binary = cv2.threshold(edge, 130, 255, cv2.THRESH_BINARY)
    margin = 8
    binary[:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, :margin] = 0
    binary[:, -margin:] = 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    out = crop.copy()
    largest_area = 0.0
    box = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        largest_area = float(cv2.contourArea(c))
        box = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, (0, 0, 255), 1)

    imwrite_unicode(out_dir / "pidinet_largest_loaded.png", out)
    imwrite_unicode(out_dir / "pidinet_binary_loaded.png", binary)
    return {
        "roi": roi,
        "crop_shape": crop.shape,
        "infer_ms": infer_ms,
        "contours": len(contours),
        "largest_area": largest_area,
        "box": box,
    }


def run_fastsam(img, image_path, out_dir):
    model = FastSAM("FastSAM-s.pt")
    upscaled_path = out_dir / "fastsam_s_5_upscaled_input.png"
    upscaled = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    imwrite_unicode(upscaled_path, upscaled)

    for _ in range(3):
        model(str(upscaled_path), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)
    torch.cuda.synchronize()

    repeats = 20
    t0 = time.perf_counter()
    last = None
    for _ in range(repeats):
        last = model(str(upscaled_path), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)[0]
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    speed = last.speed if last is not None else {}

    h, w = img.shape[:2]
    combined = np.zeros((h, w), np.uint8)
    raw = []
    if last is not None and last.masks is not None:
        for mask in last.masks.data.cpu().numpy():
            m = ((mask > 0.5).astype(np.uint8) * 255)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if area > 150 and bw > 40 and 6 < bh < 120:
                    raw.append(c)
                    cv2.drawContours(combined, [c], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 500 and cv2.boundingRect(c)[2] > 80]

    out = img.copy()
    largest_area = 0.0
    box = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        largest_area = float(cv2.contourArea(c))
        box = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, (0, 0, 255), 1)

    imwrite_unicode(out_dir / "fastsam_s_largest_loaded.png", out)
    imwrite_unicode(out_dir / "fastsam_s_mask_loaded.png", combined)
    return {
        "total_ms": total_ms,
        "speed": speed,
        "raw": len(raw),
        "final": len(contours),
        "largest_area": largest_area,
        "box": box,
    }


def main():
    image_path = ROOT.parent / "data" / "5.png"
    out_dir = ROOT.parent / "result" / "max" / "5"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    pidinet = run_pidinet(img, out_dir)
    fastsam = run_fastsam(img, image_path, out_dir)

    info = f"""image: {image_path}

Loaded-model timing, averaged after warmup

PiDiNet
- pure_forward_ms: {pidinet['infer_ms']:.3f}
- roi: {pidinet['roi']}
- crop_shape: {pidinet['crop_shape']}
- contours: {pidinet['contours']}
- largest_area: {pidinet['largest_area']:.1f}
- box: {pidinet['box']}
- output: pidinet_largest_loaded.png

FastSAM small
- loaded_total_predict_ms: {fastsam['total_ms']:.3f}
- ultralytics_preprocess_ms: {fastsam['speed'].get('preprocess', 0):.3f}
- ultralytics_inference_ms: {fastsam['speed'].get('inference', 0):.3f}
- ultralytics_postprocess_ms: {fastsam['speed'].get('postprocess', 0):.3f}
- raw_masks_kept: {fastsam['raw']}
- final_contours: {fastsam['final']}
- largest_area: {fastsam['largest_area']:.1f}
- box: {fastsam['box']}
- output: fastsam_s_largest_loaded.png
"""
    (out_dir / "loaded_model_info.txt").write_text(info, encoding="utf-8")
    print(info)


if __name__ == "__main__":
    main()
