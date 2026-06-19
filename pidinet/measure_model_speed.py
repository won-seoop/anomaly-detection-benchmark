import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "pidinet_repo"
DATA_IMAGE = ROOT.parent / "data" / "8.png"


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def load_pidinet():
    sys.path.insert(0, str(REPO))
    import models
    from models.convert_pidinet import convert_pidinet

    args = Namespace(config="carv4", sa=True, dil=True)
    model = torch.nn.DataParallel(models.pidinet_converted(args)).cuda().eval()
    ckpt = torch.load(REPO / "trained_models" / "table5_pidinet.pth", map_location="cpu")
    model.load_state_dict(convert_pidinet(ckpt["state_dict"], "carv4"))
    return model


def pidinet_input():
    img = imread_unicode(DATA_IMAGE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).cuda()


def measure_pidinet(warmup=20, repeats=100):
    model = load_pidinet()
    x = pidinet_input()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
    avg = start.elapsed_time(end) / repeats
    print(f"pidinet_forward_avg_ms={avg:.3f}")
    print(f"warmup={warmup}")
    print(f"repeats={repeats}")


def measure_fastsam(warmup=5, repeats=30):
    from ultralytics import FastSAM

    model = FastSAM(str(ROOT / "FastSAM-s.pt"))
    img = imread_unicode(DATA_IMAGE)
    temp = ROOT / "_speed_fastsam_input.png"
    ok, buf = cv2.imencode(".png", cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
    if not ok:
        raise RuntimeError("encode failed")
    buf.tofile(str(temp))

    for _ in range(warmup):
        _ = model(str(temp), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)
    torch.cuda.synchronize()

    times = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = model(str(temp), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)[0]
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    print(f"fastsam_predict_total_avg_ms={float(np.mean(times)):.3f}")
    print(f"fastsam_predict_total_min_ms={float(np.min(times)):.3f}")
    print(f"fastsam_predict_total_max_ms={float(np.max(times)):.3f}")
    if last is not None:
        print(f"fastsam_ultralytics_preprocess_ms={last.speed.get('preprocess', 0):.3f}")
        print(f"fastsam_ultralytics_inference_ms={last.speed.get('inference', 0):.3f}")
        print(f"fastsam_ultralytics_postprocess_ms={last.speed.get('postprocess', 0):.3f}")
    print(f"warmup={warmup}")
    print(f"repeats={repeats}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["pidinet", "fastsam"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device={torch.cuda.get_device_name(0)}")
    if args.model == "pidinet":
        measure_pidinet()
    else:
        measure_fastsam()


if __name__ == "__main__":
    main()
