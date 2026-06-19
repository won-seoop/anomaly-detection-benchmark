import argparse
import gc
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "pidinet_repo"
DATA_IMAGE = ROOT.parent / "data" / "8.png"


def mb(value):
    return value / 1024 / 1024


def report(label):
    torch.cuda.synchronize()
    print(f"{label}_allocated_mb={mb(torch.cuda.memory_allocated()):.2f}")
    print(f"{label}_reserved_mb={mb(torch.cuda.memory_reserved()):.2f}")
    print(f"{label}_max_allocated_mb={mb(torch.cuda.max_memory_allocated()):.2f}")
    print(f"{label}_max_reserved_mb={mb(torch.cuda.max_memory_reserved()):.2f}")


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


def measure_pidinet():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_reserved = torch.cuda.memory_reserved()
    before_allocated = torch.cuda.memory_allocated()

    model = load_pidinet()
    report("pidinet_loaded")

    x = pidinet_input()
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    report("pidinet_after_infer")
    print(f"baseline_allocated_mb={mb(before_allocated):.2f}")
    print(f"baseline_reserved_mb={mb(before_reserved):.2f}")

    del model, x
    gc.collect()
    torch.cuda.empty_cache()


def measure_fastsam():
    from ultralytics import FastSAM

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_reserved = torch.cuda.memory_reserved()
    before_allocated = torch.cuda.memory_allocated()

    model = FastSAM(str(ROOT / "FastSAM-s.pt"))
    # Ultralytics lazily moves model to CUDA on first predict.
    report("fastsam_constructed")

    img = imread_unicode(DATA_IMAGE)
    temp = ROOT / "_memory_fastsam_input.png"
    ok, buf = cv2.imencode(".png", cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
    if not ok:
        raise RuntimeError("encode failed")
    buf.tofile(str(temp))

    for _ in range(3):
        _ = model(str(temp), device=0, retina_masks=True, imgsz=1024, conf=0.05, iou=0.7, verbose=False)
    report("fastsam_after_infer")
    print(f"baseline_allocated_mb={mb(before_allocated):.2f}")
    print(f"baseline_reserved_mb={mb(before_reserved):.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


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
