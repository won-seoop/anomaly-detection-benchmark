"""
FastSAM-s 컨투어 추출 — crimp-compare conda 환경에서 실행.
usage: python run_contour.py
"""

import sys, os
sys.path.insert(0, r"C:\Users\lims\Desktop\dev\업무\비전\inspecter\code\Inspector-compare\compare")

import cv2
import numpy as np
from pathlib import Path
import fastsam_runner as fs

DATA_DIR  = Path(__file__).parent / "data"
OUT_DIR   = Path(__file__).parent / "out_fastsam"
MODEL     = Path(r"C:\Users\lims\Desktop\dev\업무\비전\inspecter\code\Inspector-compare\compare\FastSAM-s.onnx")

OUT_DIR.mkdir(exist_ok=True)

runner = fs.FastSAMRunner(str(MODEL))
params = fs.FastSAMParams()

imgs = sorted(DATA_DIR.glob("*.png")) + sorted(DATA_DIR.glob("*.jpg"))
print(f"Images: {len(imgs)}")

for img_path in imgs:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] {img_path.name}")
        continue

    res = fs.run_on_image(runner, img, params)

    vis = img.copy()
    if res:
        cnt = res["contour"].reshape(-1, 1, 2)
        cv2.drawContours(vis, [cnt], -1, (0, 220, 0), 2)
        cx, cy = res["center"]
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 220, 0), -1)
        angle = res["angle"]
        conf  = res["confidence"]
        ms    = res["inference_ms"]
        label = f"ang={angle:.1f}deg  conf={conf:.2f}  {ms:.0f}ms"
        cv2.putText(vis, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,0), 1)
        print(f"  {img_path.name}: {label}")
    else:
        cv2.putText(vis, "NO DETECTION", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        print(f"  {img_path.name}: NO DETECTION")

    ok, buf = cv2.imencode(".png", vis)
    if ok:
        (OUT_DIR / img_path.name).write_bytes(buf.tobytes())

print(f"\nSaved -> {OUT_DIR}")
