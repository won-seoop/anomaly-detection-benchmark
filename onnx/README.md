# ONNX export

FastSAM-s and PiDiNet checkpoints can be exported with one command:

```bash
python onnx/export_models.py all
```

The default checkpoint locations are `FastSAM-s.pt` and
`pidinet/pidinet_repo/trained_models/table5_pidinet.pth`. Generated models are
written to `weights/onnx/` and are ignored by Git.

When checkpoints or the PiDiNet repository are elsewhere, pass them explicitly:

```bash
python onnx/export_models.py all \
  --fastsam-checkpoint /path/to/FastSAM-s.pt \
  --pidinet-repo /path/to/pidinet \
  --pidinet-checkpoint /path/to/table5_pidinet.pth
```

`FastSAM-s.onnx` uses a fixed batch-1 square input (`--imgsz`, default 1024).
`PiDiNet.onnx` accepts dynamic batch, height, and width and returns the final
fused edge probability map named `edge`. Input images for both models are
float32 NCHW tensors; PiDiNet preprocessing remains ImageNet normalization.

## Contour extraction

Run both ONNX models on `data/2/cut`. Raw PiDiNet contours are saved to
`result/onnx/max/pidinet`, and FastSAM contour overlays are saved to
`result/onnx/max/sam`:

```bash
python onnx/extract_contours.py
```

PiDiNet uses the existing raw contour pipeline: full-frame inference, threshold
130, external contours, removal of contours at or below 20 px², and the largest
contour. It does not use tight crop, border removal, or morphology.
