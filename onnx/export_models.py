#!/usr/bin/env python3
"""Export FastSAM-s and PiDiNet checkpoints to ONNX."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import onnx
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "weights" / "onnx"


class PiDiNetFinalOutput(torch.nn.Module):
    """Expose only the fused edge map used by this project."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)[-1]


def validate_onnx(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    inputs = ", ".join(value.name for value in model.graph.input)
    outputs = ", ".join(value.name for value in model.graph.output)
    print(f"validated: {path} (inputs={inputs}; outputs={outputs})")


def export_fastsam(checkpoint: Path, output_dir: Path, imgsz: int, opset: int) -> Path:
    from ultralytics import FastSAM

    if not checkpoint.is_file():
        raise FileNotFoundError(f"FastSAM checkpoint not found: {checkpoint}")

    destination = output_dir / "FastSAM-s.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fastsam_onnx_") as temp_dir:
        temporary_checkpoint = Path(temp_dir) / "FastSAM-s.pt"
        shutil.copy2(checkpoint, temporary_checkpoint)
        model = FastSAM(str(temporary_checkpoint))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=imgsz,
                opset=opset,
                simplify=True,
                dynamic=False,
                batch=1,
            )
        )
        shutil.move(str(exported), destination)
    validate_onnx(destination)
    return destination


def export_pidinet(
    checkpoint: Path,
    repo: Path,
    output_dir: Path,
    height: int,
    width: int,
    opset: int,
) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PiDiNet checkpoint not found: {checkpoint}")
    if not (repo / "models" / "convert_pidinet.py").is_file():
        raise FileNotFoundError(f"PiDiNet repository not found: {repo}")

    sys.path.insert(0, str(repo))
    import models  # type: ignore
    from models.convert_pidinet import convert_pidinet  # type: ignore

    args = Namespace(config="carv4", sa=True, dil=True)
    model = models.pidinet_converted(args).cpu().eval()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = loaded["state_dict"] if "state_dict" in loaded else loaded
    state_dict = convert_pidinet(state_dict, "carv4")
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    export_model = PiDiNetFinalOutput(model).eval()

    destination = output_dir / "PiDiNet.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, height, width, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            sample,
            destination,
            input_names=["images"],
            output_names=["edge"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "edge": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    validate_onnx(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("all", "fastsam", "pidinet"), nargs="?", default="all")
    parser.add_argument("--fastsam-checkpoint", type=Path, default=ROOT / "FastSAM-s.pt")
    parser.add_argument(
        "--pidinet-checkpoint",
        type=Path,
        default=ROOT / "pidinet" / "pidinet_repo" / "trained_models" / "table5_pidinet.pth",
    )
    parser.add_argument("--pidinet-repo", type=Path, default=ROOT / "pidinet" / "pidinet_repo")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=1024, help="FastSAM square input size")
    parser.add_argument("--pidinet-height", type=int, default=512)
    parser.add_argument("--pidinet-width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model in ("all", "fastsam"):
        export_fastsam(args.fastsam_checkpoint.resolve(), args.output_dir.resolve(), args.imgsz, args.opset)
    if args.model in ("all", "pidinet"):
        export_pidinet(
            args.pidinet_checkpoint.resolve(),
            args.pidinet_repo.resolve(),
            args.output_dir.resolve(),
            args.pidinet_height,
            args.pidinet_width,
            args.opset,
        )


if __name__ == "__main__":
    main()
