"""Export a dinov3_ltdetr_object_detection model to ONNX format."""

import argparse
from pathlib import Path

import torch

from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
    DINOv3LTDETRObjectDetection,
)
from lightly_train._task_models.task_model_helpers import (
    download_checkpoint,
    init_model_from_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a dinov3_ltdetr_object_detection model to ONNX."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).parent / "exported_best.pt"),
        help="Model name or path to a local checkpoint.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the ONNX model. Defaults to '<decoder>_<precision>.onnx'.",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["rtdetrv2", "dfine"],
        default=None,
        help="Override the decoder. Uses the checkpoint's decoder if not set.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "fp32", "fp16", "mixed"],
        default="fp32",
        help="Precision for the ONNX model. 'mixed' keeps normalization and reduction ops in fp32.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the ONNX input.",
    )
    parser.add_argument(
        "--dynamic-batch-size",
        action="store_true",
        help="Use a dynamic batch dimension in the ONNX graph.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version. Uses PyTorch's default if not set.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxslim simplification.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX output verification.",
    )
    args = parser.parse_args()

    ckpt_path = download_checkpoint(checkpoint=args.model)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    init_args = ckpt["model_init_args"]
    decoder = args.decoder or init_args.get("decoder_name", "rtdetrv2")

    if args.decoder == "rtdetrv2":
        model = DINOv3LTDETRObjectDetection(
            model_name=init_args["model_name"],
            classes=init_args["classes"],
            image_size=tuple(init_args["image_size"]),
            image_normalize=init_args.get("image_normalize"),
            decoder_name="rtdetrv2",
            load_weights=False,
        )
    else:
        #if args.decoder is not None:
        #    ckpt["model_init_args"]["decoder_name"] = args.decoder
        model = init_model_from_checkpoint(checkpoint=ckpt)
    model.eval()

    out = args.out
    if out is None:
        name = f"{decoder}_{args.precision}"
        if args.opset is not None:
            name += f"_opset{args.opset}"
        if args.no_simplify:
            name += "_no_simplify"
        out = Path(f"{name}.onnx")

    model.export_onnx(
        out=out,
        precision=args.precision,
        batch_size=args.batch_size,
        dynamic_batch_size=args.dynamic_batch_size,
        opset_version=args.opset,
        simplify=not args.no_simplify,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
