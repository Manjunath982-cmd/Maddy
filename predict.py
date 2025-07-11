#!/usr/bin/env python
"""Run inference with a trained YOLOv8n model for food detection."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8 inference on images/videos")
    parser.add_argument("--weights", type=str, default="runs/detect/yolov8n_food/weights/best.pt", help="Path to trained weights")
    parser.add_argument("--source", type=str, default="sample.jpg", help="Image, folder, or video path/URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save annotated predictions next to source files")
    return parser


def main(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)

    # Ultralytics handles different input types (str, Path, list, numpy array) seamlessly
    results = model(args.source, conf=args.conf, save=args.save)

    # Display predictions in window for images; for videos it will create an annotated file if --save
    for r in results:
        r.show()

    if args.save:
        print("Annotated results saved to:")
        # Ultralytics returns list of result objects; each has .save_dir attribute for path used
        for r in results:
            print(Path(r.save_dir).resolve())


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)