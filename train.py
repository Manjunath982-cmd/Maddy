#!/usr/bin/env python
"""Train YOLOv8n model for food detection.

Usage (from project root):
    python train.py              # trains with default arguments
    python train.py --help       # show all available CLI options

Ensure that you have prepared the dataset directory structure and the data.yaml file
before running this script. See README.md for dataset preparation details.
"""
import argparse
from ultralytics import YOLO


def get_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on custom food dataset")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to dataset YAML file")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pre-trained weights to start from")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training/inference")
    parser.add_argument("--device", type=str, default="0", help="CUDA device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", type=str, default="runs/detect", help="Parent directory for training runs")
    parser.add_argument("--name", type=str, default="yolov8n_food", help="Run name within the project directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint in the same run")
    return parser


def main(args: argparse.Namespace) -> None:
    """Launch the Ultralytics training routine with provided arguments."""
    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)