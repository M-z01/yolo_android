"""Fine-tune a YOLO detection model on a dataset described by a data.yaml."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="Starting weights (yolo26n/s, yolo11n/s, ...)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="Keep well below the train image count so each epoch has several updates; "
                             "32 fits n/s models at 640 on a 12 GB GPU")
    parser.add_argument("--patience", type=int, default=30,
                        help="Stop after this many epochs without improvement; 0 disables (tiny val sets are noisy)")
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'")
    parser.add_argument("--name", default="train", help="Run name under runs/detect/")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        patience=args.patience,
    )

    best = Path(model.trainer.best)
    print(f"\nBest weights: {best}")
    print(f"Export for Android: python export.py --weights {best}")


if __name__ == "__main__":
    main()
