"""Download a Roboflow dataset version in YOLO format and make its data.yaml path-safe.

Requires the ROBOFLOW_API_KEY environment variable (Roboflow > Settings > API Keys).
"""

import argparse
import os
import random
from pathlib import Path

import yaml
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", type=int, required=True, help="Dataset version number")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Share of train images held out if the version has no validation split")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY first.")

    location = ROOT / "datasets" / f"{args.project}-v{args.version}"
    project = Roboflow(api_key=api_key).workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov11", location=str(location))
    prepare(Path(dataset.location), args.val_fraction)


def ensure_val_split(root: Path, fraction: float, seed: int = 0):
    """Hold out part of train as valid/ when the Roboflow version was generated without one."""
    if (root / "valid" / "images").is_dir():
        return
    images = sorted((root / "train" / "images").iterdir())
    random.Random(seed).shuffle(images)
    n_val = max(1, round(len(images) * fraction))
    for sub in ("images", "labels"):
        (root / "valid" / sub).mkdir(parents=True, exist_ok=True)
    for image in images[:n_val]:
        image.rename(root / "valid" / "images" / image.name)
        label = root / "train" / "labels" / f"{image.stem}.txt"
        if label.exists():
            label.rename(root / "valid" / "labels" / label.name)
    print(f"Version has no validation split: moved {n_val} of {len(images)} train images to valid/")


def prepare(root: Path, val_fraction: float):
    root = root.resolve()
    ensure_val_split(root, val_fraction)

    # Roboflow writes split paths like "../train/images"; pin them to the dataset root so
    # training works no matter which directory it is launched from.
    data_yaml = root / "data.yaml"
    cfg = yaml.safe_load(data_yaml.read_text())
    cfg["path"] = str(root)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (root / folder / "images").is_dir():
            cfg[split] = f"{folder}/images"
        else:
            cfg.pop(split, None)
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"Dataset ready: {data_yaml}")
    print(f"Classes ({len(cfg['names'])}): {cfg['names']}")


if __name__ == "__main__":
    main()
