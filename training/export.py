"""Export YOLO weights to ONNX, sanity-check them with ONNX Runtime, and install into the Android app."""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ANDROID_MODEL = ROOT.parent / "android" / "app" / "src" / "main" / "assets" / "model.onnx"
# The app's model folder (ModelStore.directory); adb can write here without root.
PHONE_MODEL_DIR = "/sdcard/Android/data/com.example.objectdetection/files/models"


def check(onnx_path: Path):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    names = session.get_modelmeta().custom_metadata_map.get("names", "<missing>")
    print(f"input  {inp.name} {inp.shape}")
    print(f"output {out.name} {out.shape}")
    print(f"names  {names}")

    dummy = np.random.rand(*inp.shape).astype(np.float32)
    session.run(None, {inp.name: dummy})  # warm-up
    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {inp.name: dummy})
    print(f"desktop CPU latency: {(time.perf_counter() - start) / runs * 1000:.1f} ms")


def default_name(weights: str, imgsz: int) -> str:
    path = Path(weights)
    # runs/detect/<run>/weights/last.pt -> <run>-last
    stem = f"{path.parent.parent.name}-{path.stem}" if path.parent.name == "weights" else path.stem
    return f"{stem}-{imgsz}"


def find_adb() -> str:
    if adb := shutil.which("adb"):
        return adb
    sdk = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") or Path.home() / "Android" / "Sdk"
    adb = Path(sdk) / "platform-tools" / "adb"
    if not adb.exists():
        raise SystemExit("adb not found: install Android platform-tools or set ANDROID_HOME")
    return str(adb)


def push(onnx_path: Path, name: str):
    adb = find_adb()
    # Don't create the folder here: the app can't read a folder created by adb.
    if subprocess.run([adb, "shell", "test", "-d", PHONE_MODEL_DIR]).returncode != 0:
        raise SystemExit("Model folder not found on the phone: install the app and open it once, then retry.")
    subprocess.run([adb, "push", str(onnx_path), f"{PHONE_MODEL_DIR}/{name}.onnx"], check=True)
    print(f"Pushed to the phone as '{name}'. In the app, tap the model name to switch to it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="e.g. runs/detect/train/weights/best.pt, or yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Try 320/416 for faster on-device inference")
    parser.add_argument("--opset", type=int, default=17, help="Pinned so the mobile runtime is guaranteed to support it")
    parser.add_argument("--no-install", action="store_true", help="Don't copy the model into the Android assets")
    parser.add_argument("--push", action="store_true",
                        help="Send the model to the connected phone's model folder instead of replacing the "
                             "built-in model; no app rebuild needed")
    parser.add_argument("--name", help="Model name shown in the app with --push (default: <run>-<weights>-<imgsz>)")
    args = parser.parse_args()

    # nms=False selects the NMS-free [1, 300, 6] head on models that have one (YOLO26); others ignore it.
    onnx_path = Path(
        YOLO(args.weights).export(
            format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False, nms=False
        )
    )
    check(onnx_path)

    if args.push:
        push(onnx_path, args.name or default_name(args.weights, args.imgsz))
    elif not args.no_install:
        ANDROID_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, ANDROID_MODEL)
        print(f"Installed to {ANDROID_MODEL}")


if __name__ == "__main__":
    main()
