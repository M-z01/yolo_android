# Object Detection Android

Train a YOLO detector with Ultralytics, export it to ONNX, and run it in real time on Android with ONNX Runtime + CameraX.

```
training/   train / export scripts (Python, runs on the workstation GPU)
android/    Android app (Kotlin, CameraX, ONNX Runtime)
```

## One-time setup

```bash
# NVIDIA driver (reboot afterwards; `nvidia-smi` should list the RTX 3080 Ti)
sudo ubuntu-drivers install

# Python environment (conda env with CUDA PyTorch + Ultralytics)
conda activate yolo
pip install -r training/requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # expect True

# Android Studio (bundles the JDK and SDK)
sudo snap install android-studio --classic
```

## Workflow

**1. Smoke-test the app with a pretrained COCO model** (no training needed):

```bash
cd training
python export.py --weights yolo26n.pt     # writes android/app/src/main/assets/model.onnx
```

Open `android/` in Android Studio and run it on a phone (USB debugging on).

**2. Annotate in Roboflow**, generate a dataset version, then download it:

```bash
export ROBOFLOW_API_KEY=...
python download_dataset.py --workspace <ws> --project <project> --version 1
```

**3. Train and export:**

```bash
python train.py --data datasets/<project>-v1/data.yaml --batch 8   # use 8 or less for datasets under ~100 images
python export.py --weights runs/detect/train/weights/best.pt
```

Rebuild the app. Class names are read from the model's metadata, so no label file is needed.

**4. Switch models without rebuilding.** Tap the model name (top-left) in the app to pick a model. To add one:

- From the workstation, with the phone plugged in: `python export.py --weights <path.pt> --push [--name jerry-v2]`
- On the phone: get the `.onnx` file onto it (Drive, chat app, …), then *Import model…* in the picker.

Added models live in `/sdcard/Android/data/com.example.objectdetection/files/models/` and are deleted if the app is uninstalled. The app remembers the last model picked; *Built-in* is `android/app/src/main/assets/model.onnx`.

## Tuning speed

- `export.py --imgsz 320` (or 416) is often 2–4× faster on phones; check the accuracy cost with `yolo val`.
- `yolo26n` is NMS-free (simpler, faster post-processing). `yolo11n` also works: the app detects the output layout and runs NMS itself.
- The on-screen overlay shows the execution provider, per-frame latency (pre + inference + post), and FPS.
