# Food Detection with YOLOv8n

This project demonstrates how to train and deploy a **YOLOv8n** object-detection model that can classify and locate four broad food categories:

1. **Indian food**  
2. **Foreign food**  
3. **Fruits**  
4. **Vegetables**

The sample workflow assumes **≈ 1,000 annotated images** split into *train* and *validation* subsets. You can extend the dataset or refine class granularity at any time (e.g. split dishes into individual classes).

---

## 1 — Environment setup

```bash
# Clone repo / open project folder first, then:
python -m venv .venv      # optional but recommended
source .venv/bin/activate # Linux & macOS

pip install -r requirements.txt
```

> **CUDA / GPU**: If you have an NVIDIA GPU, make sure that the corresponding `torch` wheel (CUDA 11.x/12.x) is installed. The Ultralytics package will attempt to pick the correct version; see the [PyTorch installation guide](https://pytorch.org/get-started/locally/) if you need a specific wheel.

---

## 2 — Dataset preparation

YOLOv8 uses the same text-based annotation format introduced by YOLOv5: one `*.txt` file per image, each containing one row per object:

```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are **normalized** between 0 and 1 relative to the image size. Recommended annotation tools:

* **[LabelImg](https://github.com/heartexlabs/labelImg)** ‑ open-source desktop tool  
* **[Roboflow](https://roboflow.com/)** ‑ web interface with auto-format export  
* **CVAT**, Label Studio, or any tool that can export the YOLO text format.

Directory structure expected by `data.yaml`:

```
project_root/
├── data/
│   ├── images/
│   │   ├── train/    # *.jpg / *.png
│   │   └── val/
│   └── labels/
│       ├── train/    # *.txt (same filename as images)
│       └── val/
└── data.yaml         # dataset configuration (already provided)
```

If you also have a **test** split, add it to `data.yaml`:
```yaml
path: data
train: images/train
val: images/val
test: images/test
```

---

## 3 — Training

Run with default arguments:
```bash
python train.py
```
Key flags you might want to tweak:
* `--epochs` (default 100)
* `--batch`  (depends on GPU VRAM, default 16)
* `--imgsz`  (input size, default 640)

The script downloads the base **yolov8n.pt** weights automatically (≈6 MB) on first use. Training outputs (checkpoints, metrics, TensorBoard logs) land in `runs/detect/yolov8n_food/` by default.

### Resume training
```bash
python train.py --resume --name yolov8n_food
```

---

## 4 — Inference

Predict on a single image and **display** the result:
```bash
python predict.py --weights runs/detect/yolov8n_food/weights/best.pt --source sample.jpg
```
Predict on a **folder** of images and **save** annotated copies next to originals:
```bash
python predict.py --source path/to/images --save
```

---

## 5 — Evaluation / mAP

Ultralytics automatically evaluates mAP@0.5 and mAP@0.5:0.95 at the end of every epoch. You can run a standalone evaluation later as:
```bash
from ultralytics import YOLO
model = YOLO('runs/detect/yolov8n_food/weights/best.pt')
metrics = model.val(data='data.yaml')
print(metrics.box.map50, metrics.box.map)
```

---

## 6 — Export to other frameworks

```python
model = YOLO('runs/detect/yolov8n_food/weights/best.pt')
model.export(format="onnx")   # or "openvino", "tflite", "coreml", "tensorRT", etc.
```

---

## 7 — Tips & best practices

* **Balance the dataset**: strive for similar image counts and object counts per class.
* **Augment strategically**: Ultralytics has strong default augmentations, but you can tune them via the `augment` and `degrees`/`scale`/`flipud` etc. parameters.
* **Label quality > quantity**: 1,000 well-labeled images can beat 10,000 noisy ones.
* **Monitor over-fitting**: watch the validation loss / mAP curves on TensorBoard.

Happy modelling! 🎉