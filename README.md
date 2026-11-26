# Improving Small-Defect Detection in YOLOv12 through High-Resolution Attention and Small-Object Loss

**Course:** MCTA 4364 — Machine Vision (Semester 1, 2026/2027)
**Track:** Track 2 — Research Paper Extension
**Base Paper:** Tian, Y., Ye, Q., & Doermann, D. (2025). *YOLOv12: Attention-Centric Real-Time Object Detectors*. arXiv:2502.12524v1 ([Official Repository](https://github.com/sunsmarterjie/yolov12))
**Dataset:** Northeastern University Surface Defect Database (NEU-DET, CC BY 4.0)

---

## Group Members
| Name | Matric No. | Role & Contributions |
| :--- | :---: | :--- |
| **Mohd Nafiz Zuhairee Bin Mohd Neezar** | `2317423` | Environment setup, model training pipeline, baseline evaluation, architectural extension design |
| **Wan Syazmie Akmal bin Wan Saad** | `2316957` | Dataset curation, exploratory data analysis (EDA), literature synthesis, report documentation |
| **Muhammad Aiman bin Safawi** | `2318249` | Qualitative error analysis, baseline validation verification, hyperparameter & metrics logging |

---

## Repository Structure
```text
MCTA4364-YOLOv12-SmallDefect/
├── configs/
│   └── defect_data.yaml          # NEU-DET dataset paths and 6 class definitions
├── scripts/
│   ├── eda_dataset.py            # Exploratory data analysis and bounding-box scale profiling
│   └── generate_fig1.py          # Sample defect collage generator
├── results/
│   ├── eda/                      # Class distribution, scatter plot, and sample collage
│   └── baseline/                 # Training curves, confusion matrix, and validation/test CSV logs
└── README.md
```

---

## Environment Setup
1. **Create and activate Conda environment (Python 3.10 + PyTorch CUDA 11.8):**
```bash
conda create -p D:\conda_envs\yolov12_mv python=3.10 -y
conda activate D:\conda_envs\yolov12_mv
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.3.63 thop seaborn pandas pyyaml
```

2. **Dataset Preparation:**
Download the NEU-DET dataset (YOLOv12 format, 1,799 images across 6 classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) and extract into `datasets/raw/` so that `train/images`, `valid/images`, and `test/images` align with `configs/defect_data.yaml`.

---

## How to Run the Baseline
1. **Run Exploratory Data Analysis (EDA):**
```bash
python scripts/eda_dataset.py
```

2. **Train YOLOv12n Baseline (50 Epochs, 640x640):**
```bash
yolo detect train data=configs/defect_data.yaml model=yolov12n.pt epochs=50 imgsz=640 batch=8 workers=0 device=0 project=runs/baseline name=yolov12n_neudet_base
```

3. **Evaluate Baseline on Held-Out Test Split:**
```bash
yolo detect val data=configs/defect_data.yaml model=runs/baseline/yolov12n_neudet_base/weights/best.pt split=test imgsz=640 batch=8 workers=0 device=0 project=runs/baseline name=yolov12n_neudet_test
```

---

## Baseline Results Summary (YOLOv12n)
- **Parameters / FLOPs:** 2,509,514 parameters | 5.8 GFLOPs (fused)
- **Validation Set (360 images, 870 instances):** `Precision = 0.746`, `Recall = 0.722`, `mAP50 = 0.784`, `mAP50-95 = 0.433` (10.6 ms/image)
- **Test Set (180 images, 392 instances):** `Precision = 0.704`, `Recall = 0.735`, `mAP50 = 0.754`, `mAP50-95 = 0.443` (11.9 ms/image)
