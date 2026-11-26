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
│   └── eda_dataset.py            # Exploratory data analysis and bounding-box scale profiling
├── results/
│   ├── eda/                      # Class distribution and bounding-box dimension plots
│   └── baseline/                 # Training curves, confusion matrix, and validation/test CSV logs
└── README.md
---

## Environment Setup
1. **Create and activate Conda environment (Python 3.10 + PyTorch CUDA 11.8):**
   ```bash
   conda create -p D:\conda_envs\yolov12_mv python=3.10 -y
   conda activate D:\conda_envs\yolov12_mv
   pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
   pip install ultralytics==8.3.63 thop seaborn pandas pyyaml
