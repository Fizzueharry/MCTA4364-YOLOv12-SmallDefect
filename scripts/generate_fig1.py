import os
import glob
import yaml
import cv2
import matplotlib.pyplot as plt

with open("configs/defect_data.yaml", "r") as f:
    cfg = yaml.safe_load(f)

class_names = cfg["names"]
img_dir = os.path.join(cfg["path"], "train", "images")
lbl_dir = os.path.join(cfg["path"], "train", "labels")

samples = {}
for lbl_path in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
    with open(lbl_path, "r") as f:
        lines = [l.strip().split() for l in f if len(l.strip().split()) == 5]
    if not lines:
        continue
    cls_ids = {int(l[0]) for l in lines}
    # Pick images that cleanly represent a single defect class
    if len(cls_ids) == 1:
        cid = next(iter(cls_ids))
        if cid not in samples:
            base = os.path.splitext(os.path.basename(lbl_path))[0]
            for ext in (".jpg", ".png", ".jpeg"):
                cand = os.path.join(img_dir, base + ext)
                if os.path.exists(cand):
                    samples[cid] = (cand, lines)
                    break
    if len(samples) == len(class_names):
        break

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
axes = axes.flatten()

for idx, cls_name in enumerate(class_names):
    ax = axes[idx]
    img_path, boxes = samples[idx]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    for b in boxes:
        _, xc, yc, bw, bh = map(float, b)
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 50, 50), 3)

    ax.imshow(img)
    ax.set_title(f"({chr(97+idx)}) {cls_name}", fontsize=12, fontweight="bold")
    ax.axis("off")

plt.suptitle("NEU-DET Surface Defect Dataset Sample Instances (with Ground-Truth Boxes)", fontsize=14, fontweight="bold")
plt.tight_layout()
out_path = "results/eda/figure1_sample_defects.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[SUCCESS] Saved Figure 1 to: {out_path}")