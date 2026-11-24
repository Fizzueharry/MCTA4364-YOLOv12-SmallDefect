import os
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("configs/defect_data.yaml", "r") as f:
    cfg = yaml.safe_load(f)

class_names = cfg["names"]
base_dir = cfg["path"]
splits = ["train", "valid", "test"]

records = []
for split in splits:
    label_files = glob.glob(os.path.join(base_dir, split, "labels", "*.txt"))
    for lf in label_files:
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    area_ratio = w * h
                    size_cat = "Small (<32x32 px)" if (w * 640 < 32 or h * 640 < 32) else (
                        "Medium (32-96 px)" if (w * 640 < 96 and h * 640 < 96) else "Large (>96 px)"
                    )
                    records.append({
                        "Split": split,
                        "Class": class_names[cls_id],
                        "Width_px": w * 640,
                        "Height_px": h * 640,
                        "Area_Ratio": area_ratio,
                        "Scale_Category": size_cat
                    })

df = pd.DataFrame(records)
print("=== RINGKASAN DATASET NEU-DET ===")
print(f"Jumlah Keseluruhan Bounding Box: {len(df)}")
print("\nTaburan Mengikut Split:")
print(df["Split"].value_counts())
print("\nTaburan Saiz Kecacatan (Skala 640x640):")
print(df["Scale_Category"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

df.groupby(["Class", "Scale_Category"]).size().unstack(fill_value=0).to_csv("results/eda/class_scale_summary.csv")

plt.figure(figsize=(11, 5))
sns.countplot(data=df, x="Class", hue="Scale_Category", palette="viridis")
plt.title("NEU-DET Defect Distribution by Class and Bounding Box Scale (640x640)")
plt.ylabel("Number of Defect Instances")
plt.xlabel("Defect Class")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/eda/defect_scale_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
sns.scatterplot(data=df[df["Split"] == "train"], x="Width_px", y="Height_px", hue="Class", alpha=0.6, s=25)
plt.axvline(32, color="red", linestyle="--", label="Small Object Threshold (32 px)")
plt.axhline(32, color="red", linestyle="--")
plt.title("Bounding Box Width vs. Height Distribution (Train Set)")
plt.xlabel("Bounding Box Width (pixels)")
plt.ylabel("Bounding Box Height (pixels)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("results/eda/bbox_dimensions_scatter.png", dpi=300)
plt.close()

print("\n[BERJAYA] Graf EDA dan jadual statistik disimpan di: results/eda/")