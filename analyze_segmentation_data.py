import cv2
import numpy as np
import os
from pathlib import Path

# Analyze a few samples
seg_dir = Path("segmentation_data/train_valid")
img_dir = seg_dir / "fgt_seg"
label_dir = seg_dir / "fgt_seg_labels"

# Get first 3 samples
label_files = sorted(list(label_dir.glob("*.png")))[:3]

for label_file in label_files:
    img_name = label_file.stem.replace("_LI", "") + label_file.suffix
    img_file = img_dir / img_name
    
    print(f"\n{'='*60}")
    print(f"File: {label_file.name}")
    
    if img_file.exists():
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        print(f"Image shape: {img.shape}")
    
    mask = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values in mask: {np.unique(mask)}")
    print(f"Value distribution:")
    for val in np.unique(mask):
        count = np.sum(mask == val)
        pct = count / mask.size * 100
        tissue_type = {
            0: "Background",
            64: "Adipose tissue",
            128: "Fibroglandular tissue (FGT)",
            192: "Fibroglandular tissue (FGT)",  # Sometimes 192
            255: "Pectoral muscle"
        }.get(val, f"Unknown ({val})")
        print(f"  {val:3d}: {count:7d} pixels ({pct:5.2f}%) - {tissue_type}")

print(f"\n{'='*60}")
print("Summary:")
print("- Background: 0 or 64")
print("- Adipose tissue: 64")
print("- Fibroglandular tissue (FGT - 2nd brightest): 128 or 192")
print("- Pectoral muscle (brightest): 255")
