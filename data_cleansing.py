#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CBIS-DDSM Data Cleansing Pipeline (CSV-driven)
- Validates full image + ROI mask + cropped lesion
- Filters empty/corrupt ROI masks
- Removes outliers using ResNeXt50 embeddings
- Saves cleansed dataset as new CSV
- Provides visualization of samples
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# ---------------------------
# Feature Extractor (ResNeXt)
# ---------------------------
class ResNeXtFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnext50_32x4d(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop FC
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

# ---------------------------
# Load Image Safely
# ---------------------------
def load_image(path, mode="RGB"):
    try:
        return Image.open(path).convert(mode)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None

# ---------------------------
# Main Cleansing Function
# ---------------------------
def cleanse_dataset(csv_path, save_csv="cbis_ddsm_cleansed.csv", visualize=True):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ResNeXtFeatureExtractor().to(device).eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    feats, keep_idx = [], []

    with torch.no_grad():
        for idx, row in df.iterrows():
            img = load_image(row["image_file_path"], "RGB")
            mask = load_image(row["roi_mask_file_path"], "L")
            crop = load_image(row["cropped_image_file_path"], "RGB")

            if img is None or mask is None or crop is None:
                continue

            # Check ROI validity (non-empty mask)
            mask_t = transform(mask)
            if mask_t.sum() == 0:
                continue

            # Feature extraction (full + crop)
            img_t = transform(img).unsqueeze(0).to(device)
            crop_t = transform(crop).unsqueeze(0).to(device)
            feat = torch.cat([extractor(img_t), extractor(crop_t)], dim=1)

            feats.append(feat.cpu().numpy())
            keep_idx.append(idx)

    feats = np.vstack(feats)
    print(f"Extracted features for {len(keep_idx)} valid samples")

    # Outlier filtering
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    pred = lof.fit_predict(feats)
    kept = [keep_idx[i] for i in range(len(keep_idx)) if pred[i] == 1]

    df_clean = df.iloc[kept].reset_index(drop=True)
    df_clean.to_csv(save_csv, index=False)
    print(f"Cleansed dataset saved to {save_csv} with {len(df_clean)} samples")

    # ---------------------------
    # Visualization Showcase
    # ---------------------------
    if visualize:
        samples = df_clean.sample(3, random_state=42)
        for _, row in samples.iterrows():
            img = load_image(row["image_file_path"], "RGB")
            mask = load_image(row["roi_mask_file_path"], "L")
            crop = load_image(row["cropped_image_file_path"], "RGB")

            if img is None or mask is None or crop is None:
                continue

            img_np = np.array(img)
            mask_np = np.array(mask)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Full Image")

            plt.subplot(1, 3, 2)
            plt.imshow(crop)
            plt.title("Cropped Lesion")

            plt.subplot(1, 3, 3)
            plt.imshow(img_np)
            plt.imshow(mask_np, cmap="Reds", alpha=0.4)
            plt.title("ROI Overlay")
            plt.show()


if __name__ == "__main__":
    csv_path = "CBIS-DDSM/merge_mass.csv"  # update to your actual CSV
    cleanse_dataset(csv_path, save_csv="cbis_ddsm_cleansed.csv", visualize=True)
