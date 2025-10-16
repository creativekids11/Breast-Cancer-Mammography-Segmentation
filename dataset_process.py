#!/usr/bin/env python3
"""
Unified dataset prep for CBIS-DDSM and Mini-DDSM.

This version:
 - Simplified to only process CBIS-DDSM and Mini-DDSM datasets,
 - Simplified preprocessing: applies constant CLAHE and median filtering,
 - Removed texture map calculations.
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import yaml
from typing import Tuple

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_class_map(yaml_path: str):
    """Load class_id → name mapping from data.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {i: name for i, name in enumerate(data.get("names", []))}

# ---------------- Preprocessing ---------------- #
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Simplified preprocessing: constant CLAHE and median blur.
    Returns uint8 grayscale image.
    """
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img.astype(np.uint8)

# ---------------- INBREAST label parsing ---------------- #
def yolo_to_mask_with_classes(label_path: str, img_shape: Tuple[int,int], class_map, debug=False):
    """
    Convert YOLOv11 labels to binary mask + class list.
    Handles:
      - bbox lines:  class x_center y_center width height
      - bbox+conf lines: class x y w h conf  (ignores conf)
      - polygon lines: class x1 y1 x2 y2 x3 y3 ... (normalized coords)
    Returns (mask, [unique_class_names], debug_info)
    """
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    classes = []
    debug_polys = []  # keep for optional debug overlay

    if not os.path.exists(label_path):
        return mask, classes, debug_polys

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            # skip malformed / too short
            continue
        # parse class id robustly
        try:
            cls_id = int(float(parts[0]))
        except Exception:
            continue
        cls_name = class_map.get(cls_id, f"class_{cls_id}")
        classes.append(cls_name)

        coords = [float(x) for x in parts[1:]]
        # If odd number of coords and >4, possibly trailing confidence — drop it.
        if len(coords) > 4 and (len(coords) % 2 == 1):
            # If last token is small (<=1.0) and others are normalized, treat as extra and drop
            if 0.0 <= coords[-1] <= 1.0:
                coords = coords[:-1]
            else:
                # last token seems absolute pixel (rare); attempt to drop anyway
                coords = coords[:-1]

        # bbox (x_center, y_center, w, h)
        if len(coords) == 4:
            x_c, y_c, w_rel, h_rel = coords
            x1 = int(round((x_c - w_rel / 2.0) * W))
            y1 = int(round((y_c - h_rel / 2.0) * H))
            x2 = int(round((x_c + w_rel / 2.0) * W))
            y2 = int(round((y_c + h_rel / 2.0) * H))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                debug_polys.append(((x1,y1,x2,y2), cls_name))
            continue

        # bbox with conf (ignore last value) - previously handled by trimming odd length,
        # but in some cases you might have 5 values where 5th is conf; handle safe:
        if len(coords) == 5:
            x_c, y_c, w_rel, h_rel = coords[:4]
            x1 = int(round((x_c - w_rel / 2.0) * W))
            y1 = int(round((y_c - h_rel / 2.0) * H))
            x2 = int(round((x_c + w_rel / 2.0) * W))
            y2 = int(round((y_c + h_rel / 2.0) * H))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                debug_polys.append(((x1,y1,x2,y2), cls_name))
            continue

        # polygon (x1 y1 x2 y2 x3 y3 ...)
        if len(coords) >= 6 and (len(coords) % 2 == 0):
            pts = []
            # detect whether coords are normalized (0..1) or absolute (>1)
            # We'll treat values <=1 as normalized
            for i in range(0, len(coords), 2):
                x_rel = coords[i]; y_rel = coords[i+1]
                if 0.0 <= x_rel <= 1.0 and 0.0 <= y_rel <= 1.0:
                    x_px = int(round(x_rel * W))
                    y_px = int(round(y_rel * H))
                else:
                    x_px = int(round(x_rel))
                    y_px = int(round(y_rel))
                x_px = max(0, min(W-1, x_px))
                y_px = max(0, min(H-1, y_px))
                pts.append([x_px, y_px])
            if len(pts) >= 3:
                try:
                    pts_np = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(mask, [pts_np], 255)
                    debug_polys.append((pts_np, cls_name))
                except Exception:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    x1, x2 = max(0, min(xs)), min(W-1, max(xs))
                    y1, y2 = max(0, min(ys)), min(H-1, max(ys))
                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        debug_polys.append(((x1,y1,x2,y2), cls_name))
            continue

        # otherwise skip
        continue

    unique_classes = list(dict.fromkeys(classes))
    return mask, unique_classes, debug_polys

# ---------------- CBIS / MIAS processing (simplified) ---------------- #
def process_cbis(input_csv, mask_outdir, image_outdir):
    df = pd.read_csv(input_csv)
    ensure_dir(mask_outdir); ensure_dir(image_outdir)
    rows = []
    grouped = df.groupby(["patient_id", "image_file_path"], dropna=False)
    for (pid, img_path), group in grouped:
        base_row = group.iloc[0].to_dict()
        abnormality_ids = group["abnormality_id"].astype(str).unique().tolist()
        mask_paths = [mp for mp in group["roi_mask_file_path"].dropna().unique().tolist() if isinstance(mp, str)]
        merged_mask = None
        for mp in mask_paths:
            if not os.path.exists(mp):
                continue
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8) * 255
            merged_mask = mask if merged_mask is None else cv2.bitwise_or(merged_mask, mask)
        if os.path.exists(img_path):
            full_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if full_img is None:
                continue
            processed_img = preprocess_image(full_img)
        else:
            continue
        if merged_mask is None:
            merged_mask = np.zeros_like(processed_img, dtype=np.uint8)
        # ensure shapes match
        if merged_mask.shape != processed_img.shape:
            merged_mask = cv2.resize(merged_mask, (processed_img.shape[1], processed_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        abn_str = "-".join(abnormality_ids) if abnormality_ids else "NA"
        unique_name = f"CBIS_{pid}_{basename}_{abn_str}"
        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path = os.path.join(mask_outdir, f"{unique_name}_mask.png")
        cv2.imwrite(proc_img_path, processed_img)
        cv2.imwrite(mask_path, merged_mask)
        base_row["dataset"] = "CBIS"
        base_row["image_file_path"] = proc_img_path
        base_row["roi_mask_file_path"] = mask_path
        rows.append(base_row)
    return pd.DataFrame(rows)

def process_mias_from_csv(info_csv_path, images_dir, mask_outdir, image_outdir):
    ensure_dir(mask_outdir); ensure_dir(image_outdir)
    df_info = pd.read_csv(info_csv_path)
    rows = []
    for idx, row in df_info.iterrows():
        refnum = row["REFNUM"]
        img_filename = f"{refnum}.png"
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_proc = preprocess_image(img)
        is_normal = (row["CLASS"] == "NORM")
        mask = np.zeros_like(img_proc, dtype=np.uint8)
        if not is_normal and not pd.isna(row["X"]) and not pd.isna(row["Y"]) and not pd.isna(row["RADIUS"]):
            x, y, r = int(row["X"]), int(row["Y"]), int(row["RADIUS"])
            cv2.circle(mask, (x, y), r, 255, -1)
        pid = f"MIAS_{refnum}"
        proc_img_path = os.path.join(image_outdir, f"{pid}.png")
        mask_path = os.path.join(mask_outdir, f"{pid}_mask.png")
        cv2.imwrite(proc_img_path, img_proc)
        cv2.imwrite(mask_path, mask)
        pathology = "N" if is_normal else row["SEVERITY"]
        abnormality_id = "None" if is_normal else row["CLASS"]
        row_data = {
            "dataset": "MIAS",
            "patient_id": pid,
            "image_file_path": proc_img_path,
            "roi_mask_file_path": mask_path,
            "pathology": pathology,
            "abnormality_id": abnormality_id,
        }
        rows.append(row_data)
    return pd.DataFrame(rows)

# ---------------- Dataset Selection ---------------- #
def process_datasets(cbis_csv, mias_info_csv, mias_images_dir, output_csv, outdir):
    cbis_img_dir = os.path.join(outdir, "CBIS_IMAGES"); cbis_mask_dir = os.path.join(outdir, "CBIS_MASKS")
    mias_img_dir = os.path.join(outdir, "MIAS_IMAGES"); mias_mask_dir = os.path.join(outdir, "MIAS_MASKS")

    ensure_dir(cbis_img_dir); ensure_dir(cbis_mask_dir)
    ensure_dir(mias_img_dir); ensure_dir(mias_mask_dir)

    cbis_df = process_cbis(cbis_csv, cbis_mask_dir, cbis_img_dir)
    mias_df = process_mias_from_csv(mias_info_csv, mias_images_dir, mias_mask_dir, mias_img_dir)

    merged = pd.concat([cbis_df, mias_df], ignore_index=True)
    merged.to_csv(output_csv, index=False)
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS={len(cbis_df)}, MIAS={len(mias_df)})")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + MIAS unified dataset")
    p.add_argument("--cbis-csv", type=str, required=True)
    p.add_argument("--mias-info-csv", type=str, required=True)
    p.add_argument("--mias-images-dir", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="DATASET")

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    process_datasets(
        args.cbis_csv, args.mias_info_csv, args.mias_images_dir,
        args.output_csv, args.outdir
    )
