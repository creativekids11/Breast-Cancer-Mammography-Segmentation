#!/usr/bin/env python3
"""
Unified dataset prep for CBIS-DDSM and Mini-DDSM.

This version:
 - Only processes CBIS-DDSM and Mini-DDSM datasets
 - Simplified preprocessing: applies constant CLAHE (1.5) and median filtering
 - No texture map calculations
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
from typing import Tuple

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ---------------- Preprocessing ---------------- #
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Simplified preprocessing: constant CLAHE (1.5) and median blur.
    Returns uint8 grayscale image.
    """
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img.astype(np.uint8)

# ---------------- CBIS-DDSM processing ---------------- #
def process_cbis(input_csv, mask_outdir, image_outdir):
    """Process CBIS-DDSM dataset with simplified preprocessing."""
    df = pd.read_csv(input_csv)
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)
    rows = []
    
    grouped = df.groupby(["patient_id", "image_file_path"], dropna=False)
    for (pid, img_path), group in grouped:
        base_row = group.iloc[0].to_dict()
        abnormality_ids = group["abnormality_id"].astype(str).unique().tolist()
        mask_paths = [mp for mp in group["roi_mask_file_path"].dropna().unique().tolist() if isinstance(mp, str)]
        
        # Merge masks if multiple
        merged_mask = None
        for mp in mask_paths:
            if not os.path.exists(mp):
                continue
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8) * 255
            merged_mask = mask if merged_mask is None else cv2.bitwise_or(merged_mask, mask)
        
        # Load and preprocess image
        if os.path.exists(img_path):
            full_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if full_img is None:
                continue
            processed_img = preprocess_image(full_img)
        else:
            continue
        
        if merged_mask is None:
            merged_mask = np.zeros_like(processed_img, dtype=np.uint8)
        
        # Ensure shapes match
        if merged_mask.shape != processed_img.shape:
            merged_mask = cv2.resize(merged_mask, (processed_img.shape[1], processed_img.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Save with unique name
        basename = os.path.splitext(os.path.basename(img_path))[0]
        abn_str = "-".join(abnormality_ids) if abnormality_ids else "NA"
        unique_name = f"CBIS_{pid}_{basename}_{abn_str}"
        
        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path = os.path.join(mask_outdir, f"{unique_name}_mask.png")
        
        cv2.imwrite(proc_img_path, processed_img)
        cv2.imwrite(mask_path, merged_mask)
        
        base_row["dataset"] = "CBIS-DDSM"
        base_row["image_file_path"] = proc_img_path
        base_row["roi_mask_file_path"] = mask_path
        rows.append(base_row)
    
    return pd.DataFrame(rows)

# ---------------- Mini-DDSM processing ---------------- #
def process_mini_ddsm(excel_path, base_dir, mask_outdir, image_outdir):
    """
    Process Mini-DDSM dataset with simplified preprocessing.
    
    Args:
        excel_path: Path to DataWMask.xlsx file
        base_dir: Base directory containing the Mini-DDSM images
        mask_outdir: Output directory for processed masks
        image_outdir: Output directory for processed images
    """
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)
    
    # Read the Data sheet from Excel
    df = pd.read_excel(excel_path, sheet_name="Data")
    rows = []
    
    for idx, row in df.iterrows():
        img_rel_path = row["fullPath"]
        img_path = os.path.join(base_dir, img_rel_path)
        
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
        
        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Failed to load image: {img_path}")
            continue
        
        processed_img = preprocess_image(img)
        
        # Load mask if available
        mask = np.zeros_like(processed_img, dtype=np.uint8)
        tumour_contour = row.get("Tumour_Contour", None)
        tumour_contour2 = row.get("Tumour_Contour2", None)
        
        # Check if mask paths are available (not NaN and not "-")
        if pd.notna(tumour_contour) and str(tumour_contour) != "-":
            mask_path = os.path.join(base_dir, tumour_contour)
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_img = (mask_img > 0).astype(np.uint8) * 255
                    if mask_img.shape != processed_img.shape:
                        mask_img = cv2.resize(mask_img, (processed_img.shape[1], processed_img.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, mask_img)
        
        # Check for second mask if available
        if pd.notna(tumour_contour2) and str(tumour_contour2) != "-":
            mask_path2 = os.path.join(base_dir, tumour_contour2)
            if os.path.exists(mask_path2):
                mask_img2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
                if mask_img2 is not None:
                    mask_img2 = (mask_img2 > 0).astype(np.uint8) * 255
                    if mask_img2.shape != processed_img.shape:
                        mask_img2 = cv2.resize(mask_img2, (processed_img.shape[1], processed_img.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, mask_img2)
        
        # Create unique filename
        filename = row["fileName"]
        basename = os.path.splitext(filename)[0]
        status = row["Status"]
        side = row["Side"]
        view = row["View"]
        unique_name = f"MINI_{status}_{basename}"
        
        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path_out = os.path.join(mask_outdir, f"{unique_name}_mask.png")
        
        cv2.imwrite(proc_img_path, processed_img)
        cv2.imwrite(mask_path_out, mask)
        
        # Create row data
        row_data = {
            "dataset": "Mini-DDSM",
            "patient_id": unique_name,
            "image_file_path": proc_img_path,
            "roi_mask_file_path": mask_path_out,
            "pathology": status,
            "abnormality_id": status,
            "side": side,
            "view": view,
            "age": row.get("Age", None),
            "density": row.get("Density", None),
        }
        rows.append(row_data)
    
    return pd.DataFrame(rows)

# ---------------- Main Processing Function ---------------- #
def process_datasets(cbis_csv, mini_ddsm_excel, mini_ddsm_base_dir, output_csv, outdir):
    """
    Process CBIS-DDSM and Mini-DDSM datasets.
    
    Args:
        cbis_csv: Path to CBIS-DDSM CSV file
        mini_ddsm_excel: Path to Mini-DDSM DataWMask.xlsx file
        mini_ddsm_base_dir: Base directory containing Mini-DDSM images
        output_csv: Output path for unified CSV
        outdir: Output directory for processed images and masks
    """
    cbis_img_dir = os.path.join(outdir, "CBIS_IMAGES")
    cbis_mask_dir = os.path.join(outdir, "CBIS_MASKS")
    mini_img_dir = os.path.join(outdir, "MINI_IMAGES")
    mini_mask_dir = os.path.join(outdir, "MINI_MASKS")

    ensure_dir(cbis_img_dir)
    ensure_dir(cbis_mask_dir)
    ensure_dir(mini_img_dir)
    ensure_dir(mini_mask_dir)

    print("[INFO] Processing CBIS-DDSM dataset...")
    cbis_df = process_cbis(cbis_csv, cbis_mask_dir, cbis_img_dir)
    
    print("[INFO] Processing Mini-DDSM dataset...")
    mini_df = process_mini_ddsm(mini_ddsm_excel, mini_ddsm_base_dir, mini_mask_dir, mini_img_dir)

    # Merge datasets
    merged = pd.concat([cbis_df, mini_df], ignore_index=True)
    merged.to_csv(output_csv, index=False)
    
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS-DDSM={len(cbis_df)}, Mini-DDSM={len(mini_df)})")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + Mini-DDSM unified dataset")
    p.add_argument("--cbis-csv", type=str, required=True, 
                   help="Path to CBIS-DDSM CSV file")
    p.add_argument("--mini-ddsm-excel", type=str, required=True,
                   help="Path to Mini-DDSM DataWMask.xlsx file")
    p.add_argument("--mini-ddsm-base-dir", type=str, required=True,
                   help="Base directory containing Mini-DDSM images")
    p.add_argument("--output-csv", type=str, required=True,
                   help="Output path for unified CSV")
    p.add_argument("--outdir", type=str, default="DATASET",
                   help="Output directory for processed images and masks")
    
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    process_datasets(
        args.cbis_csv,
        args.mini_ddsm_excel,
        args.mini_ddsm_base_dir,
        args.output_csv,
        args.outdir
    )
