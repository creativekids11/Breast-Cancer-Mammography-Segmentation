#!/usr/bin/env python3
"""
Unified dataset prep for CBIS-DDSM and Mini-DDSM (including Mini-DDSM2 Data-MoreThanTwoMasks).

This version:
 - Processes CBIS-DDSM and Mini-DDSM datasets
 - Applies constant CLAHE (1.5) and median filtering to images
 - For Mini-DDSM, merges Tumour_Contour, Tumour_Contour2 and any extra masks found
   in an optional `more_masks_dir` (Data-MoreThanTwoMasks from Mini-DDSM2).
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
from typing import Tuple, List
import glob

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def find_additional_masks_for_basename(more_masks_dir: str, basename: str) -> List[str]:
    """
    Recursively search more_masks_dir for files whose filename or stem contains `basename`
    (case-insensitive). Returns absolute paths.
    """
    if not more_masks_dir:
        return []
    matches = []
    basename_lower = basename.lower()
    for root, _, files in os.walk(more_masks_dir):
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                continue
            if basename_lower in f.lower():
                matches.append(os.path.join(root, f))
    return sorted(set(matches))

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

# ---------------- Mini-DDSM processing (extended for extra masks) ---------------- #
def _read_mask_if_exists(path: str, target_shape: Tuple[int,int]) -> np.ndarray:
    """
    Read mask at path as grayscale, threshold to binary 0/255, and resize to target_shape if needed.
    Returns binary mask or None on failure.
    """
    if not path or not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m_bin = (m > 0).astype(np.uint8) * 255
    if m_bin.shape != target_shape:
        m_bin = cv2.resize(m_bin, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return m_bin

def process_mini_ddsm(excel_path, base_dir, mask_outdir, image_outdir, more_masks_dir: str = ""):
    """
    Process Mini-DDSM dataset with simplified preprocessing.
    
    Args:
        excel_path: Path to DataWMask.xlsx file
        base_dir: Base directory containing the Mini-DDSM images
        mask_outdir: Output directory for processed masks
        image_outdir: Output directory for processed images
        more_masks_dir: Optional directory containing additional masks (Data-MoreThanTwoMasks)
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
        h, w = processed_img.shape[:2]
        
        # Start with an empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Primary masks referenced in the Excel
        tumour_contour = row.get("Tumour_Contour", None)
        tumour_contour2 = row.get("Tumour_Contour2", None)
        mask_sources = []
        
        # add tumour_contour if present
        if pd.notna(tumour_contour) and str(tumour_contour) not in ("-", "", "nan"):
            candidate = os.path.join(base_dir, str(tumour_contour))
            if os.path.exists(candidate):
                mask_sources.append(candidate)
        
        # add tumour_contour2 if present
        if pd.notna(tumour_contour2) and str(tumour_contour2) not in ("-", "", "nan"):
            candidate2 = os.path.join(base_dir, str(tumour_contour2))
            if os.path.exists(candidate2):
                mask_sources.append(candidate2)
        
        # Additionally look into more_masks_dir for any files that match the image basename
        if more_masks_dir:
            basename = os.path.splitext(os.path.basename(img_rel_path))[0]
            extra = find_additional_masks_for_basename(more_masks_dir, basename)
            if extra:
                mask_sources.extend(extra)
        
        # Deduplicate sources
        mask_sources = sorted(set(mask_sources))
        
        # Read and merge all mask sources
        for ms in mask_sources:
            m_bin = _read_mask_if_exists(ms, (h, w))
            if m_bin is None:
                continue
            mask = cv2.bitwise_or(mask, m_bin)
        
        # Save processed image and mask
        filename = row.get("fileName", os.path.basename(img_rel_path))
        basename = os.path.splitext(filename)[0]
        status = row.get("Status", "UNKNOWN")
        side = row.get("Side", None)
        view = row.get("View", None)
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
            "merged_mask_sources": ";".join(mask_sources) if mask_sources else ""
        }
        rows.append(row_data)
    
    return pd.DataFrame(rows)

# ---------------- Main Processing Function ---------------- #
def process_datasets(cbis_csv, mini_ddsm_excel, mini_ddsm_base_dir, output_csv, outdir, more_masks_dir: str = ""):
    """
    Process CBIS-DDSM and Mini-DDSM datasets.
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
    
    print("[INFO] Processing Mini-DDSM dataset (including extra masks if provided)...")
    mini_df = process_mini_ddsm(mini_ddsm_excel, mini_ddsm_base_dir, mini_mask_dir, mini_img_dir, more_masks_dir)

    # Merge datasets
    merged = pd.concat([cbis_df, mini_df], ignore_index=True, sort=False)
    merged.to_csv(output_csv, index=False)
    
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS-DDSM={len(cbis_df)}, Mini-DDSM={len(mini_df)})")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + Mini-DDSM (and Mini-DDSM2 extra masks) unified dataset")
    p.add_argument("--cbis-csv", type=str, required=True, 
                   help="Path to CBIS-DDSM CSV file")
    p.add_argument("--mini-ddsm-excel", type=str, required=True,
                   help="Path to Mini-DDSM DataWMask.xlsx file")
    p.add_argument("--mini-ddsm-base-dir", type=str, required=True,
                   help="Base directory containing Mini-DDSM images")
    p.add_argument("--mini-more-masks-dir", type=str, required=False, default="",
                   help="(Optional) Path to Mini-DDSM2 Data-MoreThanTwoMasks folder to merge extra masks")
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
        args.outdir,
        args.mini_more_masks_dir
    )
