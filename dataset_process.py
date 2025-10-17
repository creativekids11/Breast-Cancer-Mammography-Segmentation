#!/usr/bin/env python3
"""
Unified dataset prep for CBIS-DDSM and Mini-DDSM (incl. Data-MoreThanTwoMasks).

This version:
 - Only processes CBIS-DDSM and Mini-DDSM datasets
 - Simplified preprocessing: constant CLAHE (1.5) and median filtering
 - No texture map calculations
 - Supports a third mask column (Tumour_Contour3) and multiple mask files per row
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
from typing import Tuple, List
import shutil

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def safe_normpath_join(base: str, rel: str) -> str:
    """Join base and rel safely; if rel is absolute, return normpath(rel)."""
    if not rel or not isinstance(rel, str):
        return ""
    # normalize separators then join (handles backslashes inside rel)
    rel = rel.replace("\\", os.sep).replace("/", os.sep).strip()
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    return os.path.normpath(os.path.join(base, rel))

# ---------------- Preprocessing ---------------- #
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Simplified preprocessing: constant CLAHE (1.5) and median blur.
    Returns uint8 grayscale image.
    """
    # ensure grayscale uint8
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img.astype(np.uint8)

# ---------------- helper for merging masks ---------------- #
def read_mask_file(mask_fullpath: str, target_shape: Tuple[int, int]) -> np.ndarray:
    """Read a mask file, threshold to binary, and resize to target_shape if needed."""
    if not mask_fullpath or not os.path.exists(mask_fullpath):
        return np.zeros(target_shape, dtype=np.uint8)
    m = cv2.imread(mask_fullpath, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(target_shape, dtype=np.uint8)
    # binary threshold
    m = (m > 0).astype(np.uint8) * 255
    if m.shape != target_shape:
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return m

def collect_mask_paths_from_field(field_value: object) -> List[str]:
    """
    Accepts a cell value from the excel that may contain:
      - NaN or '-' -> return []
      - single relative path 'Benign\\0236\\C_0236_1.RIGHT_CC_Mask.png'
      - multiple paths separated by ';' or ',' or '|'
    Returns list of str (raw values, not joined with base).
    """
    if pd.isna(field_value):
        return []
    s = str(field_value).strip()
    if s == "-" or s == "":
        return []
    # split on common separators
    parts = []
    for sep in (";", ",", "|"):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            break
    if not parts:
        parts = [s]
    return parts

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

# ---------------- Mini-DDSM processing (updated for up to 3 masks) ---------------- #
def process_mini_ddsm(excel_path_list: List[str], base_dir: str, mask_outdir: str, image_outdir: str):
    """
    Process one or more Mini-DDSM Excel files (e.g. Data, Data-MoreThanTwoMasks).
    Accepts a list of excel paths so both the main Data sheet and the MoreThanTwoMasks
    workbook can be passed together.

    Args:
        excel_path_list: list of paths to XLSX files (each should have a sheet with rows)
        base_dir: Base directory containing the Mini-DDSM images
        mask_outdir: Output directory for processed masks
        image_outdir: Output directory for processed images
    """
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)
    
    all_rows = []
    rows_out = []
    
    # read and concat all provided excels (sheet named 'Data' or entire workbook first sheet)
    for excel_path in excel_path_list:
        if not excel_path:
            continue
        if not os.path.exists(excel_path):
            print(f"[WARNING] Mini-DDSM excel not found: {excel_path}, skipping")
            continue
        try:
            # prefer a sheet named 'Data' if present, otherwise read first sheet
            xl = pd.ExcelFile(excel_path)
            sheet_name = "Data" if "Data" in xl.sheet_names else xl.sheet_names[0]
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            all_rows.append(df)
        except Exception as e:
            print(f"[WARNING] Failed to read excel {excel_path}: {e}")
            continue
    
    if not all_rows:
        print("[WARNING] No Mini-DDSM excel data loaded.")
        return pd.DataFrame([])

    df_all = pd.concat(all_rows, ignore_index=True, sort=False)
    
    # iterate rows
    for idx, row in df_all.iterrows():
        img_rel_path = row.get("fullPath") or row.get("fileName")
        if not isinstance(img_rel_path, str) or img_rel_path.strip() == "":
            # try fallback to fileName
            img_rel_path = row.get("fileName", "")
        img_path = safe_normpath_join(base_dir, img_rel_path)
        
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
        
        # Load image (grayscale) and preprocess
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Failed to load image: {img_path}")
            continue
        processed_img = preprocess_image(img)
        
        # Prepare empty mask
        mask = np.zeros_like(processed_img, dtype=np.uint8)
        # Collect masks from up to 3 columns, plus allow multiple file entries per cell
        mask_fields = ["Tumour_Contour", "Tumour_Contour2", "Tumour_Contour3"]
        for mf in mask_fields:
            if mf in row.index:
                entries = collect_mask_paths_from_field(row.get(mf))
                for e in entries:
                    mask_full = safe_normpath_join(base_dir, e)
                    if mask_full and os.path.exists(mask_full):
                        mimg = read_mask_file(mask_full, processed_img.shape)
                        mask = cv2.bitwise_or(mask, mimg)
                    else:
                        # if file path wasn't absolute or not found, try raw filename only
                        # sometimes excel has only filename (C_0029_1.LEFT_CC_Mask.png)
                        fallback = safe_normpath_join(base_dir, os.path.basename(e)) if isinstance(e, str) else ""
                        if fallback and os.path.exists(fallback):
                            mimg = read_mask_file(fallback, processed_img.shape)
                            mask = cv2.bitwise_or(mask, mimg)
                        # else ignore silently
        
        # Create unique filename
        filename = row.get("fileName") or os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        status = row.get("Status", "")
        side = row.get("Side", "")
        view = row.get("View", "")
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
        rows_out.append(row_data)
    
    return pd.DataFrame(rows_out)

# ---------------- Main Processing Function ---------------- #
def process_datasets(cbis_csv, mini_ddsm_excel, mini_ddsm_extra_excel, mini_ddsm_base_dir, output_csv, outdir):
    """
    Process CBIS-DDSM and Mini-DDSM datasets including optional extra Mini-DDSM xlsx
    that contains a third mask column (Data-MoreThanTwoMasks).
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
    
    print("[INFO] Processing Mini-DDSM dataset(s)...")
    excel_list = [mini_ddsm_excel] if mini_ddsm_excel else []
    if mini_ddsm_extra_excel:
        excel_list.append(mini_ddsm_extra_excel)
    mini_df = process_mini_ddsm(excel_list, mini_ddsm_base_dir, mini_mask_dir, mini_img_dir)

    # Merge datasets
    merged = pd.concat([cbis_df, mini_df], ignore_index=True, sort=False)
    merged.to_csv(output_csv, index=False)
    
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS-DDSM={len(cbis_df)}, Mini-DDSM={len(mini_df)})")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + Mini-DDSM unified dataset (supports Data-MoreThanTwoMasks)")
    p.add_argument("--cbis-csv", type=str, required=True, 
                   help="Path to CBIS-DDSM CSV file")
    p.add_argument("--mini-ddsm-excel", type=str, required=True,
                   help="Path to Mini-DDSM DataWMask.xlsx file (primary)")
    p.add_argument("--mini-ddsm-extra-excel", type=str, required=False, default="",
                   help="Optional extra Mini-DDSM excel (e.g. Data-MoreThanTwoMasks XLSX) that contains Tumour_Contour3")
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
        args.mini_ddsm_extra_excel,
        args.mini_ddsm_base_dir,
        args.output_csv,
        args.outdir
    )
