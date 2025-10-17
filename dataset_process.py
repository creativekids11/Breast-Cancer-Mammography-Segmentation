#!/usr/bin/env python3
"""
Unified dataset prep for CBIS-DDSM and Mini-DDSM (including Data-MoreThanTwoMasks).

This version:
 - Processes CBIS-DDSM and one or more Mini-DDSM Excel files
 - Simplified preprocessing: constant CLAHE (1.5) and median filtering
 - Mini-DDSM handler automatically picks up any Tumour_Contour* columns
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
from typing import List

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

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

# ---------------- Mini-DDSM processing (supports many mask columns) ---------------- #
def process_mini_ddsm(excel_path, base_dir, mask_outdir, image_outdir, sheet_name: str = None):
    """
    Process a Mini-DDSM Excel file. Automatically searches for any columns that start with
    'Tumour_Contour' (case-insensitive) and merges all masks referenced there.
    
    Args:
        excel_path: Path to the excel file (sheet 'Data' by default if present)
        base_dir: Base directory containing the Mini-DDSM images & masks
        mask_outdir: Output directory for processed masks
        image_outdir: Output directory for processed images
        sheet_name: Optional sheet name to use (defaults to 'Data' if present else first sheet)
    """
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)
    rows = []

    # pick sheet
    try:
        if sheet_name:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        else:
            # prefer sheet named Data, else first sheet
            xl = pd.ExcelFile(excel_path)
            sname = "Data" if "Data" in xl.sheet_names else xl.sheet_names[0]
            df = xl.parse(sname)
    except Exception as e:
        print(f"[ERROR] Failed to read excel {excel_path}: {e}")
        return pd.DataFrame(rows)

    # find tumour contour columns dynamically
    tumour_cols = [c for c in df.columns if str(c).lower().startswith("tumour_contour") or str(c).lower().startswith("tumor_contour")]
    if not tumour_cols:
        # fallback to the old names if present
        tumour_cols = [c for c in ["Tumour_Contour", "Tumour_Contour2", "Tumour_Contour3"] if c in df.columns]

    print(f"[INFO] Processing {excel_path} → found tumour columns: {tumour_cols}")

    for idx, row in df.iterrows():
        # Some excel entries put the relative path in 'fullPath' while others in 'fileName'
        img_rel = None
        if "fullPath" in row and pd.notna(row["fullPath"]):
            img_rel = str(row["fullPath"])
        elif "fileName" in row and pd.notna(row["fileName"]):
            img_rel = str(row["fileName"])
        else:
            # if neither present skip
            print(f"[WARN] No image path for row {idx} in {excel_path}")
            continue

        # normalize separators and join
        img_rel_norm = img_rel.replace("\\", os.sep).replace("/", os.sep)
        img_path = os.path.normpath(os.path.join(base_dir, img_rel_norm))

        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue

        # load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Failed to load image: {img_path}")
            continue
        processed_img = preprocess_image(img)

        # initialize mask
        mask = np.zeros_like(processed_img, dtype=np.uint8)

        # iterate tumour columns and merge masks
        for tc in tumour_cols:
            if tc not in row:
                continue
            val = row[tc]
            if pd.isna(val):
                continue
            s = str(val).strip()
            if s == "-" or len(s) == 0:
                continue
            # normalize and join
            s_norm = s.replace("\\", os.sep).replace("/", os.sep)
            mask_path = os.path.normpath(os.path.join(base_dir, s_norm))
            if not os.path.exists(mask_path):
                # try relative to same folder as image (some excel paths are relative differently)
                alt = os.path.normpath(os.path.join(os.path.dirname(img_path), os.path.basename(s_norm)))
                if os.path.exists(alt):
                    mask_path = alt
                else:
                    # warn and skip this mask entry
                    # not fatal; many rows have '-' or missing masks
                    # print(f"[DEBUG] Mask path not found: {mask_path}")
                    continue
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            mask_img = (mask_img > 0).astype(np.uint8) * 255
            # resize if necessary
            if mask_img.shape != processed_img.shape:
                mask_img = cv2.resize(mask_img, (processed_img.shape[1], processed_img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, mask_img)

        # create unique filename
        filename = row.get("fileName") or os.path.basename(img_path)
        basename = os.path.splitext(os.path.basename(filename))[0]
        status = row.get("Status", "")
        side = row.get("Side", "")
        view = row.get("View", "")
        unique_name = f"MINI_{status}_{basename}_{idx}"

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
def process_datasets(cbis_csv, mini_ddsm_excels: List[str], mini_ddsm_base_dir, output_csv, outdir):
    """
    Process CBIS-DDSM and one or more Mini-DDSM Excel files.

    Args:
        cbis_csv: Path to CBIS-DDSM CSV file
        mini_ddsm_excels: List of Mini-DDSM Excel files (can include Data-MoreThanTwoMasks.xlsx)
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

    mini_dfs = []
    for excel in mini_ddsm_excels:
        print(f"[INFO] Processing Mini-DDSM Excel: {excel}")
        df_mini = process_mini_ddsm(excel, mini_ddsm_base_dir, mini_mask_dir, mini_img_dir)
        mini_dfs.append(df_mini)

    if mini_dfs:
        mini_df = pd.concat(mini_dfs, ignore_index=True)
    else:
        mini_df = pd.DataFrame(columns=["dataset", "patient_id", "image_file_path", "roi_mask_file_path"])

    # Merge datasets
    merged = pd.concat([cbis_df, mini_df], ignore_index=True)
    ensure_dir(os.path.dirname(output_csv) or ".")
    merged.to_csv(output_csv, index=False)
    
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS-DDSM={len(cbis_df)}, Mini-DDSM={len(mini_df)})")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + Mini-DDSM unified dataset (supports >2 masks in excel)")
    p.add_argument("--cbis-csv", type=str, required=True, 
                   help="Path to CBIS-DDSM CSV file")
    p.add_argument("--mini-ddsm-excels", type=str, nargs="+", required=True,
                   help="One or more Mini-DDSM Excel files (e.g. DataWMask.xlsx Data-MoreThanTwoMasks.xlsx)")
    p.add_argument("--mini-ddsm-base-dir", type=str, required=True,
                   help="Base directory containing Mini-DDSM images and mask files")
    p.add_argument("--output-csv", type=str, required=True,
                   help="Output path for unified CSV")
    p.add_argument("--outdir", type=str, default="DATASET",
                   help="Output directory for processed images and masks")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    process_datasets(
        args.cbis_csv,
        args.mini_ddsm_excels,
        args.mini_ddsm_base_dir,
        args.output_csv,
        args.outdir
    )
