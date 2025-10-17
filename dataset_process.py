#!/usr/bin/env python3
"""
prepare_cbis_mini_unified.py

Unified dataset prep for CBIS-DDSM, Mini-DDSM (DataWMask.xlsx) and
Mini-DDSM Data-MoreThanTwoMasks (supports Tumour_Contour, Tumour_Contour2, Tumour_Contour3).

Behavior:
 - Simplified preprocessing: CLAHE(1.5) + median blur (grayscale).
 - Merges multiple masks per image into a single binary mask.
 - Supports MINI and MINI_MORE (Data-MoreThanTwoMasks) in different base dirs.
 - Produces processed images and masks into outdir subfolders and writes a unified CSV.

Usage example:
 python prepare_cbis_mini_unified.py \
   --cbis-csv /path/to/cbis.csv \
   --mini-ddsm-excel /path/to/DataWMask.xlsx \
   --mini-ddsm-base-dir /path/to/MINI_IMAGES_BASE \
   --mini2-excel /path/to/Data-MoreThanTwoMasks.xlsx \
   --mini2-base-dir /path/to/MINI2_BASE \
   --outdir OUTDIR \
   --output-csv unified.csv --debug
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import shutil
from typing import List, Optional
from pathlib import Path

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)

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

def _normalize_mask_ref(mask_ref: str) -> Optional[str]:
    """Return None for empty/placeholder tokens, else normalized path string."""
    if mask_ref is None:
        return None
    s = str(mask_ref).strip()
    if not s or s in ("-", "nan", "None", "NULL"):
        return None
    return s

def _split_mask_field(s: str) -> List[str]:
    """
    A mask field may contain multiple mask paths separated by
    ';', '|', ',', or whitespace. Return list of cleaned strings.
    """
    if s is None:
        return []
    parts = []
    for token in [p for sep in [';', '|', ','] for p in s.split(sep)]:
        token = token.strip()
        if token:
            parts.append(token)
    # If above split produced nothing but original has spaces, try whitespace split
    if not parts and isinstance(s, str):
        parts = [p for p in s.split() if p]
    return parts

def _resolve_mask_path(candidate: str, base_dir: str) -> Optional[str]:
    """
    Try to resolve a mask candidate path to an existing file.
    Handles backslashes, leading/trailing slashes, and absolute paths.
    """
    if candidate is None:
        return None
    cand = candidate.strip()
    if not cand or cand in ("-", "nan", "None", "NULL"):
        return None

    # If absolute path and exists
    if os.path.isabs(cand) and os.path.exists(cand):
        return cand

    # Try as relative to base_dir (preserve internal backslashes)
    joined = os.path.join(base_dir, cand)
    norm = os.path.normpath(joined)
    if os.path.exists(norm):
        return norm

    # Try replace backslashes with os.sep pieces
    cand2 = cand.replace("\\", os.sep).replace("/", os.sep)
    joined2 = os.path.join(base_dir, cand2)
    norm2 = os.path.normpath(joined2)
    if os.path.exists(norm2):
        return norm2

    # Try only the filename inside base_dir subtree (slow for large bases; we avoid scanning entire tree)
    # but attempt a sibling check: if candidate looks like "subdir/file.png", try last two components
    parts = cand.replace("\\", "/").split("/")
    if len(parts) >= 2:
        tail = os.path.join(*parts[-2:])
        cand3 = os.path.join(base_dir, tail)
        if os.path.exists(cand3):
            return os.path.normpath(cand3)

    # Not found
    return None

def _merge_mask_files(mask_paths: List[str], target_shape: tuple) -> np.ndarray:
    """Read and OR all existing masks in mask_paths; return binary (0/255) mask shaped target_shape."""
    h, w = target_shape
    out_mask = np.zeros((h, w), dtype=np.uint8)
    for mp in mask_paths:
        if not mp:
            continue
        if not os.path.exists(mp):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # Binarize then resize if needed
        m_bin = (m > 0).astype(np.uint8) * 255
        if m_bin.shape != (h, w):
            m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        out_mask = cv2.bitwise_or(out_mask, m_bin)
    return out_mask

# ---------------- CBIS-DDSM processing ---------------- #
def process_cbis(input_csv: str, mask_outdir: str, image_outdir: str) -> pd.DataFrame:
    """Process CBIS-DDSM dataset with simplified preprocessing."""
    df = pd.read_csv(input_csv)
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)
    rows = []

    # group by patient & image path (like your original)
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
                print(f"[WARN] CBIS: cannot read image {img_path}")
                continue
            processed_img = preprocess_image(full_img)
        else:
            print(f"[WARN] CBIS: image missing {img_path}, skipping")
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

# ---------------- Mini-DDSM processing (single-sheet) ---------------- #
def process_mini_ddsm(excel_path: str, base_dir: str, mask_outdir: str, image_outdir: str,
                      contour_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Process Mini-DDSM-like Excel sheet with mask columns.
    contour_columns: list of column names to look for masks (e.g. ["Tumour_Contour","Tumour_Contour2"])
    If contour_columns is None, defaults to ["Tumour_Contour","Tumour_Contour2"].
    """
    if contour_columns is None:
        contour_columns = ["Tumour_Contour", "Tumour_Contour2"]

    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)

    df = pd.read_excel(excel_path, sheet_name="Data") if excel_path.endswith((".xlsx", ".xls")) else pd.read_csv(excel_path)
    rows = []

    for idx, r in df.iterrows():
        img_rel = r.get("fullPath") or r.get("fileName")  # prefer fullPath then fileName
        if pd.isna(img_rel):
            continue
        img_rel = str(img_rel).strip()
        img_path = os.path.join(base_dir, img_rel) if not os.path.isabs(img_rel) else img_rel
        img_path = os.path.normpath(img_path)

        if not os.path.exists(img_path):
            print(f"[WARN] MINI: image not found: {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] MINI: failed load: {img_path}")
            continue

        processed_img = preprocess_image(img)
        h, w = processed_img.shape[:2]

        # collect mask candidate strings across provided contour columns
        mask_candidates = []
        for col in contour_columns:
            if col in r:
                raw = r.get(col)
                raw = _normalize_mask_ref(raw)
                if raw:
                    parts = _split_mask_field(str(raw))
                    mask_candidates.extend(parts)

        # resolve candidate paths into actual existing files
        resolved_masks = []
        for cand in mask_candidates:
            resolved = _resolve_mask_path(cand, base_dir)
            if resolved:
                resolved_masks.append(resolved)
            else:
                # try if cand is just filename in same folder as image
                possible = os.path.join(os.path.dirname(img_path), cand)
                if os.path.exists(possible):
                    resolved_masks.append(os.path.normpath(possible))
                else:
                    # not found: warn but continue
                    # print(f"[DEBUG] mask candidate not found: {cand} (base {base_dir})")
                    pass

        # Merge masks (if any)
        mask = _merge_mask_files(resolved_masks, (h, w))

        # Save processed image & mask
        filename = r.get("fileName") or os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        status = r.get("Status", "")
        side = r.get("Side", "")
        view = r.get("View", "")
        unique_name = f"MINI_{status}_{basename}"

        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_out_path = os.path.join(mask_outdir, f"{unique_name}_mask.png")

        cv2.imwrite(proc_img_path, processed_img)
        cv2.imwrite(mask_out_path, mask)

        row_data = {
            "dataset": "Mini-DDSM",
            "patient_id": unique_name,
            "image_file_path": proc_img_path,
            "roi_mask_file_path": mask_out_path,
            "pathology": status,
            "abnormality_id": status,
            "side": side,
            "view": view,
            "age": r.get("Age", None),
            "density": r.get("Density", None),
        }
        rows.append(row_data)

    return pd.DataFrame(rows)

# ---------------- Main Processing Function ---------------- #
def process_datasets(cbis_csv: str,
                     mini_ddsm_excel: str, mini_ddsm_base_dir: str,
                     mini2_excel: Optional[str], mini2_base_dir: Optional[str],
                     output_csv: str, outdir: str, debug: bool = False):
    """
    Process CBIS-DDSM, Mini-DDSM, and optionally Data-MoreThanTwoMasks (mini2).
    """
    cbis_img_dir = os.path.join(outdir, "CBIS_IMAGES")
    cbis_mask_dir = os.path.join(outdir, "CBIS_MASKS")
    mini_img_dir = os.path.join(outdir, "MINI_IMAGES")
    mini_mask_dir = os.path.join(outdir, "MINI_MASKS")

    ensure_dir(cbis_img_dir); ensure_dir(cbis_mask_dir)
    ensure_dir(mini_img_dir); ensure_dir(mini_mask_dir)

    print("[INFO] Processing CBIS-DDSM dataset...")
    cbis_df = process_cbis(cbis_csv, cbis_mask_dir, cbis_img_dir)

    print("[INFO] Processing Mini-DDSM (DataWMask.xlsx) dataset...")
    mini_df = process_mini_ddsm(mini_ddsm_excel, mini_ddsm_base_dir, mini_mask_dir, mini_img_dir,
                                contour_columns=["Tumour_Contour", "Tumour_Contour2"])

    mini2_df = pd.DataFrame([])
    if mini2_excel and mini2_base_dir:
        mini2_img_dir = os.path.join(outdir, "MINI2_IMAGES")
        mini2_mask_dir = os.path.join(outdir, "MINI2_MASKS")
        ensure_dir(mini2_img_dir); ensure_dir(mini2_mask_dir)
        print("[INFO] Processing Mini-DDSM Data-MoreThanTwoMasks (supports 3+ masks)...")
        # pass in third contour column as well
        mini2_df = process_mini_ddsm(mini2_excel, mini2_base_dir, mini2_mask_dir, mini2_img_dir,
                                     contour_columns=["Tumour_Contour", "Tumour_Contour2", "Tumour_Contour3"])
        # relabel dataset name to distinguish
        if not mini2_df.empty:
            mini2_df["dataset"] = "Mini-DDSM-MoreThanTwoMasks"

    # Concatenate all
    dfs = [df for df in (cbis_df, mini_df, mini2_df) if df is not None and not df.empty]
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
    else:
        merged = pd.DataFrame([])

    # Save CSV
    ensure_dir(os.path.dirname(os.path.abspath(output_csv)) or ".")
    merged.to_csv(output_csv, index=False)
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS-DDSM={len(cbis_df)}, Mini-DDSM={len(mini_df)}, Mini-DDSM-MoreThanTwoMasks={len(mini2_df)})")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + Mini-DDSM unified dataset (supports Data-MoreThanTwoMasks)")
    p.add_argument("--cbis-csv", type=str, required=True, help="Path to CBIS-DDSM CSV file")
    p.add_argument("--mini-ddsm-excel", type=str, required=True, help="Path to Mini-DDSM DataWMask.xlsx file")
    p.add_argument("--mini-ddsm-base-dir", type=str, required=True, help="Base directory containing Mini-DDSM images (MINI JPEGs)")
    p.add_argument("--mini2-excel", type=str, required=False, default="", help="Path to Data-MoreThanTwoMasks.xlsx (optional)")
    p.add_argument("--mini2-base-dir", type=str, required=False, default="", help="Base dir for Data-MoreThanTwoMasks images/masks (optional)")
    p.add_argument("--output-csv", type=str, required=True, help="Output path for unified CSV")
    p.add_argument("--outdir", type=str, default="DATASET", help="Output directory for processed images and masks")
    p.add_argument("--debug", action="store_true", help="Show debug prints")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    mini2_excel = args.mini2_excel if args.mini2_excel else None
    mini2_base = args.mini2_base_dir if args.mini2_base_dir else None

    process_datasets(
        cbis_csv=args.cbis_csv,
        mini_ddsm_excel=args.mini_ddsm_excel,
        mini_ddsm_base_dir=args.mini_ddsm_base_dir,
        mini2_excel=mini2_excel,
        mini2_base_dir=mini2_base,
        output_csv=args.output_csv,
        outdir=args.outdir,
        debug=args.debug
    )
