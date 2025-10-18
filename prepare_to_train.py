#!/usr/bin/env python3
from __future__ import annotations
import argparse
import zipfile
import shutil
import subprocess
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import cv2
import re
import time
from typing import Optional, Dict, Tuple, List

# ---------------- Utilities ---------------- #
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_unzip(zip_path: Optional[Path], dest: Path):
    if not zip_path or not zip_path.exists():
        print(f"[WARN] Zip not found: {zip_path}")
        return False
    ensure_dir(dest)
    print(f"[INFO] Extracting {zip_path.name} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return True

def run_cmd(cmd: list, cwd: Optional[Path] = None, check: bool = True):
    print(f"[CMD] {' '.join(map(str, cmd))} (cwd={cwd or Path.cwd()})")
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(map(str, cmd))}")
    return res.returncode

def copy_tree_contents(src: Path, dst: Path):
    if not src.exists():
        print(f"[WARN] Source to copy does not exist: {src}")
        return
    ensure_dir(dst)
    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)

# ---------------- Auto-detect zips ---------------- #
ZIP_KEYWORDS = {
    "cbis": ["cbis", "cbis-ddsm", "cbis_ddsm", "cbisddsm", "ddsm"],
    "mini1": ["mini", "mini-ddsm", "mini-ddsm-complete", "mini-ddsm-complete-jpeg", "mini-ddsm-complete-jpeg-8"],
    "mini2": ["morethantwomasks", "more_than_two_masks", "more-than-two-masks", "data-morethan", "data-more-than-two-masks", "data-morethantwomasks"]
}

def detect_zips(zips_dir: Path):
    if not zips_dir.exists():
        return {"cbis": None, "mini1": None, "mini2": None}
    zips = [p for p in zips_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
    found = {"cbis": None, "mini1": None, "mini2": None}
    for key, kws in ZIP_KEYWORDS.items():
        for p in zips:
            nm = p.name.lower()
            if any(kw in nm for kw in kws):
                found[key] = p
                break
    # fallback heuristics
    if not found["cbis"]:
        for p in zips:
            if "cbis" in p.name.lower() or "ddsm" in p.name.lower():
                found["cbis"] = p
                break
    if not found["mini1"]:
        for p in zips:
            if "mini" in p.name.lower() and p != found["cbis"]:
                found["mini1"] = p
                break
    remaining = [p for p in zips if p not in (found["cbis"], found["mini1"], found["mini2"])]
    if not found["mini1"] and remaining:
        found["mini1"] = remaining[0]
    if not found["mini2"] and len(remaining) >= 2:
        found["mini2"] = remaining[1]
    return found

# ---------------- Path resolution helpers ---------------- #
_basename_search_cache: Dict[str, Optional[Path]] = {}

def find_file_by_basename(root: Path, basename: str, maxdirs: int = 2000) -> Optional[Path]:
    key = f"{str(root)}::{basename}"
    if key in _basename_search_cache:
        return _basename_search_cache[key]
    found = None
    checked_dirs = 0
    for dirpath, dirnames, filenames in os.walk(root):
        checked_dirs += 1
        if checked_dirs > maxdirs:
            break
        if basename in filenames:
            found = Path(dirpath) / basename
            break
    _basename_search_cache[key] = found
    return found

def normalize_mini_path_token(s: str) -> str:
    """
    Normalize MINI-style path tokens:
    - Convert sequences like '\0029' or '\000123' -> '0029' or '000123' (strip backslash)
    - Replace backslashes with os.sep
    - Collapse repeated separators
    Returns a normalized string suitable for Path joins.
    """
    if not isinstance(s, str):
        return s
    # Remove leading/trailing whitespace and quotes
    s = s.strip().strip('"').strip("'")
    # Remove backslash followed by digits sequences (e.g. '\0029' -> '0029')
    s2 = re.sub(r'\\0*([0-9]+)', r'\1', s)
    # Replace remaining backslashes with os.sep
    s2 = s2.replace('\\', os.sep).replace('/', os.sep)
    # Collapse duplicate separators
    sep = re.escape(os.sep)
    s2 = re.sub(f'{sep}+', os.sep.replace('\\', r'\\'), s2)
    return s2

def generate_candidate_paths(raw_ref: str, base_dir: Optional[str], csv_dir: Optional[str]) -> List[Path]:
    """
    Generate candidate filesystem paths to try for a raw reference string.
    - normalize tokens like '\0029' first
    - produce: absolute (if provided), csv_dir/raw, base_dir/raw, csv_dir/normalized, base_dir/normalized,
      normalized relative, just basename fallback etc.
    """
    candidates: List[Path] = []
    raw = (raw_ref or "").strip()
    if raw == "":
        return []
    # Normalize early (handles \0029 etc)
    norm_token = normalize_mini_path_token(raw)
    # Try raw provided as absolute (after converting backslashes)
    try_raw_abs = Path(raw.replace('\\', os.sep))
    if try_raw_abs.is_absolute():
        candidates.append(try_raw_abs)
    # csv_dir / base_dir with raw
    if csv_dir:
        candidates.append(Path(csv_dir) / raw)
    if base_dir:
        candidates.append(Path(base_dir) / raw)
    # csv_dir / base_dir with normalized token
    if csv_dir:
        candidates.append(Path(csv_dir) / norm_token)
    if base_dir:
        candidates.append(Path(base_dir) / norm_token)
    # normalized relative (current working dir)
    candidates.append(Path(norm_token))
    # bare basename fallback
    basename = Path(norm_token).name
    candidates.append(Path(basename))
    # also append original basename (in case normalization changed it)
    orig_basename = Path(raw).name
    if orig_basename != basename:
        candidates.append(Path(orig_basename))
    # Deduplicate preserving order
    seen = set()
    res: List[Path] = []
    for c in candidates:
        try:
            key = str(c)
        except Exception:
            key = repr(c)
        if key not in seen:
            seen.add(key)
            res.append(c)
    return res

# ---------------- Special heuristics for MoreThanTwoMasks layout ---------------- #
def resolve_mini2_base(workspace: Path) -> Path:
    """
    Return the correct base directory for the 'mini2' dataset inside the workspace.
    Tries multiple plausible folder names that people use in different extractions.
    """
    candidates = [
        workspace / "Data-MoreThanTwoMasks",
        workspace / "MoreThanTwoMasks",
        workspace / "Data-MoreThanTwoMasks-MINI",
        workspace / "MoreThanTwoMasks-MINI",
        workspace / "Data-MoreThanTwoMasks_Complete",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    # fallback to the first candidate even if missing (calling code should check exists)
    return candidates[0]

def try_resolve_mini_image_in_class_dirs(raw: str, base_dir: Path) -> Optional[Path]:
    """
    Given a raw token (already normalized), try to resolve to a file by inspecting
    the base_dir subfolders. Heuristics:
      - If token contains a class+id like 'Cancer48' -> split into 'Cancer' + '48' and look under base_dir/Cancer/0048
      - If filename contains digits (e.g. C_0048_1.*) extract digits and look under any subdir/*/<padded_id>
      - Fall back to searching each subdir/<padded_id> for the basename.
    """
    if not base_dir.exists():
        return None
    norm = normalize_mini_path_token(raw)
    parts = Path(norm).parts
    basename = Path(norm).name

    # 1) look for a part like Cancer48 or Normal13
    for part in parts:
        m = re.match(r'^([A-Za-z]+)(\d+)$', part)
        if m:
            class_name = m.group(1)
            num = m.group(2)
            try:
                num_i = int(num)
                padded = f"{num_i:04d}"
            except Exception:
                padded = num.zfill(4)
            # try candidate dir names (preserve original capitalization)
            possible_class_dirs = []
            possible_class_dirs.append(class_name)
            possible_class_dirs.append(class_name.capitalize())
            possible_class_dirs.append(class_name.upper())
            possible_class_dirs.append(class_name.lower())
            # try each class dir / padded / basename
            for class_dir in possible_class_dirs:
                cand_root = base_dir / class_dir / padded
                if cand_root.exists():
                    # try to find the exact basename under this folder (or subfolders)
                    found = find_file_by_basename(cand_root, basename, maxdirs=5000)
                    if found:
                        return found
                    # try simple file names inside folder (maybe images are stored differently)
                    possible_file = cand_root / basename
                    if possible_file.exists():
                        return possible_file

    # 2) Extract digits from basename and search all subdirs/<padded>
    nums = re.findall(r'(\d{1,6})', basename)
    if nums:
        num = nums[0]
        try:
            padded = f"{int(num):04d}"
        except Exception:
            padded = num.zfill(4)
        # iterate subdirs in base_dir
        for sub in base_dir.iterdir():
            if not sub.is_dir():
                continue
            candidate_dir = sub / padded
            if candidate_dir.exists():
                # search for file by basename
                found = find_file_by_basename(candidate_dir, basename, maxdirs=5000)
                if found:
                    return found
                # also try any file in candidate_dir with same prefix (e.g. prefix matching)
                for f in candidate_dir.iterdir():
                    if f.is_file() and basename.split('.')[0] in f.name:
                        return f

    # 3) Try a shallow basename search under base_dir (limit depth)
    found = find_file_by_basename(base_dir, basename, maxdirs=5000)
    if found:
        return found

    return None

# ---------------- New: rewrite CSV to point to files inside workspace ---------------- #
def make_paths_relative_in_csv(orig_csv_path: Path, cbis_workspace_dir: Path) -> Path:
    """
    Read a CBIS merge CSV and try to convert absolute or external paths to
    resolved paths inside cbis_workspace_dir. Writes a new CSV with suffix '_rel.csv'
    and returns its Path. If no changes were needed, returns original path.
    """
    if not orig_csv_path.exists():
        print(f"[WARN] make_paths_relative_in_csv: original CSV not found: {orig_csv_path}")
        return orig_csv_path
    df = pd.read_csv(orig_csv_path, dtype=str).fillna("")
    path_cols = []
    # heuristically find columns that look like path columns
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ("image", "path", "mask", "file")):
            path_cols.append(c)
    if not path_cols:
        path_cols = ["image_file_path", "roi_mask_file_path", "full_image_path", "cropped_image_file_path"]
    changed = False
    cbis_workspace_dir = Path(cbis_workspace_dir)
    csv_dir = orig_csv_path.parent
    for col in path_cols:
        if col not in df.columns:
            continue
        for idx, val in df[col].items():
            if not isinstance(val, str) or val.strip() == "":
                continue
            val_str = val.strip()
            p = Path(val_str.replace('\\', os.sep))
            if p.exists():
                # already valid on disk -> leave as-is
                continue
            candidates = generate_candidate_paths(val_str, base_dir=str(cbis_workspace_dir), csv_dir=str(csv_dir))
            resolved = None
            for c in candidates:
                try:
                    if c.exists():
                        resolved = c
                        break
                except Exception:
                    pass
            if resolved is None:
                basename = Path(val_str).name
                found = find_file_by_basename(cbis_workspace_dir, basename)
                if found:
                    resolved = found
            if resolved:
                df.at[idx, col] = str(resolved)
                changed = True
            else:
                # windows-drive strip fallback
                m = re.match(r'^[A-Za-z]:(\\|/)(.*)$', val_str)
                if m:
                    tail = m.group(2)
                    tail_norm = tail.replace('\\', os.sep).replace('/', os.sep)
                    cand = cbis_workspace_dir / tail_norm
                    if cand.exists():
                        df.at[idx, col] = str(Path(cand).resolve())
                        changed = True
                        continue
                # last-ditch basename search
                basename = Path(val_str).name
                found2 = find_file_by_basename(cbis_workspace_dir, basename)
                if found2:
                    df.at[idx, col] = str(found2)
                    changed = True
    if not changed:
        return orig_csv_path
    out = orig_csv_path.parent / (orig_csv_path.stem + "_rel" + orig_csv_path.suffix)
    df.to_csv(out, index=False)
    print(f"[INFO] Wrote adjusted CBIS CSV with workspace-resolved paths: {out}")
    return out

# ---------------- Write cleansing script (robust) ---------------- #
def write_cleansing_script(dest: Path):
    p = dest / "cbis_cleansing.py"
    content = r'''#!/usr/bin/env python3
"""
cbis_cleansing.py (robust) - placeholder
"""
print("Placeholder cleansing script (no-op).")
'''
    p.write_text(content)
    p.chmod(0o755)
    print(f"[INFO] Wrote cleansing script: {p}")
    return p

# ---------------- Write train script ---------------- #
def write_train_script(workspace: Path):
    sh_path = workspace / "train_cascade.sh"
    content = f"""#!/bin/bash
# Quick Start Script for Cascade Segmentation Training (relative paths)
echo "Running training script using relative paths"
python3 cascade_segmentation_model.py \\
    --train-both \\
    --tissue-data-dir segmentation_data/train_valid \\
    --cancer-csv unified_segmentation_dataset.csv \\
    --epochs-stage1 40 \\
    --epochs-stage2 150 \\
    --lr-stage1 5e-4 \\
    --lr-stage2 1e-3 \\
    --batch-size-stage1 32 \\
    --batch-size-stage2 32 \\
    --img-size-stage1 512 \\
    --img-size-stage2 512 \\
    --num-workers 4 \\
    --stage1-checkpoint-dir checkpoints_cascade/stage1 \\
    --stage2-checkpoint-dir checkpoints_cascade/stage2 \\
    --logdir runs/cascade_segmentation \\
    --l1-lambda 4.5e-5
"""
    sh_path.write_text(content)
    sh_path.chmod(0o755)
    print(f"[INFO] Wrote training script: {sh_path}")
    return sh_path

# ---------------- Integrated dataset processing (CBIS + Mini + Mini More) ---------------- #
def preprocess_image_simple(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img.astype(np.uint8)

_missing_cbis_paths: List[str] = []
_processed_counts = {"cbis": 0, "mini": 0, "mini2": 0}
_start_time = time.time()

def process_cbis(input_csv: str, mask_outdir: str, image_outdir: str, workspace: Path) -> pd.DataFrame:
    if not input_csv or not os.path.exists(input_csv):
        print(f"[INFO] process_cbis: input CSV missing or unreadable ({input_csv}). Skipping CBIS.")
        return pd.DataFrame([])
    df = pd.read_csv(input_csv, dtype=str).fillna("")
    ensure_dir(Path(mask_outdir)); ensure_dir(Path(image_outdir))
    rows = []
    if df.empty:
        print("[INFO] process_cbis: CSV empty, skipping.")
        return pd.DataFrame([])
    csv_dir = os.path.dirname(os.path.abspath(input_csv))
    grouped = df.groupby(["patient_id", "image_file_path"], dropna=False)
    for (pid, img_path_raw), group in grouped:
        base_row = group.iloc[0].to_dict()
        abnormality_ids = group["abnormality_id"].astype(str).unique().tolist()
        mask_raws = [mp for mp in group["roi_mask_file_path"].dropna().unique().tolist() if isinstance(mp, str) and mp != ""]
        mask_paths_resolved = []
        for raw in mask_raws:
            candidates = generate_candidate_paths(raw, base_dir=str(workspace / "CBIS-DDSM"), csv_dir=csv_dir)
            resolved = None
            for c in candidates:
                if c.exists():
                    resolved = c
                    break
            if resolved is None:
                candidate_basename = Path(raw).name
                found = find_file_by_basename(workspace / "CBIS-DDSM", candidate_basename)
                if found:
                    resolved = found
            if resolved:
                mask_paths_resolved.append(str(resolved))
        merged_mask = None
        for mp in mask_paths_resolved:
            try:
                mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            except Exception:
                mask = None
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8) * 255
            merged_mask = mask if merged_mask is None else cv2.bitwise_or(merged_mask, mask)
        # image resolution
        image_candidates = generate_candidate_paths(img_path_raw, base_dir=str(workspace / "CBIS-DDSM"), csv_dir=csv_dir)
        resolved_img = None
        for c in image_candidates:
            if c.exists():
                resolved_img = c
                break
        if resolved_img is None:
            basename = Path(img_path_raw).name
            found = find_file_by_basename(workspace / "CBIS-DDSM", basename)
            if found:
                resolved_img = found
        if resolved_img is None or not Path(resolved_img).exists():
            msg = f"{img_path_raw}"
            _missing_cbis_paths.append(msg)
            if len(_missing_cbis_paths) <= 50:
                print(f"[WARN] CBIS: image missing {img_path_raw}, skipping")
            continue
        try:
            full_img = cv2.imread(str(resolved_img), cv2.IMREAD_GRAYSCALE)
        except Exception:
            full_img = None
        if full_img is None:
            print(f"[WARN] CBIS: cannot read image {resolved_img}")
            continue
        processed_img = preprocess_image_simple(full_img)
        if merged_mask is None:
            merged_mask = np.zeros_like(processed_img, dtype=np.uint8)
        if merged_mask.shape != processed_img.shape:
            merged_mask = cv2.resize(merged_mask, (processed_img.shape[1], processed_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        basename = Path(resolved_img).stem
        abn_str = "-".join(abnormality_ids) if abnormality_ids else "NA"
        unique_name = f"CBIS_{pid}_{basename}_{abn_str}"
        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path_out = os.path.join(mask_outdir, f"{unique_name}_mask.png")
        cv2.imwrite(proc_img_path, processed_img)
        cv2.imwrite(mask_path_out, merged_mask)
        base_row["dataset"] = "CBIS-DDSM"
        base_row["image_file_path"] = proc_img_path
        base_row["roi_mask_file_path"] = mask_path_out
        rows.append(base_row)
        _processed_counts["cbis"] += 1
    print(f"[INFO] CBIS processed: {_processed_counts['cbis']}, missing references: {len(_missing_cbis_paths)}")
    return pd.DataFrame(rows)

def _normalize_mask_ref(mask_ref: str):
    if mask_ref is None:
        return None
    s = str(mask_ref).strip()
    if not s or s in ("-", "nan", "None", "NULL"):
        return None
    return s

def _split_mask_field(s: str):
    if s is None:
        return []
    parts = []
    for sep in [';', '|', ',']:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            break
    if not parts:
        parts = [p for p in s.split() if p]
    return parts

def _merge_mask_files(mask_paths, target_shape):
    h, w = target_shape
    out_mask = np.zeros((h, w), dtype=np.uint8)
    for mp in mask_paths:
        if not mp or not os.path.exists(mp):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m_bin = (m > 0).astype(np.uint8) * 255
        if m_bin.shape != (h, w):
            m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        out_mask = cv2.bitwise_or(out_mask, m_bin)
    return out_mask

def process_mini_ddsm(excel_path: str, base_dir: str, mask_outdir: str, image_outdir: str, workspace: Path, contour_columns=None) -> pd.DataFrame:
    if contour_columns is None:
        contour_columns = ["Tumour_Contour", "Tumour_Contour2"]
    ensure_dir(Path(mask_outdir)); ensure_dir(Path(image_outdir))
    if not excel_path or not os.path.exists(excel_path):
        print(f"[WARN] Mini sheet missing: {excel_path}. Skipping.")
        return pd.DataFrame([])
    csv_dir = os.path.dirname(os.path.abspath(excel_path))
    if str(excel_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(excel_path, sheet_name="Data")
    else:
        df = pd.read_csv(excel_path)
    rows = []

    # for mini2 special resolution, try to find actual base folder name
    base_dir_path = Path(base_dir)
    # prefer actual folder names that exist (support alternative names)
    if "MoreThanTwoMasks" in str(base_dir_path.name):
        resolved_mini2_base = resolve_mini2_base(workspace)
        if resolved_mini2_base.exists():
            base_dir_path = resolved_mini2_base

    for idx, r in df.iterrows():
        img_rel = r.get("fullPath") or r.get("fileName")
        if pd.isna(img_rel):
            continue
        img_rel = str(img_rel).strip()
        # normalize token first so backslash escapes become usable
        norm_img_rel = normalize_mini_path_token(img_rel)
        # generate candidates (handles 'Benign\0029\...' tokens)
        candidates = generate_candidate_paths(norm_img_rel, base_dir=str(base_dir_path), csv_dir=csv_dir)
        resolved_img = None
        for c in candidates:
            if c.exists():
                resolved_img = c
                break
        # if not found, try searching for basename under base_dir path (fast)
        if resolved_img is None:
            basename = Path(norm_img_rel).name
            found = find_file_by_basename(base_dir_path, basename)
            if found:
                resolved_img = found
        # if still None, try more advanced heuristic for MoreThanTwoMasks layout
        if resolved_img is None:
            # try splitting class+id and searching under base_dir/<Class>/<0001>/...
            found2 = try_resolve_mini_image_in_class_dirs(norm_img_rel, base_dir_path)
            if found2:
                resolved_img = found2
        # try join of base_dir + norm_img_rel
        if resolved_img is None:
            cand2 = Path(base_dir_path) / norm_img_rel
            if cand2.exists():
                resolved_img = cand2
        # as last attempt try csv_dir relative
        if resolved_img is None:
            cand3 = Path(csv_dir) / norm_img_rel
            if cand3.exists():
                resolved_img = cand3

        if resolved_img is None:
            print(f"[WARN] MINI: image not found (attempted normalized): {Path(base_dir_path) / norm_img_rel}")
            continue
        img = cv2.imread(str(resolved_img), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] MINI: failed load: {resolved_img}")
            continue
        processed_img = preprocess_image_simple(img)
        h, w = processed_img.shape[:2]
        mask_candidates = []
        for col in contour_columns:
            if col in r:
                raw = r.get(col)
                raw = _normalize_mask_ref(raw)
                if raw:
                    parts = _split_mask_field(str(raw))
                    mask_candidates.extend(parts)
        resolved_masks = []
        for cand in mask_candidates:
            cand_norm = normalize_mini_path_token(cand)
            mcands = generate_candidate_paths(cand_norm, base_dir=str(base_dir_path), csv_dir=csv_dir)
            resolved_mask = None
            for m in mcands:
                if m.exists():
                    resolved_mask = m
                    break
            if resolved_mask is None:
                possible = Path(resolved_img).parent / cand_norm
                if possible.exists():
                    resolved_mask = possible
            if resolved_mask is None:
                found = find_file_by_basename(base_dir_path, Path(cand_norm).name)
                if found:
                    resolved_mask = found
            if resolved_mask is not None:
                resolved_masks.append(str(resolved_mask))
        mask = _merge_mask_files(resolved_masks, (h, w))
        filename = r.get("fileName") or Path(resolved_img).name
        basename = Path(filename).stem
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
        _processed_counts["mini" if "MoreThanTwoMasks" not in str(excel_path) else "mini2"] += 1
    return pd.DataFrame(rows)

# ---------------- Orchestrator ---------------- #
def process_datasets_and_save(cbis_csv: Optional[str],
                              mini_ddsm_excel: str, mini_ddsm_base_dir: str,
                              mini2_excel: str, mini2_base_dir: str,
                              output_csv: str, outdir: str, workspace: Path):
    cbis_img_dir = os.path.join(outdir, "CBIS_IMAGES")
    cbis_mask_dir = os.path.join(outdir, "CBIS_MASKS")
    mini_img_dir = os.path.join(outdir, "MINI_IMAGES")
    mini_mask_dir = os.path.join(outdir, "MINI_MASKS")
    ensure_dir(Path(cbis_img_dir)); ensure_dir(Path(cbis_mask_dir))
    ensure_dir(Path(mini_img_dir)); ensure_dir(Path(mini_mask_dir))
    cbis_df = pd.DataFrame([])
    if cbis_csv and os.path.exists(cbis_csv):
        print("[INFO] Processing CBIS-DDSM dataset...")
        try:
            cbis_df = process_cbis(cbis_csv, cbis_mask_dir, cbis_img_dir, workspace)
        except Exception as e:
            print(f"[WARN] CBIS processing failed: {e}. Continuing with other datasets.")
            cbis_df = pd.DataFrame([])
    else:
        print("[INFO] No CBIS CSV provided or file missing; skipping CBIS processing.")
    print("[INFO] Processing Mini-DDSM (DataWMask.xlsx) dataset...")
    mini_df = process_mini_ddsm(mini_ddsm_excel, mini_ddsm_base_dir, mini_mask_dir, mini_img_dir, workspace,
                                contour_columns=["Tumour_Contour", "Tumour_Contour2"])
    mini2_df = pd.DataFrame([])
    if mini2_excel and os.path.exists(mini2_excel) and mini2_base_dir and os.path.exists(mini2_base_dir):
        # try to resolve actual base name (support alternative folder names)
        resolved_mini2_base = resolve_mini2_base(workspace)
        mini2_img_dir = os.path.join(outdir, "MINI2_IMAGES")
        mini2_mask_dir = os.path.join(outdir, "MINI2_MASKS")
        ensure_dir(Path(mini2_img_dir)); ensure_dir(Path(mini2_mask_dir))
        print("[INFO] Processing Mini-DDSM Data-MoreThanTwoMasks (supports 3+ masks)...")
        mini2_df = process_mini_ddsm(mini2_excel, str(resolved_mini2_base), mini2_mask_dir, mini2_img_dir, workspace,
                                     contour_columns=["Tumour_Contour", "Tumour_Contour2", "Tumour_Contour3"])
        if not mini2_df.empty:
            mini2_df["dataset"] = "Mini-DDSM-MoreThanTwoMasks"
    dfs = [df for df in (cbis_df, mini_df, mini2_df) if df is not None and not df.empty]
    merged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame([])
    ensure_dir(Path(os.path.dirname(os.path.abspath(output_csv)) or "."))
    merged.to_csv(output_csv, index=False)
    elapsed = time.time() - _start_time
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS-DDSM={len(cbis_df)}, Mini-DDSM={len(mini_df)}, Mini-DDSM-MoreThanTwoMasks={len(mini2_df)})")
    print(f"[INFO] Processed counts summary: {_processed_counts}, elapsed {elapsed:.1f}s")
    if _missing_cbis_paths:
        print("[INFO] Example missing CBIS image refs (first 40):")
        for p in _missing_cbis_paths[:40]:
            print("  ", p)
    return merged

# ---------------- Main flow ---------------- #
def main(args):
    workspace = (Path(args.workspace)).resolve()
    zips_dir = (Path(args.zips_dir)).resolve()
    ensure_dir(workspace)
    ensure_dir(zips_dir)
    print(f"[INFO] Workspace: {workspace}")
    print(f"[INFO] Zips dir: {zips_dir}")

    cbis_zip = Path(args.cbis_zip) if args.cbis_zip else None
    mini1_zip = Path(args.mini1_zip) if args.mini1_zip else None
    mini2_zip = Path(args.mini2_zip) if args.mini2_zip else None

    detected = detect_zips(zips_dir)
    if not cbis_zip or not cbis_zip.exists():
        cbis_zip = detected.get("cbis")
    if not mini1_zip or not mini1_zip.exists():
        mini1_zip = detected.get("mini1")
    if not mini2_zip or not mini2_zip.exists():
        mini2_zip = detected.get("mini2")

    print(f"[INFO] Detected zips -> CBIS: {cbis_zip}, MINI1: {mini1_zip}, MINI2: {mini2_zip}")

    ds_map = {
        "cbis": (cbis_zip, workspace / "CBIS-DDSM"),
        "mini1": (mini1_zip, workspace / "MINI-DDSM-Complete-JPEG-8"),
        "mini2": (mini2_zip, workspace / "Data-MoreThanTwoMasks"),
    }
    for key, (zp, dest) in ds_map.items():
        if zp:
            safe_unzip(zp, dest)
        else:
            print(f"[INFO] Zip for {key} not present; skipping unzip.")

    repo_dir = workspace / "repo"
    if not repo_dir.exists():
        try:
            print(f"[INFO] Cloning repo into {repo_dir}")
            run_cmd(["git", "clone", "https://github.com/creativekids11/Breast-Cancer-Mammography-Segmentation.git", str(repo_dir)])
        except Exception as e:
            print(f"[WARN] Git clone failed (continuing): {e}")
    else:
        print(f"[INFO] Repo already present: {repo_dir}")
    try:
        copy_tree_contents(repo_dir, workspace)
    except Exception as e:
        print(f"[WARN] copy_tree_contents failed (continuing): {e}")

    # copy merge_mass.csv into workspace/CBIS-DDSM if provided (try relative to zips-dir)
    if args.merge_csv:
        src_merge = Path(args.merge_csv)
        if not src_merge.exists():
            alt = zips_dir / args.merge_csv
            if alt.exists():
                src_merge = alt
        if src_merge.exists():
            dst = workspace / "CBIS-DDSM" / src_merge.name
            ensure_dir(dst.parent)
            shutil.copy2(src_merge, dst)
            print(f"[INFO] Copied merge CSV to {dst}")
            # rewrite CSV paths to be workspace-resolved
            adjusted = make_paths_relative_in_csv(dst, workspace / "CBIS-DDSM")
            cbis_merge_csv = adjusted
        else:
            print(f"[WARN] merge_mass.csv not found at {args.merge_csv} or {zips_dir}")
            cbis_merge_csv = workspace / "CBIS-DDSM" / (args.merge_csv or "merge_mass.csv")
    else:
        cbis_merge_csv = workspace / "CBIS-DDSM" / (args.merge_csv or "merge_mass.csv")

    cleansing_py = write_cleansing_script(workspace)
    if Path(cbis_merge_csv).exists():
        out_cleansed = workspace / "cbis_ddsm_cleansed.csv"
        print(f"[INFO] Running cleansing on {cbis_merge_csv} -> {out_cleansed}")
        try:
            run_cmd([sys.executable, str(cleansing_py), "--csv", str(cbis_merge_csv), "--out", str(out_cleansed)], cwd=workspace)
        except Exception as e:
            print(f"[WARN] Cleansing failed: {e}. Continuing.")
    else:
        print(f"[WARN] CBIS merge CSV not found at {cbis_merge_csv}. Skipping cleansing.")

    cbis_cleansed_path = workspace / "cbis_ddsm_cleansed.csv"
    if cbis_cleansed_path.exists():
        cbis_csv_to_use = str(cbis_cleansed_path)
        print(f"[INFO] Using cleansed CBIS CSV: {cbis_csv_to_use}")
    elif Path(cbis_merge_csv).exists():
        cbis_csv_to_use = str(cbis_merge_csv)
        print(f"[INFO] Cleansed file not available. Falling back to original: {cbis_csv_to_use}")
    else:
        cbis_csv_to_use = None
        print("[WARN] No CBIS CSV available (neither cleansed nor original). CBIS processing will be skipped.")

    mini_excel = workspace / "MINI-DDSM-Complete-JPEG-8" / "DataWMask.xlsx"
    mini_base = workspace / "MINI-DDSM-Complete-JPEG-8"
    # try alternative names for mini2 base
    mini2_base_candidate = workspace / "Data-MoreThanTwoMasks"
    mini2_alternative = resolve_mini2_base(workspace)
    if mini2_alternative.exists():
        mini2_base = mini2_alternative
    else:
        mini2_base = mini2_base_candidate
    
    # Define mini2_excel for the MoreThanTwoMasks dataset
    mini2_excel = mini2_base / "DataWMask.xlsx"

    output_csv = workspace / "unified_segmentation_dataset.csv"
    outdir = workspace / "data_files"

    print("[INFO] Running integrated dataset processing (CBIS optional).")
    try:
        process_datasets_and_save(cbis_csv_to_use, str(mini_excel), str(mini_base), str(mini2_excel), str(mini2_base), str(output_csv), str(outdir), workspace)
    except Exception as e:
        print(f"[ERROR] Dataset processing failed: {e}")

    write_train_script(workspace)

    archive_name = workspace.name + "-ready"
    shutil.make_archive(str(workspace / archive_name), 'zip', root_dir=str(workspace))
    print(f"[INFO] Created archive: {workspace / (archive_name + '.zip')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset workspace using relative paths and integrated processing")
    parser.add_argument("--workspace", default="./Breast-Cancer-AI", help="Workspace root (relative)")
    parser.add_argument("--zips-dir", default="./zips/", help="Directory containing dataset zip files")
    parser.add_argument("--cbis-zip", default="", help="Optional explicit CBIS zip filename (overrides auto-detection)")
    parser.add_argument("--mini1-zip", default="", help="Optional explicit MINI zip filename (overrides auto-detection)")
    parser.add_argument("--mini2-zip", default="", help="Optional explicit MINI_MORE zip filename (overrides auto-detection)")
    parser.add_argument("--merge-csv", default="merge_mass.csv", help="merge_mass.csv filename (relative or located in zips-dir)")
    args = parser.parse_args()

    if args.cbis_zip:
        args.cbis_zip = str(Path(args.zips_dir) / args.cbis_zip)
    else:
        args.cbis_zip = ""
    if args.mini1_zip:
        args.mini1_zip = str(Path(args.zips_dir) / args.mini1_zip)
    else:
        args.mini1_zip = ""
    if args.mini2_zip:
        args.mini2_zip = str(Path(args.zips_dir) / args.mini2_zip)
    else:
        args.mini2_zip = ""

    main(args)
