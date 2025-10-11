#!/usr/bin/env python3
"""
Unified dataset prep with texture overlay enhancement.

This version:
 - Handles Roboflow YOLOv11 segmentation-style labels (polygons as many x y pairs),
 - Handles bbox lines (x_center y_center w h) and bbox+conf,
 - Trims a trailing token if a line has an odd number of coords (common if conf appended),
 - Ensures mask and image sizes match (resizes mask to image size if necessary),
 - Adds optional DEBUG mode (save overlays).
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import yaml
from typing import Tuple

from scipy.ndimage import gaussian_filter
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

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
    Base preprocessing: normalize → median blur → CLAHE → normalize.
    Returns uint8 grayscale image.
    """
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

# ---------------- Texture computations ---------------- #
def glcm_contrast_map_exact(img_uint8: np.ndarray, win: int = 31, levels: int = 8, step: int = 8) -> np.ndarray:
    H, W = img_uint8.shape
    half = win // 2
    if win < 3:
        win = 3; half = 1
    q = (img_uint8.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)
    heat = np.zeros((H, W), dtype=np.float32)
    ys = range(half, H - half, step)
    xs = range(half, W - half, step)
    for y in ys:
        for x in xs:
            patch = q[y - half : y + half + 1, x - half : x + half + 1]
            try:
                glcm = graycomatrix(patch, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
                contrast = graycoprops(glcm, "contrast")[0, 0]
            except Exception:
                contrast = 0.0
            heat[y, x] = float(contrast)
    heat = gaussian_filter(heat, sigma=max(1.0, step / 2.0))
    mn, mx = float(heat.min()), float(heat.max())
    if mx - mn < 1e-8:
        return np.zeros((H, W), dtype=np.float32)
    heat = (heat - mn) / (mx - mn + 1e-8)
    return heat.astype(np.float32)

def glcm_contrast_map_sobel(img_uint8: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    gx = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = gaussian_filter(mag, sigma=sigma)
    mn, mx = float(mag.min()), float(mag.max())
    if mx - mn < 1e-8:
        return np.zeros_like(mag, dtype=np.float32)
    heat = (mag - mn) / (mx - mn + 1e-8)
    return heat.astype(np.float32)

def glcm_contrast_map_local_var(img_uint8: np.ndarray, win: int = 31) -> np.ndarray:
    img_f = img_uint8.astype(np.float32)
    k = max(3, win // 2 * 2 + 1)
    mean = cv2.boxFilter(img_f, ddepth=-1, ksize=(k, k), normalize=True)
    mean_sq = cv2.boxFilter(img_f * img_f, ddepth=-1, ksize=(k, k), normalize=True)
    var = mean_sq - mean * mean
    var = np.clip(var, a_min=0.0, a_max=None)
    var = gaussian_filter(var, sigma=max(1.0, k / 6.0))
    mn, mx = float(var.min()), float(var.max())
    if mx - mn < 1e-8:
        return np.zeros_like(var, dtype=np.float32)
    heat = (var - mn) / (mx - mn + 1e-8)
    return heat.astype(np.float32)

def compute_texture_map(img_uint8: np.ndarray, method: str = "local_var",
                        win: int = 31, levels: int = 8, step: int = 8,
                        sobel_sigma: float = 3.0, local_var_win: int = 31) -> np.ndarray:
    if method == "local_var":
        return glcm_contrast_map_local_var(img_uint8, win=local_var_win)
    elif method == "sobel":
        return glcm_contrast_map_sobel(img_uint8, sigma=sobel_sigma)
    elif method == "glcm":
        return glcm_contrast_map_exact(img_uint8, win=win, levels=levels, step=step)
    else:
        raise ValueError(f"Unknown texture method: {method}")

# ---------------- Enhancement overlay ---------------- #
def enhance_with_texture(img_uint8: np.ndarray, texture: np.ndarray,
                         strength: float = 0.25, alpha: float = 0.6) -> np.ndarray:
    if img_uint8.dtype != np.uint8:
        img = img_uint8.astype(np.uint8)
    else:
        img = img_uint8
    h, w = img.shape
    if texture.shape != (h, w):
        texture = cv2.resize(texture, (w, h), interpolation=cv2.INTER_LINEAR)
    img_f = img.astype(np.float32) / 255.0
    overlay = img_f + (texture * strength)
    overlay = np.clip(overlay, 0.0, 1.0)
    out = (1.0 - alpha) * img_f + alpha * overlay
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

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

# ---------------- INBREAST processing (with debug overlays & size checks) ---------------- #
def process_inbreast_roboflow_v11(base_dir, yaml_path, mask_outdir, image_outdir, texture_cfg, debug=False):
    ensure_dir(mask_outdir); ensure_dir(image_outdir)
    if debug:
        debug_dir = os.path.join(image_outdir, "DEBUG_OVERLAYS")
        ensure_dir(debug_dir)
    class_map = load_class_map(yaml_path)
    rows = []
    splits = ["train", "valid", "test"]
    for split in splits:
        img_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")
        if not os.path.exists(img_dir):
            continue
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Cannot read {img_path}")
                continue
            img_proc = preprocess_image(img)
            mask, class_list, debug_polys = yolo_to_mask_with_classes(label_path, img_proc.shape, class_map, debug=debug)

            # if mask and img_proc shape mismatch (rare), resize mask to image
            if mask.shape != img_proc.shape:
                mask = cv2.resize(mask, (img_proc.shape[1], img_proc.shape[0]), interpolation=cv2.INTER_NEAREST)

            # texture enhance
            texture = compute_texture_map(img_proc, method=texture_cfg["method"],
                                          win=texture_cfg["win"], levels=texture_cfg["levels"],
                                          step=texture_cfg["step"], sobel_sigma=texture_cfg["sobel_sigma"],
                                          local_var_win=texture_cfg["local_var_win"])
            enhanced = enhance_with_texture(img_proc, texture,
                                            strength=texture_cfg["strength"], alpha=texture_cfg["alpha"])

            pid = f"{split}_{os.path.splitext(img_file)[0]}"
            proc_img_path = os.path.join(image_outdir, f"INBREAST_{pid}.png")
            mask_path = os.path.join(mask_outdir, f"INBREAST_{pid}_mask.png")

            cv2.imwrite(proc_img_path, enhanced)
            cv2.imwrite(mask_path, mask)

            # debug overlay if requested
            if debug:
                # color image for visualization
                vis = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR)
                # overlay mask edges
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0,0,255), 2)
                # optionally draw bbox/poly from debug_polys
                for poly, cls_name in debug_polys:
                    if isinstance(poly, tuple) and len(poly) == 4:
                        x1,y1,x2,y2 = poly
                        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 1)
                        cv2.putText(vis, cls_name, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                    else:
                        # poly is ndarray
                        try:
                            cv2.polylines(vis, [poly], isClosed=True, color=(0,255,0), thickness=1)
                        except Exception:
                            pass
                debug_path = os.path.join(debug_dir, f"DEBUG_{pid}.png")
                cv2.imwrite(debug_path, vis)

            row = {
                "dataset": "INBREAST",
                "patient_id": f"INBREAST_{pid}",
                "image_file_path": proc_img_path,
                "roi_mask_file_path": mask_path,
                "pathology": "Abnormality" if mask.sum() > 0 else "N",
                "abnormality_id": ";".join(class_list) if len(class_list) > 0 else "None",
            }
            rows.append(row)
    return pd.DataFrame(rows)

# ---------------- CBIS / MIAS / NORMAL (unchanged but ensure mask save shape check) ---------------- #
def process_cbis(input_csv, mask_outdir, image_outdir, texture_cfg):
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
        texture = compute_texture_map(processed_img, method=texture_cfg["method"],
                                      win=texture_cfg["win"], levels=texture_cfg["levels"],
                                      step=texture_cfg["step"], sobel_sigma=texture_cfg["sobel_sigma"],
                                      local_var_win=texture_cfg["local_var_win"])
        enhanced = enhance_with_texture(processed_img, texture,
                                        strength=texture_cfg["strength"], alpha=texture_cfg["alpha"])
        basename = os.path.splitext(os.path.basename(img_path))[0]
        abn_str = "-".join(abnormality_ids) if abnormality_ids else "NA"
        unique_name = f"CBIS_{pid}_{basename}_{abn_str}"
        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path = os.path.join(mask_outdir, f"{unique_name}_mask.png")
        cv2.imwrite(proc_img_path, enhanced)
        cv2.imwrite(mask_path, merged_mask)
        base_row["dataset"] = "CBIS"
        base_row["image_file_path"] = proc_img_path
        base_row["roi_mask_file_path"] = mask_path
        rows.append(base_row)
    return pd.DataFrame(rows)

def process_mias_from_csv(info_csv_path, images_dir, mask_outdir, image_outdir, texture_cfg):
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
        texture = compute_texture_map(img_proc, method=texture_cfg["method"],
                                      win=texture_cfg["win"], levels=texture_cfg["levels"],
                                      step=texture_cfg["step"], sobel_sigma=texture_cfg["sobel_sigma"],
                                      local_var_win=texture_cfg["local_var_win"])
        enhanced = enhance_with_texture(img_proc, texture,
                                        strength=texture_cfg["strength"], alpha=texture_cfg["alpha"])
        pid = f"MIAS_{refnum}"
        proc_img_path = os.path.join(image_outdir, f"{pid}.png")
        mask_path = os.path.join(mask_outdir, f"{pid}_mask.png")
        cv2.imwrite(proc_img_path, enhanced)
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

def process_normal_images(normal_img_dir, mask_outdir, image_outdir, texture_cfg):
    ensure_dir(mask_outdir); ensure_dir(image_outdir)
    rows = []
    normal_img_files = [f for f in os.listdir(normal_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in normal_img_files:
        img_path = os.path.join(normal_img_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_proc = preprocess_image(img)
        texture = compute_texture_map(img_proc, method=texture_cfg["method"],
                                      win=texture_cfg["win"], levels=texture_cfg["levels"],
                                      step=texture_cfg["step"], sobel_sigma=texture_cfg["sobel_sigma"],
                                      local_var_win=texture_cfg["local_var_win"])
        enhanced = enhance_with_texture(img_proc, texture,
                                        strength=texture_cfg["strength"], alpha=texture_cfg["alpha"])
        pid = os.path.splitext(img_file)[0]
        proc_img_path = os.path.join(image_outdir, f"NORMAL_{pid}.png")
        mask_path = os.path.join(mask_outdir, f"NORMAL_{pid}_mask.png")
        cv2.imwrite(proc_img_path, enhanced)
        mask = np.zeros_like(img_proc, dtype=np.uint8)
        cv2.imwrite(mask_path, mask)
        row = {
            "dataset": "NORMAL",
            "patient_id": f"NORMAL_{pid}",
            "image_file_path": proc_img_path,
            "roi_mask_file_path": mask_path,
            "pathology": "N",
            "abnormality_id": "None",
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------- UNIFIED ---------------- #
def prepare_unified(cbis_csv, mias_info_csv, mias_images_dir, normal_dir,
                    inbreast_base_dir, inbreast_yaml, output_csv, outdir,
                    texture_cfg, debug=False):
    cbis_img_dir = os.path.join(outdir, "CBIS_IMAGES"); cbis_mask_dir = os.path.join(outdir, "CBIS_MASKS")
    mias_img_dir = os.path.join(outdir, "MIAS_IMAGES"); mias_mask_dir = os.path.join(outdir, "MIAS_MASKS")
    normal_img_dir = os.path.join(outdir, "NORMAL_IMAGES"); normal_mask_dir = os.path.join(outdir, "NORMAL_MASKS")
    inbreast_img_outdir = os.path.join(outdir, "INBREAST_IMAGES"); inbreast_mask_outdir = os.path.join(outdir, "INBREAST_MASKS")

    ensure_dir(cbis_img_dir); ensure_dir(cbis_mask_dir)
    ensure_dir(mias_img_dir); ensure_dir(mias_mask_dir)
    ensure_dir(normal_img_dir); ensure_dir(normal_mask_dir)
    ensure_dir(inbreast_img_outdir); ensure_dir(inbreast_mask_outdir)

    cbis_df = process_cbis(cbis_csv, cbis_mask_dir, cbis_img_dir, texture_cfg)
    mias_df = process_mias_from_csv(mias_info_csv, mias_images_dir, mias_mask_dir, mias_img_dir, texture_cfg)
    normal_df = process_normal_images(normal_dir, normal_mask_dir, normal_img_dir, texture_cfg)
    inbreast_df = process_inbreast_roboflow_v11(inbreast_base_dir, inbreast_yaml, inbreast_mask_outdir, inbreast_img_outdir, texture_cfg, debug=debug)

    merged = pd.concat([cbis_df, mias_df, normal_df, inbreast_df], ignore_index=True)
    merged.to_csv(output_csv, index=False)
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS={len(cbis_df)}, MIAS={len(mias_df)}, NORMAL={len(normal_df)}, INBREAST={len(inbreast_df)})")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + MIAS + INBREAST (Roboflow YOLOv11) unified dataset")
    p.add_argument("--cbis-csv", type=str, required=True)
    p.add_argument("--mias-info-csv", type=str, required=True)
    p.add_argument("--mias-images-dir", type=str, required=True)
    p.add_argument("--normal-dir", type=str, required=True)
    p.add_argument("--inbreast-base-dir", type=str, required=True, help="Roboflow INBreast dataset root dir")
    p.add_argument("--inbreast-yaml", type=str, required=True, help="Roboflow data.yaml file")
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="DATASET")

    # Texture options
    p.add_argument("--texture-method", choices=["local_var", "sobel", "glcm"], default="local_var",
                   help="Texture map method: local_var (fast), sobel (fast), glcm (slow)")
    p.add_argument("--texture-strength", type=float, default=0.25,
                   help="How much texture is added to overlay (0..1). Smaller = subtler.")
    p.add_argument("--texture-alpha", type=float, default=0.6,
                   help="Blend factor between original and textured overlay (0..1).")
    p.add_argument("--texture-win", type=int, default=31, help="Window size for GLCM/local_var")
    p.add_argument("--texture-levels", type=int, default=8, help="Levels for exact glcm")
    p.add_argument("--texture-step", type=int, default=8, help="Step for exact glcm")
    p.add_argument("--sobel-sigma", type=float, default=3.0)
    p.add_argument("--local-var-win", type=int, default=31)

    p.add_argument("--debug", action="store_true", help="Save debug overlays for INBREAST label parsing (in INBREAST_IMAGES/DEBUG_OVERLAYS).")

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    texture_cfg = {
        "method": args.texture_method,
        "strength": float(args.texture_strength),
        "alpha": float(args.texture_alpha),
        "win": int(args.texture_win),
        "levels": int(args.texture_levels),
        "step": int(args.texture_step),
        "sobel_sigma": float(args.sobel_sigma),
        "local_var_win": int(args.local_var_win),
    }
    prepare_unified(
        args.cbis_csv, args.mias_info_csv, args.mias_images_dir,
        args.normal_dir, args.inbreast_base_dir, args.inbreast_yaml,
        args.output_csv, args.outdir, texture_cfg, debug=args.debug
    )
