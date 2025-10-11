#!/usr/bin/env python3
"""
quick_scan.py

Lightweight iterator to scan a CSV of (image, mask) pairs, run a segmentation model,
save 'problem' cases, and optionally view dataset predictions with an interactive OpenCV viewer.

This version runs inference at two scales (default 512 and 1024) and merges predictions
so small tumours (better seen at 1024) and large ones (often captured at 512) are both preserved.
"""

import os
import argparse
import traceback
import numpy as np
import pandas as pd
import cv2
import torch

# Attempt to import helper model loaders from your segmentation_model module.
try:
    from segmentation_model import load_model_from_checkpoint, create_model, ACAAtrousResUNet as ACAAtrousUNet
    HAS_HELPERS = True
except Exception:
    HAS_HELPERS = False
    try:
        from segmentation_model import ACAAtrousUNet
    except Exception:
        ACAAtrousUNet = None

# ----------------- simple adaptive CLAHE (optional) ----------------- #
def extract_breast_mask(img_uint8):
    blur = cv2.GaussianBlur(img_uint8, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() < 128:
        th = 255 - th
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((th>0).astype('uint8'), connectivity=8)
    if labels is None or stats is None or stats.shape[0] <= 1:
        return np.ones_like(img_uint8, dtype=np.uint8) * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    mask = (labels == largest_idx).astype(np.uint8) * 255
    mask = cv2.blur(mask, (7,7))
    mask = (mask > 127).astype(np.uint8) * 255
    return mask

def adaptive_clahe(img_uint8, base_clip=2.0, min_clip=0.5, max_clip=6.0):
    img = img_uint8 if img_uint8.dtype == np.uint8 else img_uint8.astype(np.uint8)
    img_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    mask = extract_breast_mask(img_filtered)
    breast_pixels = img_filtered[mask>0] if mask.sum()>0 else img_filtered.flatten()
    std = float(np.std(breast_pixels)) if breast_pixels.size>0 else 0.0
    std_small, std_large = 6.0, 40.0
    clip = float(np.clip(np.interp(std, [std_small, std_large], [max_clip, min_clip]), min_clip, max_clip))
    clip = max(0.1, clip * (base_clip / 2.0))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    clahe_img = clahe.apply(img_filtered)
    out = img.copy()
    if mask.sum()>0:
        feather = cv2.GaussianBlur((mask>0).astype(np.float32), (31,31), 0)
        feather = np.clip(feather, 0.0, 1.0)
        out_f = (img.astype(np.float32) * (1.0 - feather) + clahe_img.astype(np.float32) * feather)
        out = np.clip(out_f, 0, 255).astype(np.uint8)
    else:
        out = clahe_img
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = cv2.medianBlur(out, 3)
    return out

# ----------------- utilities ----------------- #
def to_binary_mask(mask_uint8):
    if mask_uint8 is None:
        return None
    m = mask_uint8.copy()
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() > 1:
        m = (m > 0).astype(np.uint8)
    else:
        m = (m > 0.5).astype(np.uint8)
    return m

def save_overlay(out_dir, base_name, img, gt_mask, pred_mask):
    """
    Save three files: raw image, overlay GT (red), overlay pred (green).
    """
    os.makedirs(out_dir, exist_ok=True)
    img_8 = (np.clip(img, 0, 255)).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, base_name + "_img.png"), img_8)

    rgb = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)
    # GT contours
    if gt_mask is not None and gt_mask.sum() > 0:
        contours, _ = cv2.findContours((gt_mask>0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0,0,255), 1)  # red
    cv2.imwrite(os.path.join(out_dir, base_name + "_gt.png"), rgb)

    rgbp = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)
    if pred_mask is not None and pred_mask.sum() > 0:
        contours_p, _ = cv2.findContours((pred_mask>0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgbp, contours_p, -1, (0,255,0), 1)  # green
    cv2.imwrite(os.path.join(out_dir, base_name + "_pred.png"), rgbp)

# ----------------- model loader ----------------- #
def load_model(ckpt_path, img_size=1024, device='cpu', preferred="aca-atrous-unet"):
    device = torch.device(device)
    # Try helper loader first
    if HAS_HELPERS:
        try:
            mdl, info, chosen = load_model_from_checkpoint(ckpt_path, preferred_model_name=preferred, device=device, img_size=img_size)
            return mdl.eval(), f"Loaded via load_model_from_checkpoint (preferred={preferred})", chosen
        except Exception as e:
            print("[INFO] load_model_from_checkpoint failed:", e)
    # Try create_model helper
    if HAS_HELPERS:
        try:
            mdl = create_model(preferred, device, img_size)
            state = torch.load(ckpt_path, map_location=device)
            sd = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
            try:
                mdl.load_state_dict(sd, strict=False)
                return mdl.eval(), "Loaded via create_model + load_state_dict(strict=False)", preferred
            except Exception as e:
                print("[WARN] create_model load_state_dict failed:", e)
        except Exception as e:
            print("[INFO] create_model failed:", e)
    # Fallback to ACAAtrousUNet
    if ACAAtrousUNet is not None:
        try:
            mdl = ACAAtrousUNet(in_ch=1, out_ch=1)
            state = torch.load(ckpt_path, map_location='cpu')
            sd = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
            try:
                mdl.load_state_dict(sd, strict=False)
                mdl.to(device)
                return mdl.eval(), "Loaded ACAAtrousUNet with strict=False", "acaatrousunet"
            except Exception as e:
                print("[WARN] ACAAtrousUNet load_state_dict failed:", e)
        except Exception as e:
            print("[INFO] cannot instantiate ACAAtrousUNet:", e)
    raise RuntimeError("Failed to load model from checkpoint with available fallbacks.")

# ----------------- multiscale inference ----------------- #
def infer_multiscale(img_uint8, model, device='cpu', sizes=(512,1024), adaptive=False):
    """
    Run inference at multiple scales and combine probability maps.

    Args:
        img_uint8: single-channel uint8 (original image or a resized version)
        model: torch model (already on device)
        sizes: iterable of ints (image sizes to run inference on). Values are pixel square sizes.
        adaptive: whether to use adaptive_clahe preprocessing.

    Returns:
        combined_prob: numpy float32 array with shape (max_size, max_size), values in [0,1]
        final_proc_img: processed grayscale uint8 image at max_size (for saving/viewing)
    """
    device = torch.device(device)
    sizes_sorted = sorted(set(int(s) for s in sizes))
    max_size = sizes_sorted[-1]

    # We'll produce probability maps at max_size
    combined_prob = np.zeros((max_size, max_size), dtype=np.float32)
    proc_img_max = None

    for s in sizes_sorted:
        # resize source image to s
        img_resized = cv2.resize(img_uint8, (s, s), interpolation=cv2.INTER_LINEAR)
        if adaptive:
            img_proc = adaptive_clahe(img_resized)
        else:
            img_proc = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_proc = cv2.medianBlur(img_proc, 3)

        # save processed image for max size
        if s == max_size:
            proc_img_max = img_proc

        # prepare tensor
        inp = torch.tensor(img_proc / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            if isinstance(out, tuple):
                out = out[0]
            pred = torch.sigmoid(out).squeeze().cpu().numpy()
            # handle case (C,H,W)
            if pred.ndim == 3:
                pred = pred[0]
            # ensure predicted prob is s x s
            if pred.shape != (s, s):
                pred = cv2.resize(pred, (s, s), interpolation=cv2.INTER_LINEAR)

        # upsample to max_size
        if s != max_size:
            pred_up = cv2.resize(pred.astype(np.float32), (max_size, max_size), interpolation=cv2.INTER_LINEAR)
        else:
            pred_up = pred.astype(np.float32)

        # combine by pixel-wise maximum (keeps high-conf small detections from larger scale)
        combined_prob = np.maximum(combined_prob, pred_up)

    if proc_img_max is None:
        # produce a fallback proc image at max_size
        proc_img_max = cv2.resize(img_uint8, (max_size, max_size), interpolation=cv2.INTER_LINEAR)
        proc_img_max = cv2.normalize(proc_img_max, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        proc_img_max = cv2.medianBlur(proc_img_max, 3)

    return combined_prob, proc_img_max

# ----------------- main scan loop ----------------- #
def scan_dataset(csv_path, ckpt_path, outdir, sizes=(512,1024), adaptive=False, iou_thresh=0.05, device='cpu', limit=None):
    df = pd.read_csv(csv_path)
    sizes = tuple(int(s) for s in sizes)
    max_size = max(sizes)

    # load model with max size to initialize lazy modules (if any)
    model, info, chosen = load_model(ckpt_path, img_size=max_size, device=device)
    print("[MODEL INFO]", info, "chosen:", chosen)
    model.eval()

    problems = []
    os.makedirs(outdir, exist_ok=True)
    for idx, row in df.iterrows():
        if limit is not None and idx >= limit:
            break
        try:
            img_path = row['image_file_path']
            mask_path = row.get('roi_mask_file_path', None)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[SKIP] cannot read image: {img_path}")
                continue
            if mask_path and os.path.exists(mask_path):
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_bin = to_binary_mask(gt)
            else:
                gt_bin = np.zeros_like(img, dtype=np.uint8)

            # multi-scale inference -> combined probability map at max_size
            prob_map, proc_img = infer_multiscale(img, model, device=device, sizes=sizes, adaptive=adaptive)
            pred_bin = (prob_map > 0.5).astype(np.uint8)

            # resize gt to max_size for evaluation
            gt_resized = cv2.resize((gt_bin*255).astype(np.uint8), (max_size, max_size), interpolation=cv2.INTER_NEAREST)
            gt_resized = (gt_resized > 0).astype(np.uint8)

            # problem detection
            if gt_resized.sum() > 0 and pred_bin.sum() == 0:
                note = "missed_lesion"
                print(f"[PROBLEM] {idx} missed lesion -> saving")
                base = f"{idx:05d}_" + os.path.splitext(os.path.basename(img_path))[0]
                save_overlay(outdir, base, proc_img, gt_resized, pred_bin)
                problems.append({"index": idx, "image": img_path, "mask": mask_path, "note": note})
                continue

            inter = np.logical_and(gt_resized>0, pred_bin>0).sum()
            union = np.logical_or(gt_resized>0, pred_bin>0).sum()
            iou = float(inter) / (union + 1e-8)
            if union > 0 and iou < iou_thresh:
                note = f"low_iou_{iou:.4f}"
                print(f"[LOW IOU] {idx} iou={iou:.4f} -> saving")
                base = f"{idx:05d}_" + os.path.splitext(os.path.basename(img_path))[0]
                save_overlay(outdir, base, proc_img, gt_resized, pred_bin)
                problems.append({"index": idx, "image": img_path, "mask": mask_path, "note": note, "iou": iou})
                continue

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] row {idx} -> {e}\n{tb}")
            base = f"{idx:05d}_" + (os.path.splitext(os.path.basename(row.get('image_file_path','unknown')))[0])
            save_overlay(outdir, base, img if 'img' in locals() else np.zeros((max_size,max_size),dtype=np.uint8),
                         gt_bin if 'gt_bin' in locals() else np.zeros((max_size,max_size),dtype=np.uint8),
                         np.zeros((max_size,max_size),dtype=np.uint8))
            problems.append({"index": idx, "image": row.get('image_file_path'), "mask": row.get('roi_mask_file_path'), "note": "error", "error": str(e)})

    log_path = os.path.join(outdir, "problem_log.csv")
    pd.DataFrame(problems).to_csv(log_path, index=False)
    print(f"[DONE] scanned {len(df)} samples. Problems found: {len(problems)}. Log saved to {log_path}")

# Add near top of file (imports): try to import tkinter for screen size detection
try:
    import tkinter as tk
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False

def get_screen_size():
    """Return (width, height) of primary display. Fallback to 1280x720."""
    try:
        if _TK_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.destroy()
            return int(w), int(h)
    except Exception:
        pass
    # fallback
    return 1280, 720

def stack_display(img_gray, gt_mask, pred_mask_prob):
    """Create a side-by-side RGB visualization: original | GT overlay | Pred overlay + heatmap"""
    h, w = img_gray.shape[:2]
    base_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # GT overlay
    gt_rgb = base_rgb.copy()
    if gt_mask is not None and gt_mask.sum() > 0:
        contours, _ = cv2.findContours((gt_mask>0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_rgb, contours, -1, (0,0,255), 1)
    # Pred overlay + heatmap
    pred_rgb = base_rgb.copy()
    cmap = cv2.applyColorMap((np.clip(pred_mask_prob*255,0,255)).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.addWeighted(base_rgb, 0.5, cmap, 0.5, 0)
    pred_bin = (pred_mask_prob > 0.5).astype(np.uint8)
    if pred_bin.sum() > 0:
        contours_p, _ = cv2.findContours(pred_bin.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pred_rgb, contours_p, -1, (0,255,0), 1)
    top = np.hstack([base_rgb, gt_rgb, pred_rgb, heat])
    return top

def view_dataset(csv_path, ckpt_path, sizes=(512,1024), adaptive=False, device='cpu', only_problems=False, problems_log_dir="problem_cases"):
    df = pd.read_csv(csv_path)
    sizes = tuple(int(s) for s in sizes)
    max_size = max(sizes)

    # if only_problems, load problem_log.csv
    indices = list(range(len(df)))
    if only_problems:
        log_path = os.path.join(problems_log_dir, "problem_log.csv")
        if not os.path.exists(log_path):
            print(f"[WARN] problem_log.csv not found in {problems_log_dir} - falling back to full dataset")
        else:
            plog = pd.read_csv(log_path)
            if 'index' in plog.columns:
                indices = [int(x) for x in plog['index'].tolist() if int(x) < len(df)]
            else:
                imgs = plog['image'].tolist() if 'image' in plog.columns else []
                indices = []
                for i, r in df.iterrows():
                    if r['image_file_path'] in imgs:
                        indices.append(i)

    model, info, chosen = load_model(ckpt_path, img_size=max_size, device=device)
    print("[MODEL INFO]", info, "chosen:", chosen)
    model.eval()

    # screen/display sizing
    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * 0.90)
    max_h = int(screen_h * 0.90)

    i = 0
    total = len(indices)
    cur_idx = indices[i] if total>0 else None
    adaptive_on = adaptive
    outdir = problems_log_dir
    os.makedirs(outdir, exist_ok=True)

    print("[VIEWER] keys: n/Right=next, p/Left=prev, c=toggle adaptive CLAHE, s=save current as problem, q/Esc=quit")
    cv2.namedWindow("viewer", cv2.WINDOW_NORMAL)  # allow manual resizing if desired

    while True:
        if cur_idx is None:
            print("[VIEWER] no items to show.")
            break
        row = df.iloc[cur_idx]
        img_path = row['image_file_path']
        mask_path = row.get('roi_mask_file_path', None)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[VIEWER] cannot read {img_path}")
            i = (i + 1) % total
            cur_idx = indices[i]
            continue
        gt_bin = np.zeros_like(img, dtype=np.uint8)
        if mask_path and os.path.exists(mask_path):
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt_bin = to_binary_mask(gt)

        prob_map, proc_img = infer_multiscale(img, model, device=device, sizes=sizes, adaptive=adaptive_on)
        pred_bin = (prob_map > 0.5).astype(np.uint8)

        # gt resized to max_size for display/eval
        gt_resized = cv2.resize((gt_bin*255).astype(np.uint8), (max_size, max_size), interpolation=cv2.INTER_NEAREST)
        gt_resized = (gt_resized > 0).astype(np.uint8)

        display = stack_display(proc_img, gt_resized, prob_map)

        # scale display down if too large for screen while preserving aspect ratio
        disp_h, disp_w = display.shape[:2]
        scale = min(max_w / float(disp_w), max_h / float(disp_h), 1.0)
        if scale < 1.0:
            new_w = int(disp_w * scale)
            new_h = int(disp_h * scale)
            display_show = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)
            try:
                cv2.resizeWindow("viewer", new_w, new_h)
            except Exception:
                pass
        else:
            display_show = display
            try:
                cv2.resizeWindow("viewer", disp_w, disp_h)
            except Exception:
                pass

        # annotate with text: index, IoU etc.
        inter = np.logical_and(gt_resized>0, pred_bin>0).sum()
        union = np.logical_or(gt_resized>0, pred_bin>0).sum()
        iou = float(inter) / (union + 1e-8) if union > 0 else 0.0
        note = f"idx={cur_idx} ({i+1}/{total}) IoU={iou:.4f} adaptive={'ON' if adaptive_on else 'OFF'} sizes={sizes}"
        cv2.putText(display_show, note, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("viewer", display_show)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), 83):
            i = (i + 1) % total
            cur_idx = indices[i]
        elif key in (ord('p'), 81):
            i = (i - 1) % total
            cur_idx = indices[i]
        elif key == ord('c'):
            adaptive_on = not adaptive_on
        elif key == ord('s'):
            base = f"{cur_idx:05d}_" + os.path.splitext(os.path.basename(img_path))[0]
            save_overlay(outdir, base, proc_img, gt_resized, pred_bin)
            # append to problem log CSV (or create)
            log_path = os.path.join(outdir, "problem_log.csv")
            entry = {"index": int(cur_idx), "image": img_path, "mask": mask_path, "note": "manually_saved", "iou": iou}
            if os.path.exists(log_path):
                df_log = pd.read_csv(log_path)
                df_log = df_log._append(entry, ignore_index=True)
            else:
                df_log = pd.DataFrame([entry])
            df_log.to_csv(log_path, index=False)
            print(f"[SAVED] {base} saved to {outdir}")
        else:
            pass
    cv2.destroyAllWindows()

# ----------------- CLI ----------------- #
def get_args():
    p = argparse.ArgumentParser(description="Quick scan dataset for segmentation problems (multiscale inference)")
    p.add_argument("--csv", required=True, help="CSV with image_file_path and roi_mask_file_path columns")
    p.add_argument("--ckpt", required=True, help="Model checkpoint path")
    p.add_argument("--outdir", default="problem_cases", help="Where to save problem images/CSV")
    p.add_argument("--sizes", default="512,1024", help="Comma-separated inference sizes, e.g. '512,1024' (small,large)")
    p.add_argument("--adaptive", action="store_true", help="Use adaptive CLAHE preprocessing before inference")
    p.add_argument("--iou-thresh", type=float, default=0.05, help="IoU threshold for low-IoU reporting")
    p.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples (for quick tests)")
    p.add_argument("--view", action="store_true", help="Open interactive viewer instead of scanning everything")
    p.add_argument("--view-problems", action="store_true", help="When viewing, only iterate previously-saved problem cases (requires problem_log.csv)")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    sizes = tuple(int(s.strip()) for s in args.sizes.split(",") if s.strip())
    if args.view:
        view_dataset(args.csv, args.ckpt, sizes=sizes, adaptive=args.adaptive, device=args.device,
                     only_problems=args.view_problems, problems_log_dir=args.outdir)
    else:
        scan_dataset(args.csv, args.ckpt, args.outdir, sizes=sizes, adaptive=args.adaptive,
                     iou_thresh=args.iou_thresh, device=args.device, limit=args.limit)
