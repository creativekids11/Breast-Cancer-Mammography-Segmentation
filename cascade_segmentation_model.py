#!/usr/bin/env python3
"""
Cascade Staged Segmentation Model for Breast Cancer Detection

Stage 1: Tissue Segmentation (Adipose, Fibroglandular, Pectoral Muscle)
Stage 2: Cancer ROI Segmentation within Fibroglandular Region

Uses ACA-ResUNet architecture for both stages.
"""

import os
import argparse
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torchvision

# -------------------------
# Utility: find files by basename under search dirs
# -------------------------
def find_file_in_dirs(filename: str, search_dirs: List[str]) -> Optional[str]:
    """
    Search for `filename` (exact match) inside directories listed in `search_dirs`.
    Returns absolute path to first match or None if not found.
    """
    # If filename is already absolute and exists -> return it
    if os.path.isabs(filename) and os.path.exists(filename):
        return os.path.abspath(filename)

    # If filename is a relative path and exists -> return absolute path
    if os.path.exists(filename):
        return os.path.abspath(filename)

    # Search by basename match (case-sensitive). If input contains dirs, strip to basename.
    target_basename = os.path.basename(filename)

    for d in search_dirs:
        if not d:
            continue
        for root, _, files in os.walk(d):
            if target_basename in files:
                return os.path.abspath(os.path.join(root, target_basename))
    return None

def normalize_csv_paths(df: pd.DataFrame, image_col: str, mask_col: str, search_dirs: List[str]) -> pd.DataFrame:
    """
    For rows where image_col or mask_col paths are missing on disk, try to locate files
    by basename in `search_dirs` and replace with relative paths (relative to cwd).
    Returns a new DataFrame with updated paths.
    """
    df = df.copy()
    cwd = os.getcwd()
    updated_count = 0
    for idx, row in df.iterrows():
        # IMAGE
        img_path = str(row[image_col])
        if not img_path or not os.path.exists(img_path):
            found = find_file_in_dirs(img_path, search_dirs)
            if found:
                rel = os.path.relpath(found, start=cwd)
                df.at[idx, image_col] = rel
                updated_count += 1
            # else keep original (will be flagged later)

        # MASK
        mask_path = row.get(mask_col, "")
        if pd.isna(mask_path) or str(mask_path).strip() == "":
            # keep as empty string (no mask)
            df.at[idx, mask_col] = ""
        else:
            mask_path = str(mask_path)
            if not os.path.exists(mask_path):
                found_m = find_file_in_dirs(mask_path, search_dirs)
                if found_m:
                    relm = os.path.relpath(found_m, start=cwd)
                    df.at[idx, mask_col] = relm
                    updated_count += 1
                # else keep original (will be flagged later)
    print(f"[INFO] normalize_csv_paths: updated {updated_count} paths by searching {len(search_dirs)} directories.")
    return df

# ========================================
# Stage 1: Tissue Segmentation Dataset
# ========================================

class TissueSegmentationDataset(Dataset):
    """
    Dataset for Stage 1: Multi-class tissue segmentation
    Classes: 0=background, 1=adipose, 2=fibroglandular, 3=pectoral
    """
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 img_size: Tuple[int, int] = (512, 512), augment: bool = False):
        assert len(image_paths) == len(mask_paths), "image_paths and mask_paths must be same length"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        common_transforms = [
            A.Resize(self.img_size[0], self.img_size[1]),
        ]
        
        if self.augment:
            aug_transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
            ]
            transforms = common_transforms + aug_transforms + [ToTensorV2()]
        else:
            transforms = common_transforms + [ToTensorV2()]
        
        return A.Compose(transforms)
    
    def _convert_mask_to_classes(self, mask):
        """
        Convert intensity values to class indices:
        64 -> 1 (adipose)
        128/192 -> 2 (fibroglandular)
        255 -> 3 (pectoral muscle)
        0 or others -> 0 (background)
        """
        class_mask = np.zeros_like(mask, dtype=np.uint8)
        class_mask[mask == 64] = 1   # Adipose
        class_mask[(mask == 128) | (mask == 192)] = 2  # Fibroglandular (FGT)
        class_mask[mask == 255] = 3  # Pectoral muscle
        return class_mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            raise RuntimeError(f"Failed to load: {img_path} or {mask_path}")
        
        # Convert mask to class indices
        mask = self._convert_mask_to_classes(mask)
        
        # Apply transforms
        augmented = self.transform(image=img, mask=mask)
        
        # ToTensorV2 converts to tensor automatically
        img_t = augmented["image"]
        mask_t = augmented["mask"]
        
        # Ensure correct types
        if not isinstance(img_t, torch.Tensor):
            img_t = torch.from_numpy(img_t)
        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.from_numpy(mask_t)
        
        # Normalize image and ensure correct dtypes
        img_t = img_t.float() / 255.0
        mask_t = mask_t.long()
        
        return img_t, mask_t

# ========================================
# Stage 2: Cancer ROI Segmentation Dataset
# ========================================

class CancerROIDataset(Dataset):
    """
    Dataset for Stage 2: Binary cancer segmentation within FGT regions
    Accepts either a CSV file path or a prepared DataFrame with columns:
      - image_file_path
      - roi_mask_file_path  (can be empty)
    """
    def __init__(self, csv_file: Optional[str] = None, img_size: Tuple[int, int] = (384, 384), 
                 augment: bool = False, dataframe: Optional[pd.DataFrame] = None, search_dirs: Optional[List[str]] = None):
        if dataframe is not None:
            df = dataframe.copy()
        else:
            if csv_file is None:
                raise ValueError("Either csv_file or dataframe must be provided to CancerROIDataset")
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
            df = pd.read_csv(csv_file)
        
        # For safety: ensure columns exist
        if "image_file_path" not in df.columns or "roi_mask_file_path" not in df.columns:
            raise ValueError("CSV/DataFrame must contain 'image_file_path' and 'roi_mask_file_path' columns")
        
        # If user passed search_dirs, attempt to normalize missing paths before filtering
        if search_dirs:
            df = normalize_csv_paths(df, "image_file_path", "roi_mask_file_path", search_dirs)
        
        # Filter out rows with missing or unreadable files
        valid_image_paths = []
        valid_mask_paths = []
        for idx, row in df.iterrows():
            img_path = str(row["image_file_path"])
            mask_path = row["roi_mask_file_path"]
            mask_path = str(mask_path) if pd.notna(mask_path) and str(mask_path).strip() != "" else ""
            if os.path.exists(img_path):
                if mask_path == "" or os.path.exists(mask_path):
                    valid_image_paths.append(img_path)
                    valid_mask_paths.append(mask_path)
                else:
                    print(f"[WARNING] Skipping row {idx}: mask not found -> {mask_path}")
            else:
                print(f"[WARNING] Skipping row {idx}: image not found -> {img_path}")
        
        self.image_paths = valid_image_paths
        self.mask_paths = valid_mask_paths
        self.img_size = img_size
        self.augment = augment
        self.transform = self._get_transforms()

    def _get_transforms(self):
        common_transforms = [
            A.Resize(self.img_size[0], self.img_size[1]),
        ]
        
        if self.augment:
            transforms = common_transforms + [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                ToTensorV2(),
            ]
        else:
            transforms = common_transforms + [ToTensorV2()]
        
        return A.Compose(transforms)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = None
        mask_path = self.mask_paths[idx] if idx < len(self.mask_paths) else ""
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise RuntimeError(f"Failed to load: {self.image_paths[idx]}")
        
        if mask is None:
            mask = np.zeros_like(img, dtype=np.uint8)
        
        # Binary mask (0 or 255)
        mask = ((mask > 0).astype(np.uint8) * 255)
        
        augmented = self.transform(image=img, mask=mask)
        
        # ToTensorV2 converts to tensor automatically
        img_t = augmented["image"]
        mask_t = augmented["mask"]
        
        # Ensure correct types
        if not isinstance(img_t, torch.Tensor):
            img_t = torch.from_numpy(img_t)
        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.from_numpy(mask_t)

        # Ensure channel dimension is present for image and mask
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        if img_t.ndim == 3 and img_t.shape[0] > 1:
            img_t = img_t[:1]

        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        if mask_t.ndim == 3 and mask_t.shape[0] > 1:
            mask_t = mask_t[:1]

        # Normalize to [0,1]
        img_t = img_t.float() / 255.0
        mask_t = mask_t.float() / 255.0
        
        return img_t, mask_t

# ========================================
# ACA-ResUNet Architecture Components
# ========================================

class ACAModule(nn.Module):
    """Adaptive Context Aggregation Module"""
    def __init__(self, skip_channels, gate_channels, reduction=8):
        super().__init__()
        reduced_ch = max(skip_channels // reduction, 1)
        
        # Channel Attention
        self.ca = nn.Sequential(
            nn.Conv2d(skip_channels + gate_channels, reduced_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, skip_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(skip_channels + gate_channels, reduced_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, skip, gate):
        concat = torch.cat([skip, gate], dim=1)
        ca_weight = self.ca(concat)
        sa_weight = self.sa(concat)
        refined = skip * ca_weight * sa_weight + skip
        return self.fuse(refined)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False)
            for r in rates
        ])
        self.bn = nn.BatchNorm2d(out_ch * len(rates))
        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        feats = [blk(x) for blk in self.blocks]
        x = torch.cat(feats, dim=1)
        x = self.relu(self.bn(x))
        x = self.project(x)
        return x

class UpACA(nn.Module):
    """Upsampling block with ACA module"""
    def __init__(self, out_ch=64, dropout=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self._out_ch = out_ch
        self._dropout = dropout
        self.aca = None
        self.conv = None
    
    def _create_modules(self, skip_ch, gate_ch, device=None, dtype=None):
        self.aca = ACAModule(skip_channels=skip_ch, gate_channels=gate_ch)
        
        layers = [
            nn.Conv2d(skip_ch + gate_ch, self._out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self._out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self._out_ch, self._out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self._out_ch),
            nn.ReLU(inplace=True)
        ]
        if self._dropout:
            layers.append(nn.Dropout2d(0.1))
        self.conv = nn.Sequential(*layers)
        
        if device is not None:
            self.aca = self.aca.to(device=device, dtype=dtype)
            self.conv = self.conv.to(device=device, dtype=dtype)
    
    def forward(self, x_decoder, x_encoder):
        x = self.up(x_decoder)
        if x.shape[2:] != x_encoder.shape[2:]:
            x = F.interpolate(x, size=x_encoder.shape[2:], mode='bilinear', align_corners=False)
        
        if self.aca is None or self.conv is None:
            skip_ch = x_encoder.shape[1]
            gate_ch = x.shape[1]
            self._create_modules(skip_ch, gate_ch, device=x_encoder.device, dtype=x_encoder.dtype)
        
        skip_ref = self.aca(x_encoder, x)
        out = torch.cat([skip_ref, x], dim=1)
        return self.conv(out)

# ========================================
# ACA-ResUNet Models
# ========================================

class ACAAtrousResUNet(nn.Module):
    """ACA-ResUNet for multi-class or binary segmentation"""
    def __init__(self, in_ch=1, out_ch=4, encoder_name="resnet34"):
        super().__init__()
        # Use SMP Unet as encoder body, then replace decoder with ACA blocks
        self.encoder = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights="imagenet" if in_ch == 3 else None,
            in_channels=in_ch, 
            classes=out_ch
        )
        encoder_channels = self.encoder.encoder.out_channels
        
        # ASPP bottleneck (use last encoder channel -> produce feature map with earlier out_ch size)
        self.aspp = ASPP(in_ch=encoder_channels[-1], out_ch=encoder_channels[-2])
        
        # Decoder with ACA
        # create UpACA with appropriate out_ch sizes
        self.up_aca1 = UpACA(out_ch=encoder_channels[-3])
        self.up_aca2 = UpACA(out_ch=encoder_channels[-4])
        self.up_aca3 = UpACA(out_ch=encoder_channels[-5])
        self.up_aca4 = UpACA(out_ch=encoder_channels[-5])
        
        self.outc = nn.Conv2d(encoder_channels[-5], out_ch, kernel_size=1)
    
    def forward(self, x):
        # Encode
        feats = self.encoder.encoder(x)
        # feats is a list of encoder feature maps
        if len(feats) >= 6:
            e1, e2, e3, e4, bottleneck = feats[1], feats[2], feats[3], feats[4], feats[5]
        else:
            e1, e2, e3, e4, bottleneck = feats[-5], feats[-4], feats[-3], feats[-2], feats[-1]
        
        # ASPP
        d5 = self.aspp(bottleneck)
        
        # Decode with ACA
        d4 = self.up_aca1(d5, e4)
        d3 = self.up_aca2(d4, e3)
        d2 = self.up_aca3(d3, e2)
        d1 = self.up_aca4(d2, e1)
        
        logits = self.outc(d1)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

# ========================================
# Cascade Model
# ========================================

class CascadeSegmentationModel(nn.Module):
    """
    Two-stage cascade model:
    Stage 1: Tissue segmentation (4 classes)
    Stage 2: Cancer segmentation (binary) within FGT region
    """
    def __init__(self, stage1_weights=None, stage2_weights=None, device='cuda'):
        super().__init__()
        self.device = device
        
        # Stage 1: Multi-class tissue segmentation (4 classes)
        self.stage1_model = ACAAtrousResUNet(in_ch=1, out_ch=4, encoder_name="resnet34").to(device)
        if stage1_weights and os.path.exists(stage1_weights):
            self.stage1_model.load_state_dict(torch.load(stage1_weights, map_location=device))
            print(f"Loaded Stage 1 weights from {stage1_weights}")
        
        # Stage 2: Binary cancer segmentation
        self.stage2_model = ACAAtrousResUNet(in_ch=1, out_ch=1, encoder_name="resnet34").to(device)
        if stage2_weights and os.path.exists(stage2_weights):
            self.stage2_model.load_state_dict(torch.load(stage2_weights, map_location=device))
            print(f"Loaded Stage 2 weights from {stage2_weights}")
    
    def extract_fgt_roi(self, image: torch.Tensor, tissue_mask: torch.Tensor, padding=20, target_size=(384,384)):
        """
        Extract FGT (fibroglandular tissue) region from image
        tissue_mask: predicted tissue segmentation (argmax of stage 1) shape [B,1,H,W]
        image: original image tensor [B, C, H, W]
        Returns cropped batch tensor sized to target_size
        """
        # Convert tissue mask to CPU numpy for bbox extraction
        fgt_mask = (tissue_mask == 2).cpu().numpy().astype(np.uint8)
        
        batch_crops = []
        B = fgt_mask.shape[0]
        for b in range(B):
            if fgt_mask.ndim == 4:
                fgt_b = fgt_mask[b, 0]
            elif fgt_mask.ndim == 3:
                fgt_b = fgt_mask[b]
            else:
                raise RuntimeError("Unexpected fgt_mask shape")
            
            rows = np.any(fgt_b, axis=1)
            cols = np.any(fgt_b, axis=0)
            
            if rows.sum() == 0 or cols.sum() == 0:
                crop = image[b]  # full image
            else:
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                h, w = fgt_b.shape
                rmin = max(0, rmin - padding)
                rmax = min(h - 1, rmax + padding)
                cmin = max(0, cmin - padding)
                cmax = min(w - 1, cmax + padding)
                
                crop = image[b, :, rmin:(rmax + 1), cmin:(cmax + 1)]
            
            if not torch.is_floating_point(crop):
                crop = crop.float()
            crop_resized = F.interpolate(
                crop.unsqueeze(0), 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            batch_crops.append(crop_resized)
        
        if len(batch_crops) == 0:
            resized = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
            return resized
        return torch.cat(batch_crops, dim=0).to(image.device)
    
    def forward(self, x, return_stage1=False):
        """
        Forward pass through cascade
        x: input image [B, 1, H, W]
        """
        # Stage 1: Tissue segmentation
        tissue_logits = self.stage1_model(x)
        tissue_pred = torch.argmax(tissue_logits, dim=1, keepdim=True)
        
        # Extract FGT ROI
        fgt_crops = self.extract_fgt_roi(x, tissue_pred)
        
        # Stage 2: Cancer segmentation on FGT region
        cancer_logits = self.stage2_model(fgt_crops)
        
        if return_stage1:
            return tissue_logits, cancer_logits, tissue_pred
        return cancer_logits

# ========================================
# Loss Functions
# ========================================

class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for binary segmentation"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        
        return bce + (1 - dice)

class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # inputs: [B, C, H, W]
        # targets: [B, H, W]
        num_classes = inputs.shape[1]
        dice_scores = []
        
        inputs_soft = F.softmax(inputs, dim=1)
        
        for c in range(num_classes):
            input_c = inputs_soft[:, c].contiguous().view(-1)
            target_c = (targets == c).float().contiguous().view(-1)
            
            intersection = (input_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (input_c.sum() + target_c.sum() + self.smooth)
            dice_scores.append(dice)
        
        return 1 - torch.mean(torch.stack(dice_scores))

# ========================================
# Training Functions
# ========================================

def l1_regularization(model, lambda_l1=1e-5):
    """Compute L1 regularization loss"""
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.norm(param, 1)
    return lambda_l1 * l1_loss

def train_stage1(model, train_loader, val_loader, args):
    """Train Stage 1: Tissue Segmentation"""
    print("\n" + "="*60)
    print("STAGE 1: Training Tissue Segmentation Model")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = MultiClassDiceLoss()
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=5e-5
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, "stage1"))
    os.makedirs(args.stage1_checkpoint_dir, exist_ok=True)

    best_val_dice = 0.0

    for epoch in range(1, args.epochs_stage1 + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Stage1 Train E{epoch}/{args.epochs_stage1}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss_dice = criterion(outputs, masks)
            loss_ce = ce_criterion(outputs, masks)
            l1_loss = l1_regularization(model, args.l1_lambda)
            loss = 0.5 * loss_dice + 0.5 * loss_ce + l1_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(1, train_steps)
        writer.add_scalar("train/loss", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_steps = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Stage1 Val E{epoch}"):
                imgs, masks = imgs.to(device), masks.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate Dice for FGT class (class 2)
                preds = torch.argmax(outputs, dim=1)
                fgt_pred = (preds == 2).float()
                fgt_target = (masks == 2).float()

                intersection = (fgt_pred * fgt_target).sum()
                dice = (2. * intersection + 1e-5) / (fgt_pred.sum() + fgt_target.sum() + 1e-5)
                val_dice += dice.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        avg_val_dice = val_dice / max(1, val_steps)

        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("val/dice_fgt", avg_val_dice, epoch)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Dice(FGT)={avg_val_dice:.4f}")

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_path = os.path.join(args.stage1_checkpoint_dir, "best_stage1.pth")
            torch.save(model.state_dict(), save_path)
            print(f"\u2713 Saved best Stage 1 model (Dice: {best_val_dice:.4f})")

        # Save checkpoint
        if epoch % 10 == 0:
            save_path = os.path.join(args.stage1_checkpoint_dir, f"stage1_epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path)

        scheduler.step()

    writer.close()
    print(f"\nStage 1 training complete! Best Val Dice: {best_val_dice:.4f}")
    return model

def train_stage2(model, train_loader, val_loader, args):
    """Train Stage 2: Cancer Segmentation"""
    print("\n" + "="*60)
    print("STAGE 2: Training Cancer Segmentation Model")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, "stage2"))
    os.makedirs(args.stage2_checkpoint_dir, exist_ok=True)

    best_val_dice = 0.0

    for epoch in range(1, args.epochs_stage2 + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Stage2 Train E{epoch}/{args.epochs_stage2}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            l1_loss = l1_regularization(model, args.l1_lambda)
            loss = criterion(outputs, masks) + l1_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(1, train_steps)
        writer.add_scalar("train/loss", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_steps = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Stage2 Val E{epoch}"):
                imgs, masks = imgs.to(device), masks.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Dice score
                preds = torch.sigmoid(outputs)
                preds_bin = (preds > 0.5).float()
                
                intersection = (preds_bin * masks).sum()
                dice = (2. * intersection + 1e-5) / (preds_bin.sum() + masks.sum() + 1e-5)
                val_dice += dice.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        avg_val_dice = val_dice / max(1, val_steps)

        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("val/dice", avg_val_dice, epoch)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Dice={avg_val_dice:.4f}")

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_path = os.path.join(args.stage2_checkpoint_dir, "best_stage2.pth")
            torch.save(model.state_dict(), save_path)
            print(f"\u2713 Saved best Stage 2 model (Dice: {best_val_dice:.4f})")

        # Save checkpoint
        if epoch % 10 == 0:
            save_path = os.path.join(args.stage2_checkpoint_dir, f"stage2_epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path)

        scheduler.step()

    writer.close()
    print(f"\nStage 2 training complete! Best Val Dice: {best_val_dice:.4f}")
    return model

# ========================================
# Main Training Pipeline
# ========================================

def get_args():
    parser = argparse.ArgumentParser(description="Cascade Staged Segmentation Model")
    
    # Data paths (defaults are repository-relative)
    parser.add_argument("--tissue-data-dir", type=str, 
                       default="segmentation_data/train_valid",
                       help="Path to tissue segmentation data (relative)")
    parser.add_argument("--cancer-csv", type=str,
                       default="unified_segmentation_dataset.csv",
                       help="Path to cancer segmentation CSV (relative)")
    
    # Stage 1 parameters
    parser.add_argument("--epochs-stage1", type=int, default=30,
                       help="Number of epochs for Stage 1")
    parser.add_argument("--lr-stage1", type=float, default=1e-3,
                       help="Learning rate for Stage 1")
    parser.add_argument("--img-size-stage1", type=int, default=512,
                       help="Image size for Stage 1")
    parser.add_argument("--batch-size-stage1", type=int, default=8,
                       help="Batch size for Stage 1")
    
    # Stage 2 parameters
    parser.add_argument("--epochs-stage2", type=int, default=150,
                       help="Number of epochs for Stage 2")
    parser.add_argument("--lr-stage2", type=float, default=3e-4,
                       help="Learning rate for Stage 2")
    parser.add_argument("--img-size-stage2", type=int, default=384,
                       help="Image size for Stage 2")
    parser.add_argument("--batch-size-stage2", type=int, default=12,
                       help="Batch size for Stage 2")
    
    # General parameters
    parser.add_argument("--num-workers", type=int, default=4,
                       help="DataLoader workers")
    parser.add_argument("--stage1-checkpoint-dir", type=str,
                       default="checkpoints_cascade/stage1",
                       help="Stage 1 checkpoint directory")
    parser.add_argument("--stage2-checkpoint-dir", type=str,
                       default="checkpoints_cascade/stage2",
                       help="Stage 2 checkpoint directory")
    parser.add_argument("--logdir", type=str,
                       default="runs/cascade_segmentation",
                       help="TensorBoard log directory")
    
    # Regularization / Training control
    parser.add_argument("--l1-lambda", type=float, default=5e-5,
                       help="L1 regularization strength")
    # Training control
    parser.add_argument("--train-stage1", action="store_true",
                       help="Train Stage 1 (tissue segmentation)")
    parser.add_argument("--train-stage2", action="store_true",
                       help="Train Stage 2 (cancer segmentation)")
    parser.add_argument("--train-both", action="store_true",
                       help="Train both stages sequentially")
    
    return parser.parse_args()

def prepare_tissue_data(data_dir: str, val_split: float = 0.2):
    """Prepare tissue segmentation dataset (robust to extensions and mask suffixes)."""
    img_dir = Path(data_dir) / "fgt_seg"
    label_dir = Path(data_dir) / "fgt_seg_labels"

    # Accept common image extensions and search recursively
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    img_files = []
    if img_dir.exists():
        for ext in exts:
            img_files.extend(list(img_dir.rglob(f"*{ext}")))
    else:
        print(f"[WARN] Image directory not found: {img_dir}")

    img_files = sorted(set(img_files))

    image_paths = []
    mask_paths = []

    # Helper mask name candidates (stem variations)
    mask_suffixes = ["_LI", "_mask", "_label", ""]  # "" means same stem
    for img_file in img_files:
        stem = img_file.stem
        found_mask = None

        # First try common suffix patterns in the label_dir
        if label_dir.exists():
            for suffix in mask_suffixes:
                for ext in exts:
                    candidate = label_dir / f"{stem}{suffix}{ext}"
                    if candidate.exists():
                        found_mask = candidate
                        break
                if found_mask:
                    break

        # If not found, try same folder as image (maybe labels placed beside images)
        if found_mask is None:
            for suffix in mask_suffixes:
                for ext in exts:
                    candidate = img_file.parent / f"{stem}{suffix}{ext}"
                    if candidate.exists():
                        found_mask = candidate
                        break
                if found_mask:
                    break

        # If still not found, try any file in label_dir that contains the stem
        if found_mask is None and label_dir.exists():
            for p in label_dir.rglob("*"):
                if p.is_file() and stem in p.stem:
                    found_mask = p
                    break

        if found_mask is not None:
            image_paths.append(str(img_file))
            mask_paths.append(str(found_mask))
        else:
            # Keep concise: log debug optionally
            # print(f"[DEBUG] No mask for image: {img_file}")
            pass

    if len(image_paths) == 0:
        print(f"[WARN] Found 0 tissue segmentation samples in '{img_dir}' (label dir: '{label_dir}').")
        print("  - Check folder paths, file extensions, or mask naming conventions.")
        print("  - Example mask name patterns checked: *_LI.png, *_mask.png, *_label.png, same-stem files.")
        # Return four empty lists so main() can decide what to do next
        return [], [], [], []

    print(f"Found {len(image_paths)} tissue segmentation samples (paired images+masks)")

    # Train/val split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42, shuffle=True
    )

    return train_imgs, val_imgs, train_masks, val_masks

def main():
    args = get_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Construct sensible search directories (repo-relative)
    repo_root = os.getcwd()
    search_dirs = [
        os.path.join(repo_root, "data_files"),
        os.path.join(repo_root, "MINI-DDSM-Complete-JPEG-8"),
        os.path.join(repo_root, "CBIS-DDSM"),
        os.path.join(repo_root, "MINI-DDSM-Complete-JPEG-8", "MINI_IMAGES"),
        os.path.join(repo_root, "data_files", "MINI_IMAGES"),
        os.path.join(repo_root, "data_files", "CBIS_IMAGES"),
        repo_root,
    ]
    # Filter out non-existing directories to speed search
    search_dirs = [d for d in search_dirs if os.path.exists(d)]
    print(f"[INFO] Using {len(search_dirs)} search dirs for resolving CSV paths.")

    # ========================================
    # Stage 1: Tissue Segmentation Training
    # ========================================
    if args.train_stage1 or args.train_both:
        print("\n" + "="*60)
        print("Preparing Stage 1: Tissue Segmentation Data")
        print("="*60)
        
        train_imgs, val_imgs, train_masks, val_masks = prepare_tissue_data(args.tissue_data_dir)
        
        if len(train_imgs) == 0:
            # Skip Stage 1 training but continue program execution if train_both was set
            print("[ERROR] No tissue segmentation samples found. Skipping Stage 1 training.")
            print("  -> If you intended to train Stage 1, verify '--tissue-data-dir' and mask filenames.")
        else:
            train_dataset = TissueSegmentationDataset(
                train_imgs, train_masks,
                img_size=(args.img_size_stage1, args.img_size_stage1),
                augment=True
            )
            val_dataset = TissueSegmentationDataset(
                val_imgs, val_masks,
                img_size=(args.img_size_stage1, args.img_size_stage1),
                augment=False
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size_stage1,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size_stage1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # Create and train Stage 1 model
            stage1_model = ACAAtrousResUNet(in_ch=1, out_ch=4, encoder_name="resnet34")
            stage1_model = train_stage1(stage1_model, train_loader, val_loader, args)
    else:
        print("[INFO] Stage 1 training not requested (use --train-stage1 or --train-both).")
    
    # ========================================
    # Stage 2: Cancer Segmentation Training
    # ========================================
    if args.train_stage2 or args.train_both:
        print("\n" + "="*60)
        print("Preparing Stage 2: Cancer Segmentation Data")
        print("="*60)
        
        os.makedirs(args.stage2_checkpoint_dir, exist_ok=True)
        
        if not os.path.exists(args.cancer_csv):
            raise FileNotFoundError(f"Cancer CSV not found: {args.cancer_csv}")
        df_raw = pd.read_csv(args.cancer_csv)
        if df_raw.shape[0] == 0:
            raise RuntimeError("Cancer CSV is empty.")
        # Normalize CSV paths by searching for missing files under search_dirs
        df = normalize_csv_paths(df_raw, "image_file_path", "roi_mask_file_path", search_dirs)

        # For Stage 2, use image and mask paths directly from the normalized DataFrame
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        train_dataset = CancerROIDataset(
            img_size=(args.img_size_stage2, args.img_size_stage2),
            augment=True,
            dataframe=train_df,
            search_dirs=search_dirs
        )
        val_dataset = CancerROIDataset(
            img_size=(args.img_size_stage2, args.img_size_stage2),
            augment=False,
            dataframe=val_df,
            search_dirs=search_dirs
        )

        # Quick sanity: dataset sizes
        print(f"Stage 2 train samples (valid files): {len(train_dataset)} | val: {len(val_dataset)}")
        if len(train_dataset) == 0:
            # helpful diagnostics: show how many rows had existing images before normalization
            existing_before = df_raw["image_file_path"].apply(lambda p: os.path.exists(str(p))).sum()
            existing_after = df["image_file_path"].apply(lambda p: os.path.exists(str(p))).sum()
            print(f"[ERROR] Stage 2 has 0 valid training samples after normalization.")
            print(f"  - images existing before normalization: {existing_before}")
            print(f"  - images existing after normalization:  {existing_after}")
            print("  - Searched directories:", search_dirs)
            print("  - Tip: ensure that your image files are located under one of the search directories (data_files, MINI-DDSM-Complete-JPEG-8, CBIS-DDSM) or update the CSV to contain relative paths.")
            raise RuntimeError("Stage 2 training has 0 valid samples after filtering. Check your CSV paths and files exist.")

        # Build weighted sampler to mitigate class imbalance (positive vs negative masks)
        labels = []
        for mpath in train_dataset.mask_paths:
            try:
                if mpath and os.path.exists(mpath):
                    m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                    labels.append(1 if (m is not None and np.any(m > 0)) else 0)
                else:
                    labels.append(0)
            except Exception:
                labels.append(0)
        class_counts = np.bincount(labels, minlength=2)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        sample_weights = [float(class_weights[l]) for l in labels]
        if len(sample_weights) == 0:
            raise RuntimeError("Stage 2: No sample weights computed; training set is empty.")
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size_stage2,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_stage2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Create and train Stage 2 model
        stage2_model = ACAAtrousResUNet(in_ch=1, out_ch=1, encoder_name="resnet34")
        stage2_model = train_stage2(stage2_model, train_loader, val_loader, args)
    else:
        print("[INFO] Stage 2 training not requested (use --train-stage2 or --train-both).")
    
    print("Training pipeline complete.")

if __name__ == "__main__":
    main()
