#!/usr/bin/env python3
# segmentation_model_refactored.py
"""
Refactored script for breast cancer segmentation using advanced U-Net architectures.

This variant treats each CSV row image_file_path as a full, preprocessed image
(produced by dataset_process.py). ROI cropping is disabled — whole-image input only.
"""
from __future__ import annotations
import argparse
import os
import random
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torchvision

# ----------------------------
# Dataset
# ----------------------------
class BreastSegDataset(Dataset):
    """
    Dataset that expects the CSV rows to point to full preprocessed images (grayscale)
    and full masks (same size). ROI cropping is intentionally disabled so every sample
    is the entire image produced by dataset_process.py.
    """
    def __init__(self, csv_file: str, resize: Tuple[int,int] = (512,512), augment: bool = False,
                 use_meta: bool = False):
        df = pd.read_csv(csv_file)
        self.images = df["image_file_path"].tolist()
        self.masks = df["roi_mask_file_path"].tolist()
        # keep roi columns if present but ignore them — full-image training
        self.resize = resize
        self.augment = augment
        self.use_meta = use_meta
        self.transform = self._get_transforms()

    def _get_transforms(self):
        common_transforms = [
            A.Resize(self.resize[0], self.resize[1]),
            ToTensorV2()
        ]
        if self.augment:
            aug_transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                *common_transforms
            ]
            return A.Compose(aug_transforms, additional_targets={"mask":"mask"})
        else:
            return A.Compose(common_transforms, additional_targets={"mask":"mask"})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if mask is None:
            # allow missing mask but create empty one to avoid crashes
            mask = np.zeros_like(img, dtype=np.uint8)

        # Ignore any roi columns — use whole image
        img_full = img
        mask_full = mask

        # ensure mask is binary and same size as image
        if mask_full.shape != img_full.shape:
            mask_full = cv2.resize(mask_full, (img_full.shape[1], img_full.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_full = ((mask_full > 0).astype(np.uint8) * 255)

        augmented = self.transform(image=img_full, mask=mask_full)
        img_t = augmented["image"]         # shape: CxHxW (C=1)
        mask_t = augmented["mask"].unsqueeze(0) / 255.0  # -> 1xHxW, 0..1

        if self.use_meta:
            meta = {"orig_shape": img.shape}
            return img_t.float(), mask_t.float(), meta
        else:
            return img_t.float(), mask_t.float()

def check_masks(args):
    """Save a small set of image+mask overlay checks for quick visual sanity check."""
    ds = BreastSegDataset(args.csv, resize=(args.img_size, args.img_size), augment=False)
    outdir = os.path.join(args.outdir, "check_masks")
    os.makedirs(outdir, exist_ok=True)
    n = min(32, len(ds))
    for i in range(n):
        img_t, mask_t = ds[i]
        # albumentations ToTensorV2 yields floats in 0..1 (uint -> float)
        img = (img_t * 255.0).squeeze(0).cpu().numpy().astype(np.uint8)
        mask = (mask_t.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0,0,255), 1)  # red contours
        fname = os.path.join(outdir, f"check_{i}.png")
        cv2.imwrite(fname, overlay)
    print(f"[INFO] Saved {n} mask checks to {outdir}")

# ----------------------------
# ACA block, UNet, ASPP, etc.
# (unchanged from your original, only abbreviated here for clarity)
# ----------------------------

class ACAModule(nn.Module):
    def __init__(self, skip_channels, gate_channels, reduction=8):
        super().__init__()
        self.ca = nn.Sequential(
            nn.Conv2d(skip_channels + gate_channels, max(skip_channels // reduction, 1), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(skip_channels // reduction, 1), skip_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(skip_channels + gate_channels, max(skip_channels // reduction, 1), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(skip_channels // reduction, 1), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, skip, gate):
        concat = torch.cat([skip, gate], dim=1)
        ca = self.ca(concat)
        sa = self.spatial(concat)
        refined = skip * ca * sa + skip
        return self.fuse(refined)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, bilinear=True, dropout=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if bilinear else nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_ch + in_ch, out_ch, dropout=dropout)
    def forward(self, x_decoder, x_encoder):
        x = self.up(x_decoder)
        if x.shape[2:] != x_encoder.shape[2:]:
            x = F.interpolate(x, size=x_encoder.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x_encoder, x], dim=1)
        return self.conv(out)

class UpACA(nn.Module):
    def __init__(self, in_ch_hint=None, out_ch=64, skip_ch_hint=None, dropout=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self._in_ch_hint = in_ch_hint
        self._skip_ch_hint = skip_ch_hint
        self._out_ch = out_ch
        self._dropout = dropout
        self.aca = None
        self.conv = None

    def _create_modules(self, skip_ch, gate_ch, device=None, dtype=None):
        self.aca = ACAModule(skip_channels=skip_ch, gate_channels=gate_ch)
        in_conv = skip_ch + gate_ch
        self.conv = DoubleConv(in_conv, self._out_ch, dropout=self._dropout)
        if device is not None:
            if dtype is not None:
                self.aca.to(device=device, dtype=dtype)
                self.conv.to(device=device, dtype=dtype)
            else:
                self.aca.to(device)
                self.conv.to(device)

    def forward(self, x_decoder, x_encoder):
        x = self.up(x_decoder)
        if x.shape[2:] != x_encoder.shape[2:]:
            x = F.interpolate(x, size=x_encoder.shape[2:], mode='bilinear', align_corners=False)

        skip_ch = x_encoder.shape[1]
        gate_ch = x.shape[1]

        if self.aca is None or self.conv is None:
            device = x_encoder.device
            dtype = x_encoder.dtype
            self._create_modules(skip_ch, gate_ch, device=device, dtype=dtype)

        skip_ref = self.aca(x_encoder, x)
        out = torch.cat([skip_ref, x], dim=1)
        return self.conv(out)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1,6,12,18)):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False) for r in rates])
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

class ACAAtrousUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        self.aspp = ASPP(base_ch*8, base_ch*2)
        self.up1 = UpACA(in_ch_hint=base_ch*8, out_ch=base_ch*4, skip_ch_hint=base_ch*8, dropout=False)
        self.up2 = UpACA(in_ch_hint=base_ch*4, out_ch=base_ch*2, skip_ch_hint=base_ch*4, dropout=False)
        self.up3 = UpACA(in_ch_hint=base_ch*2, out_ch=base_ch, skip_ch_hint=base_ch*2, dropout=False)
        self.up4 = UpACA(in_ch_hint=base_ch, out_ch=base_ch, skip_ch_hint=base_ch, dropout=False)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.aspp(x5)
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        self.up1 = Up(base_ch*8, base_ch*4, base_ch*8, dropout=False)
        self.up2 = Up(base_ch*4, base_ch*2, base_ch*4, dropout=False)
        self.up3 = Up(base_ch*2, base_ch, base_ch*2, dropout=False)
        self.up4 = Up(base_ch, base_ch, base_ch, dropout=False)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1)
        x3 = self.down2(x2); x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.up1(x5, x4); u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2); u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

class ConnectUNets(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.net1 = UNet(in_ch, out_ch, base_ch)
        self.net2 = UNet(in_ch + out_ch, out_ch, base_ch)
    def forward(self, x):
        pred1 = torch.sigmoid(self.net1(x))
        inp2 = torch.cat([x, pred1], dim=1)
        pred2 = self.net2(inp2)
        return pred2, pred1

class ACAAtrousResUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.encoder = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=out_ch)
        encoder_channels = self.encoder.encoder.out_channels
        self.aspp = ASPP(in_ch=encoder_channels[-1], out_ch=encoder_channels[-2])
        self.up_aca1 = UpACA(in_ch_hint=encoder_channels[-2], out_ch=encoder_channels[-3], skip_ch_hint=encoder_channels[-2])
        self.up_aca2 = UpACA(in_ch_hint=encoder_channels[-3], out_ch=encoder_channels[-4], skip_ch_hint=encoder_channels[-3])
        self.up_aca3 = UpACA(in_ch_hint=encoder_channels[-4], out_ch=encoder_channels[-5], skip_ch_hint=encoder_channels[-4])
        self.up_aca4 = UpACA(in_ch_hint=encoder_channels[-5], out_ch=encoder_channels[-5], skip_ch_hint=encoder_channels[-5])
        self.outc = nn.Conv2d(in_channels=encoder_channels[-5], out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        feats = self.encoder.encoder(x)
        if len(feats) >= 6:
            e1, e2, e3, e4, bottleneck = feats[1], feats[2], feats[3], feats[4], feats[5]
        else:
            e1, e2, e3, e4, bottleneck = feats[-5], feats[-4], feats[-3], feats[-2], feats[-1]

        d5 = self.aspp(bottleneck)
        d4 = self.up_aca1(d5, e4)
        d3 = self.up_aca2(d4, e3)
        d2 = self.up_aca3(d3, e2)
        d1 = self.up_aca4(d2, e1)
        logits = self.outc(d1)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

# ----------------------------
# Loss, Metrics & Regularization
# ----------------------------
class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for segmentation."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        inputs_sigmoid = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_sigmoid * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs_sigmoid.sum() + targets_flat.sum() + self.smooth)
        return bce + (1 - dice_score)

def l1_regularization(model: nn.Module, l1_lambda: float) -> torch.Tensor:
    return l1_lambda * sum(p.abs().sum() for p in model.parameters() if p.requires_grad)

def dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> float:
    preds_binary = (preds > 0.5).float()
    intersection = (preds_binary * targets).sum(dim=(1,2,3))
    union = preds_binary.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return ((2. * intersection + smooth) / (union + smooth)).mean().item()

# ----------------------------
# Trainer Class
# ----------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, train_loader, val_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.writer = SummaryWriter(log_dir=args.logdir)
        self.best_val_dice = 0.0
        self.pos_weight = torch.tensor([args.pos_weight], device=self.device)
        os.makedirs(args.outdir, exist_ok=True)

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}/{self.args.epochs}")
        n_batches = len(self.train_loader)
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(imgs)
            if isinstance(pred, tuple):  # Handle Connect-UNets output
                pred = pred[0]

            loss = self.criterion(pred, masks, pos_weight=self.pos_weight)
            l1_penalty = l1_regularization(self.model, self.args.l1_lambda)
            total_loss_with_reg = loss + l1_penalty

            total_loss_with_reg.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                try:
                    frac_epoch = float(epoch - 1) + float(batch_idx) / max(1, n_batches)
                    self.scheduler.step(frac_epoch)
                except Exception:
                    pass

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1_penalty.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("train/loss", avg_loss, epoch)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("train/lr", current_lr, epoch)
        print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}, LR: {current_lr:.6e}")

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        val_loss, val_dice = 0, 0
        pbar = tqdm(self.val_loader, desc=f"Val E{epoch}/{self.args.epochs}")
        with torch.no_grad():
            for imgs, masks in pbar:
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                pred = self.model(imgs)
                if isinstance(pred, tuple):
                    pred = pred[0]

                loss = self.criterion(pred, masks, pos_weight=self.pos_weight)
                val_loss += loss.item()

                preds_sigmoid = torch.sigmoid(pred)
                val_dice += dice_score(preds_sigmoid, masks)
                pbar.set_postfix(dice=f"{val_dice / (pbar.n + 1):.4f}")

        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_dice = val_dice / len(self.val_loader)

        self.writer.add_scalar("val/loss", avg_val_loss, epoch)
        self.writer.add_scalar("val/dice", avg_val_dice, epoch)
        self.writer.add_scalar("val/lr", self.optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        if avg_val_dice > self.best_val_dice:
            self.best_val_dice = avg_val_dice
            torch.save(self.model.state_dict(), os.path.join(self.args.outdir, "best.pth"))
            print(f"New best model saved with Dice: {self.best_val_dice:.4f}")

        torch.save(self.model.state_dict(), os.path.join(self.args.outdir, f"epoch_{epoch}.pth"))

        self._log_images(epoch)

    def _log_images(self, epoch: int):
        """Logs a random sample of validation predictions to TensorBoard."""
        if epoch % 3 != 0: return

        imgs, masks = next(iter(self.val_loader))
        idx = random.randint(0, imgs.size(0) - 1)
        img, mask = imgs[idx:idx+1].to(self.device), masks[idx:idx+1]

        with torch.no_grad():
            pred_logits = self.model(img)
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_prob = torch.sigmoid(pred_logits)

        def to_rgb(x: torch.Tensor) -> torch.Tensor:
            # x: 1xHxW or 3xHxW; we want 3xHxW rgb-like tensor for TB
            t = x.squeeze(0).cpu()
            if t.shape[0] == 1:
                return t.repeat(3, 1, 1)
            return t

        grid = torchvision.utils.make_grid([
            to_rgb(img.cpu()), to_rgb(mask.cpu()), to_rgb((pred_prob > 0.5).float().cpu())
        ], nrow=3, normalize=False, scale_each=True)

        self.writer.add_image("val/sample_prediction", grid, epoch)

    def run(self):
        """Main training loop."""
        for epoch in range(1, self.args.epochs + 1):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
        self.writer.close()
        print("Training finished.")

# ----------------------------
# Setup and Main Execution
# ----------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a segmentation model.")
    p.add_argument("--csv", type=str, required=True, help="Path to the training CSV file.")
    p.add_argument("--outdir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    p.add_argument("--logdir", type=str, default="runs/segmentation_cascade", help="TensorBoard log directory.")

    # Dataset and DataLoader
    p.add_argument("--img-size", type=int, default=512, help="Image size for resizing.")
    p.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers.")

    # Training Hyperparameters
    p.add_argument("--epochs", type=int, default=125, help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    p.add_argument("--pos-weight", type=float, default=11.0, help="Positive class weight for BCE loss.")
    p.add_argument("--l1-lambda", type=float, default=5e-5, help="L1 regularization strength.")

    # Model Selection
    p.add_argument("--model", type=str, default="aca-atrous-resunet",
                     choices=["aca-atrous-unet", "connect-unet", "smp-unet-resnet34", "aca-atrous-resunet"],
                     help="Select the model architecture.")

    p.add_argument("--check-masks", action="store_true",
               help="Quickly save a few image+mask overlay PNGs to outdir/check_masks for visual inspection.")

    # CosineAnnealingWarmRestarts options
    p.add_argument("--t0", type=int, default=12, help="T_0 for CosineAnnealingWarmRestarts (first restart epoch count).")
    p.add_argument("--t-mult", type=int, default=2, help="T_mult for CosineAnnealingWarmRestarts (cycle multiplier).")
    p.add_argument("--eta-min", type=float, default=5e-6, help="Minimum LR (eta_min) for CosineAnnealingWarmRestarts.")

    return p.parse_args()

def setup_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Creates and splits dataset, and returns DataLoaders."""
    dataset = BreastSegDataset(args.csv, resize=(args.img_size, args.img_size), augment=True)
    val_len = int(len(dataset) * 0.2)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    # Weighted sampler for handling class imbalance at the image level
    df = pd.read_csv(args.csv)
    mask_paths = [df.loc[idx, "roi_mask_file_path"] for idx in train_ds.indices]

    class_labels = []
    for p in tqdm(mask_paths, desc="Reading masks for sampler"):
        try:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                class_labels.append(0)
            else:
                class_labels.append(1 if m.sum() > 0 else 0)
        except Exception:
            class_labels.append(0)

    class_counts = np.bincount(class_labels)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in class_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader

# ----------------------------
# Checkpoint-aware model instantiation / robust loading helpers
# ----------------------------
import torch

def _unwrap_state_dict(raw):
    """Return a raw state_dict from typical torch.save formats."""
    if isinstance(raw, dict):
        # common wrappers
        if "state_dict" in raw:
            return raw["state_dict"]
        # sometimes saved with 'model_state_dict' etc.
        for key in ("model_state_dict", "state_dict", "net", "model"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw if isinstance(raw, dict) else {}

def _partial_load_state_dict(model: torch.nn.Module, state_dict: dict):
    """
    Update model.state_dict() with any keys in state_dict that have matching shapes.
    Returns tuple (num_loaded, num_skipped, loaded_keys, skipped_keys)
    """
    mstate = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    # Normalize keys (strip "module." if present) and match
    for k_src, v in state_dict.items():
        k = k_src
        if k.startswith("module."):
            k = k[len("module."):]
        if k in mstate and mstate[k].shape == v.shape:
            mstate[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k_src)
    model.load_state_dict(mstate)
    return len(loaded_keys), len(skipped_keys), loaded_keys, skipped_keys

def load_model_from_checkpoint(ckpt_path: str, preferred_model_name: str = "aca-atrous-unet",
                               device: torch.device = torch.device("cpu"),
                               img_size: int = 512):
    """
    Infers which architecture the checkpoint corresponds to (checks for 'encoder' keys)
    and returns an instantiated model with as many weights loaded as possible.

    Returns (model, info_str, chosen_model_name)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    state = _unwrap_state_dict(raw)

    # decide which architecture matches checkpoint keys
    keys = list(state.keys())
    # heuristic: if keys contain 'encoder.' or 'encoder.encoder' likely saved from SMP-based model
    has_encoder_keys = any(k.startswith("encoder.") or "encoder.encoder" in k or k.startswith("encoder.encoder") for k in keys)
    chosen = preferred_model_name
    if has_encoder_keys:
        # prefer SMP-based wrapper model
        chosen = "aca-atrous-resunet"
    # instantiate model
    models_map = {
        "aca-atrous-unet": ACAAtrousUNet(in_ch=1, out_ch=1, base_ch=64),
        "connect-unet": ConnectUNets(in_ch=1, out_ch=1, base_ch=64),
        "smp-unet-resnet34": smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1),
        "aca-atrous-resunet": ACAAtrousResUNet(in_ch=1, out_ch=1)
    }
    if chosen not in models_map:
        chosen = preferred_model_name  # fallback

    model = models_map[chosen].to(device)

    # run dummy forward to init lazy modules where possible
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size, device=device)
            _ = model(dummy)
        model.train()
    except Exception:
        # ignore; we'll attempt to load state_dict anyway (lazy modules may be created on first forward)
        pass

    info_lines = []
    # try strict load first
    try:
        model.load_state_dict(state, strict=True)
        info_lines.append("Loaded checkpoint with strict=True (all keys matched).")
        return model, "\n".join(info_lines), chosen
    except Exception as e:
        info_lines.append(f"strict load failed: {e}")
        # attempt partial load by matching shapes
        n_loaded, n_skipped, loaded_keys, skipped_keys = _partial_load_state_dict(model, state)
        info_lines.append(f"Partial load: loaded {n_loaded} keys, skipped {n_skipped} keys.")
        if n_skipped > 0:
            # include example skipped keys for debugging
            info_lines.append("Example skipped keys (first 10): " + ", ".join(skipped_keys[:10]))
        return model, "\n".join(info_lines), chosen

# Replace or call this helper in create_model if you prefer:
def create_model(model_name: str, device: torch.device, img_size: int, checkpoint_path: str = None) -> nn.Module:
    """
    Instantiates the selected model and optionally loads checkpoint via load_model_from_checkpoint to
    auto-detect architecture and partially load weights where possible.
    """
    if checkpoint_path is not None:
        model, info, chosen = load_model_from_checkpoint(checkpoint_path, preferred_model_name=model_name, device=device, img_size=img_size)
        print(f"[MODEL LOAD] preferred={model_name}, chosen={chosen}. Info:\n{info}")
        return model

    # original behavior if no checkpoint requested
    models: Dict[str, nn.Module] = {
        "aca-atrous-unet": ACAAtrousUNet(in_ch=1, out_ch=1, base_ch=64),
        "connect-unet": ConnectUNets(in_ch=1, out_ch=1, base_ch=64),
        "smp-unet-resnet34": smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1),
        "aca-atrous-resunet": ACAAtrousResUNet(in_ch=1, out_ch=1)
    }
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(models.keys())}")

    model = models[model_name].to(device)
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size, device=device)
            _ = model(dummy)
        model.train()
    except Exception as e:
        print(f"[WARN] Dummy init forward failed: {e}; lazy modules (if any) may not be registered for optimizer.")
    print(f"Using model: {model_name}")
    return model

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = setup_dataloaders(args)

    model = create_model(args.model, device, img_size=args.img_size)

    criterion = DiceBCELoss(smooth=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t0,
        T_mult=args.t_mult,
        eta_min=args.eta_min
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )
    trainer.run()

if __name__ == "__main__":
    main()
