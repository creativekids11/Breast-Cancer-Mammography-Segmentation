    `Breast Cancer Segmentation` is a project made by ~ Devansh Madake
<hr>

**NOTE: USE COMMAND `git lfs clone https://github.com/creativekids11/Breast-Cancer-Mammography-Segmentation.git` for cloning this repo, due to large files**

## Get Started: 
- Download the prepare_to_train.py only in a new folder
- Then run it to get fully prepared zip file ready for training!
- just run the cmd in the respective folder of the code including the `sh` or `cmd` files and run them to start training!

In this 2nd stage breast cancer segmentation model is trained on data of `The Complete Mini-DDSM` (5 gb) and `CBIS-DDSM`.
With the stage 1 model trained on data given on: https://github.com/tiryakiv/Mammogram-segmentation
A lot thanks for the data providers to give the data for free

The used model and methodologies archieve a high ~0.81 dice scores

# Summary of used type of: ACAAtrousResUNet

**ACAAtrousResUNet** is a hybrid segmentation head that reuses a pretrained **ResNet34 encoder** (via `segmentation_models_pytorch.Unet`) and replaces the usual decoder with:

- an **ASPP (atrous spatial pyramid pooling)** module on the encoder bottleneck to capture multiscale context, and then
- a stack of **UpACA** decoder blocks that perform upsampling + **ACA attention** (channel + spatial attention that refines encoder skip features using the decoder “gate”), followed by regular convolutional fusion.

So the net combines the representational power of a ResNet encoder, multiscale context (ASPP), and attention-guided skip fusion (ACA) in a U-Net–like upsampling path.

---

## Components (what they are & why)

### DoubleConv
Two conv layers (3×3) each followed by `BatchNorm` + `ReLU`. Standard U‑Net style block for local feature extraction and refinement. Optional small `Dropout2d` appended when `dropout=True`.

**Effect:** learns stronger local features, reduces checkerboard artifacts vs single conv.

### Down
`MaxPool2d(2)` then `DoubleConv`: encoder downsampling block. Produces progressively smaller spatial maps with more channels.

### Up
A conventional upsampling block: bilinear upsample (or `ConvTranspose` if `bilinear=False`) → concat with encoder skip → `DoubleConv`.

Used in the plain `UNet` implementation.

### ACAModule (the core of ACA)
This is the attention block that refines the encoder skip-map using the decoder features (the “gate”):

- It concatenates `skip` and `gate` along channel dim.
- `ca` (channel attention) path: reduces channels → `ReLU` → project back → `Sigmoid`. Produces a per-channel attention vector that modulates `skip`.
- `spatial` path: a 3×3 conv (padding=1) that reduces channels → produces a single-channel spatial mask → `Sigmoid`. Gives location-specific weighting.
- Result: `refined = skip * ca * sa + skip` (residual-style refinement).
- `fuse` : `1×1` Conv + `BatchNorm` + `ReLU` to mix channels after refinement.

**Effect:** the decoder feature `gate` guides which channels and spatial positions of the encoder `skip` are most relevant before concatenation. This reduces irrelevant information passed through skip connections.

### UpACA
Decoder upsample block that:

- Upsamples decoder tensor,
- If not already created, lazily constructs an `ACAModule` and a `DoubleConv` matching the real skip and gate channel sizes (determined at first forward; this avoids having to know encoder channel counts at init time),
- Applies ACA to refine the skip, concatenates refined skip + upsampled decoder, then applies `DoubleConv`.

**Important:** It lazily creates submodules on first call (so the dummy forward initialization elsewhere is important to register parameters with optimizer).

### ASPP (Atrous Spatial Pyramid Pooling)
- Multiple parallel dilated convs with dilation rates in `rates=(1,6,12,18)` by default.
- Each conv outputs `out_ch` channels → they are concatenated → `BN` + `ReLU` → `1×1` projection back to `out_ch`.

**Purpose:** capture multiscale context and enlarge receptive field without reducing spatial resolution — useful for small/varied tumour sizes.

### ACAAtrousUNet
A pure-from-scratch U-Net variant that stacks the `Down`/`ASPP`/`UpACA` components. Useful as a baseline when you don't want pretrained encoders.

### ACAAtrousResUNet (the model described)
- `self.encoder = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=out_ch)`  
  Uses `segmentation_models_pytorch.Unet` primarily to access a `ResNet34` encoder (`self.encoder.encoder`), whose extracted feature maps will be used.  
  **Note:** specifying `in_channels=1` with `encoder_weights="imagenet"` commonly produces a warning because ImageNet weights expect 3 channels; `smp` adjusts the first conv or reinitializes it but pretrained benefit may be reduced for single-channel input.

- `encoder_channels = self.encoder.encoder.out_channels` — collects channel counts for each encoder stage (used to size `ASPP` and `UpACA` constructors).

- `self.aspp = ASPP(in_ch=encoder_channels[-1], out_ch=encoder_channels[-2])` — runs `ASPP` on the deepest encoder feature (bottleneck) and projects to a channel count matching the previous encoder stage for decoder compatibility.

- `self.up_aca1..4` — a chain of `UpACA` blocks that progressively upsample from the `ASPP` output and refine encoder skips `e4, e3, e2, e1`.

- `self.outc` — `1×1` conv to produce final logits; final output interpolated to input resolution.

---

## Forward pass (step-by-step)

1. `feats = self.encoder.encoder(x)` — get the list of intermediate encoder features from ResNet34. Typical order in `smp` is `[stage0, stage1, stage2, stage3, stage4, stage5]` with `stage5` being deepest (bottleneck).
2. Extract `e1..e4` and `bottleneck` (the code handles variable-length by indexing from end if needed).
3. `d5 = self.aspp(bottleneck)` — apply multi-rate context on bottleneck.
4. `d4 = self.up_aca1(d5, e4)` — upsample and refine with ACA using encoder feature `e4`.
5. Continue: `d3 = up_aca2(d4, e3)`, `d2 = up_aca3(d3, e2)`, `d1 = up_aca4(d2, e1)`.
6. `logits = self.outc(d1)` and return interpolated logits to input size.

---

## Why this architecture?

- **ResNet encoder** provides strong, pretrained features (edges → semantics).
- **ASPP** gives large receptive field and multiscale context — important when tumor sizes vary widely.
- **ACA** improves skip connection usefulness by removing irrelevant encoder features (channel + spatial gating guided by decoder state), which helps reduce false positive propagation from encoder to decoder.
- Combining these produces a decoder that is both context-aware and selective about skip information.

<hr>

Though this model may seem complex it offer a fast inference on codes

# Cascade Staged Segmentation Model for Breast Cancer Detection

## Overview

This project implements a **two-stage cascade segmentation model** using **ACA-ResUNet** architecture for breast cancer detection from mammography images.

### Architecture

#### Stage 1: Tissue Segmentation
- **Input**: Full mammogram image (grayscale)
- **Output**: Multi-class segmentation (4 classes)
  - Class 0: Background
  - Class 1: Adipose tissue
  - Class 2: Fibroglandular tissue (FGT) - **Region of Interest**
  - Class 3: Pectoral muscle
- **Model**: ACA-ResUNet with ResNet34 encoder
- **Purpose**: Identify the fibroglandular tissue region where breast cancer typically occurs

#### Stage 2: Cancer Segmentation
- **Input**: Cropped FGT region from Stage 1
- **Output**: Binary segmentation (cancer vs. normal tissue)
- **Model**: ACA-ResUNet with ResNet34 encoder
- **Purpose**: Detect and segment cancerous regions within the FGT

### Why Cascade Approach?

1. **Focus on Relevant Regions**: Fibroglandular tissue is where breast cancer develops. By segmenting it first, we can focus computational resources on the most important area.

2. **Reduced False Positives**: By constraining cancer detection to FGT regions, we eliminate false positives from adipose tissue and pectoral muscle.

3. **Better Feature Learning**: Each stage specializes in its task, leading to better overall performance.

4. **Clinical Relevance**: Mirrors radiologist workflow - identify tissue types first, then focus on suspicious areas.

## Data Structure

### Stage 1 Data: Tissue Segmentation
```
segmentation_data/
├── train_valid/
│   ├── fgt_seg/              # Original mammogram images
│   │   ├── A_0002_1.RIGHT_MLO.png
│   │   └── ...
│   └── fgt_seg_labels/       # Tissue segmentation masks
│       ├── A_0002_1.RIGHT_MLO_LI.png
│       └── ...
└── test/
    ├── fgt_seg/
    └── fgt_seg_labels/
```

**Mask Values**:
- 64: Adipose tissue
- 128/192: Fibroglandular tissue (FGT)
- 255: Pectoral muscle

### Stage 2 Data: Cancer Segmentation
Uses your existing cancer segmentation CSV:
```
unified_segmentation_dataset.csv
```

## Installation

```bash
# Install required packages
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install pandas numpy tqdm
pip install tensorboard matplotlib
```

## Training

### Train Both Stages Sequentially
```bash
python cascade_segmentation_model.py \
  --train-both \
  --tissue-data-dir segmentation_data/train_valid \
  --cancer-csv unified_segmentation_dataset.csv \
  --epochs-stage1 30 \
  --epochs-stage2 150 \
  --lr-stage1 1e-3 \
  --lr-stage2 3e-4 \
  --batch-size-stage1 8 \
  --batch-size-stage2 12 \
  --img-size-stage1 512 \
  --img-size-stage2 384 \
  --l1-lambda 5e-5 \
  --num-workers 4 \
  --stage1-checkpoint-dir checkpoints_cascade/stage1 \
  --stage2-checkpoint-dir checkpoints_cascade/stage2 \
  --logdir runs/cascade_segmentation
```

### Train Stage 1 Only (Tissue Segmentation)
```bash
python cascade_segmentation_model.py \
  --train-stage1 \
  --tissue-data-dir segmentation_data/train_valid \
  --epochs-stage1 30 \
  --lr-stage1 1e-3 \
  --batch-size-stage1 8 \
  --img-size-stage1 512 \
  --l1-lambda 5e-5 \
  --num-workers 4 \
  --stage1-checkpoint-dir checkpoints_cascade/stage1 \
  --logdir runs/cascade_segmentation
```

### Train Stage 2 Only (Cancer Segmentation)
```bash
python cascade_segmentation_model.py \
  --train-stage2 \
  --cancer-csv unified_segmentation_dataset.csv \
  --epochs-stage2 150 \
  --lr-stage2 3e-4 \
  --batch-size-stage2 12 \
  --img-size-stage2 384 \
  --l1-lambda 5e-5 \
  --num-workers 4 \
  --stage2-checkpoint-dir checkpoints_cascade/stage2 \
  --logdir runs/cascade_segmentation
```

## Inference

### Single Image Prediction
```bash
python cascade_inference.py \
    --stage1-weights checkpoints_cascade/stage1/best_stage1.pth \
    --stage2-weights checkpoints_cascade/stage2/best_stage2.pth \
    --image path/to/mammogram.png \
    --output-dir predictions
```

### Batch Prediction
```bash
python cascade_inference.py \
    --stage1-weights checkpoints_cascade/stage1/best_stage1.pth \
    --stage2-weights checkpoints_cascade/stage2/best_stage2.pth \
    --image-dir path/to/image/folder \
    --output-dir predictions
```

## Model Architecture Details

### ACA-ResUNet Components

1. **Encoder**: ResNet34 pretrained on ImageNet
   - Captures hierarchical features from mammogram images
   - 5 encoding stages with increasing receptive fields

2. **ASPP (Atrous Spatial Pyramid Pooling)**
   - Multiple parallel dilated convolutions
   - Captures multi-scale context
   - Dilation rates: 1, 6, 12, 18

3. **ACA (Adaptive Context Aggregation) Module**
   - **Channel Attention**: Emphasizes important feature channels
   - **Spatial Attention**: Focuses on relevant spatial locations
   - **Feature Fusion**: Combines attention-refined features

4. **Decoder**: Progressive upsampling with skip connections
   - 4 upsampling stages
   - Each stage uses ACA module for skip connection refinement
   - Gradually recovers spatial resolution

### Loss Functions

**Stage 1** (Multi-class):
- Combination of Multi-class Dice Loss + Cross-Entropy Loss
- Weights: 0.5 Dice + 0.5 CE

**Stage 2** (Binary):
- Dice-BCE Loss
- Combines Dice coefficient for overlap + BCE for pixel-wise accuracy

## Training Parameters

### Stage 1: Tissue Segmentation
- **Epochs**: 50
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Image Size**: 512x512
- **Batch Size**: 8
- **Augmentation**: Flip, Rotate, Brightness/Contrast, GaussNoise

### Stage 2: Cancer Segmentation
- **Epochs**: 100
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=15, T_mult=2)
- **Image Size**: 384x384
- **Batch Size**: 12
- **Augmentation**: Flip, Rotate, Brightness/Contrast, ElasticTransform

## Output Files

After training:
```
checkpoints_cascade/
├── stage1/
│   ├── best_stage1.pth          # Best Stage 1 model
│   └── stage1_epoch*.pth        # Epoch checkpoints
└── stage2/
    ├── best_stage2.pth          # Best Stage 2 model
    └── stage2_epoch*.pth        # Epoch checkpoints

runs/cascade_segmentation/       # TensorBoard logs
├── stage1/
└── stage2/
```

After inference:
```
predictions/
├── image_name_tissue_seg.png      # Tissue segmentation mask
├── image_name_fgt_mask.png        # FGT region mask
├── image_name_cancer_prob.png     # Cancer probability map
└── image_name_visualization.png   # Combined visualization
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir runs/cascade_segmentation
```

Metrics tracked:
- Training/Validation Loss
- Dice Score (FGT for Stage 1, Cancer for Stage 2)
- Learning Rate

## Expected Performance

### Stage 1: Tissue Segmentation
- **FGT Dice Score**: ~0.85-0.92
- Accurately identifies fibroglandular tissue regions
- Clear separation between tissue types

### Stage 2: Cancer Segmentation
- **Cancer Dice Score**: ~0.75-0.85 (depends on cancer dataset quality)
- Focused detection within FGT regions
- Reduced false positives

## Key Features

1. ✅ **Two-stage cascade architecture** for focused cancer detection
2. ✅ **ACA-ResUNet** with attention mechanisms for better feature learning
3. ✅ **ASPP** for multi-scale context aggregation
4. ✅ **Automatic FGT ROI extraction** for Stage 2
5. ✅ **Comprehensive augmentation** pipeline
6. ✅ **TensorBoard logging** for monitoring
7. ✅ **Modular design** - train stages independently or together
8. ✅ **Visualization tools** for results interpretation

## License
This project is for research and educational purposes.

## Authors
Devansh Madake
Created for Hackathon 2.0 - Breast Cancer AI Project

# Data Preproccessing
* The data is pre-proccessed with convectional CLAHE and Median filter.

# Testing Part
* For testing the simple and fast cv2 platform is made with guide printed in terminal on execution
* There is also a adaptive CLAHE feature which helps in giving the model more confidence on confusing images
