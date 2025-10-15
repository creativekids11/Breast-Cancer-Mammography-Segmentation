CASCADE SEGMENTATION MODEL - VISUAL WORKFLOW
============================================

                    INPUT MAMMOGRAM IMAGE
                           |
                           | (512×512 grayscale)
                           ▼
        ╔══════════════════════════════════════════╗
        ║          STAGE 1: TISSUE SEGMENTATION     ║
        ║                                          ║
        ║    ┌─────────────────────────────────┐  ║
        ║    │   ResNet34 Encoder              │  ║
        ║    │   (5 levels of features)        │  ║
        ║    └──────────┬──────────────────────┘  ║
        ║               ▼                          ║
        ║    ┌─────────────────────────────────┐  ║
        ║    │   ASPP (Atrous Spatial Pyramid) │  ║
        ║    │   Multi-scale context           │  ║
        ║    │   Dilations: 1, 6, 12, 18       │  ║
        ║    └──────────┬──────────────────────┘  ║
        ║               ▼                          ║
        ║    ┌─────────────────────────────────┐  ║
        ║    │   ACA Decoder (4 stages)        │  ║
        ║    │   • Channel Attention           │  ║
        ║    │   • Spatial Attention           │  ║
        ║    │   • Skip Connections            │  ║
        ║    └──────────┬──────────────────────┘  ║
        ║               ▼                          ║
        ║    Output: 4-class Segmentation         ║
        ║    • Class 0: Background (dark)         ║
        ║    • Class 1: Adipose (red)             ║
        ║    • Class 2: Fibroglandular (green) ★  ║
        ║    • Class 3: Pectoral (blue)           ║
        ╚══════════════════════════════════════════╝
                           |
                           | Extract FGT Region (Class 2)
                           ▼
              ┌────────────────────────┐
              │  FGT Bounding Box      │
              │  • Find class 2 pixels │
              │  • Get min/max coords  │
              │  • Add padding         │
              │  • Crop image          │
              └──────────┬─────────────┘
                         ▼
                  FGT ROI (384×384)
                         |
                         ▼
        ╔══════════════════════════════════════════╗
        ║      STAGE 2: CANCER SEGMENTATION        ║
        ║                                          ║
        ║    ┌─────────────────────────────────┐  ║
        ║    │   ResNet34 Encoder              │  ║
        ║    │   (5 levels of features)        │  ║
        ║    └──────────┬──────────────────────┘  ║
        ║               ▼                          ║
        ║    ┌─────────────────────────────────┐  ║
        ║    │   ASPP (Atrous Spatial Pyramid) │  ║
        ║    │   Multi-scale context           │  ║
        ║    └──────────┬──────────────────────┘  ║
        ║               ▼                          ║
        ║    ┌─────────────────────────────────┐  ║
        ║    │   ACA Decoder (4 stages)        │  ║
        ║    │   • Channel Attention           │  ║
        ║    │   • Spatial Attention           │  ║
        ║    │   • Skip Connections            │  ║
        ║    └──────────┬──────────────────────┘  ║
        ║               ▼                          ║
        ║    Output: Binary Segmentation          ║
        ║    • Cancer probability [0-1]           ║
        ║    • Threshold at 0.5 for mask          ║
        ╚══════════════════════════════════════════╝
                           |
                           ▼
                  FINAL OUTPUTS
          ┌────────────────────────────────┐
          │ 1. Tissue Segmentation Map     │
          │ 2. FGT Mask                    │
          │ 3. Cancer Probability Map      │
          │ 4. Combined Visualization      │
          └────────────────────────────────┘


ACA MODULE DETAIL
=================

    Input: Skip Connection + Gate Signal
           |
           ├──────────────┬──────────────┐
           |              |              |
           ▼              ▼              ▼
    Channel Attention  Spatial Attention
           |              |
           ▼              ▼
      [1×1 Conv]    [3×3 Conv]
           |              |
           ▼              ▼
       Sigmoid        Sigmoid
           |              |
           └──────┬───────┘
                  ▼
         Element-wise Multiply
                  ▼
            Refined Features
                  ▼
           Feature Fusion
                  ▼
              Output


TRAINING FLOW
=============

Stage 1 Training:
    Data Loading → Augmentation → Forward Pass → 
    Loss Calculation (Dice + CE) → Backprop → 
    Optimizer Step → Validation → Save Best Model

Stage 2 Training:
    Data Loading → Augmentation → Forward Pass → 
    Loss Calculation (Dice-BCE) → Backprop → 
    Optimizer Step → Validation → Save Best Model


INFERENCE FLOW
==============

1. Load trained Stage 1 model
2. Load trained Stage 2 model
3. For each input image:
   a. Stage 1 prediction → Tissue types
   b. Extract FGT region (class 2)
   c. Stage 2 prediction on FGT → Cancer
   d. Combine results
   e. Visualize and save


DATA FLOW DIAGRAM
=================

TRAINING:

  segmentation_data/        unified_segmentation_dataset.csv
          |                              |
          ▼                              ▼
  TissueSegDataset              CancerROIDataset
          |                              |
          ▼                              ▼
    DataLoader                      DataLoader
          |                              |
          ▼                              ▼
   ACAAtrousResUNet              ACAAtrousResUNet
      (4 classes)                   (1 class)
          |                              |
          ▼                              ▼
    Training Loop                  Training Loop
          |                              |
          ▼                              ▼
  best_stage1.pth              best_stage2.pth


INFERENCE:

     Input Image
          |
          ▼
  best_stage1.pth → Tissue Mask
          |
          ▼
    Extract FGT
          |
          ▼
  best_stage2.pth → Cancer Mask
          |
          ▼
    Visualizations


PERFORMANCE METRICS
===================

Stage 1 (Tissue Segmentation):
    ┌─────────────────┬──────────┐
    │ Metric          │ Expected │
    ├─────────────────┼──────────┤
    │ FGT Dice        │ 0.85-0.92│
    │ Overall Acc     │ 0.88-0.94│
    │ Training Time   │ 2-4 hrs  │
    └─────────────────┴──────────┘

Stage 2 (Cancer Segmentation):
    ┌─────────────────┬──────────┐
    │ Metric          │ Expected │
    ├─────────────────┼──────────┤
    │ Cancer Dice     │ 0.75-0.85│
    │ Sensitivity     │ 0.80-0.90│
    │ Specificity     │ 0.85-0.92│
    │ Training Time   │ 4-6 hrs  │
    └─────────────────┴──────────┘


FILE ORGANIZATION
=================

BreastCancerAI/
│
├── CASCADE_*.md/.py          ← Documentation & Summary
│   ├── CASCADE_README.md
│   ├── QUICK_START.md
│   ├── CASCADE_IMPLEMENTATION_SUMMARY.py
│   └── CASCADE_VISUAL_DIAGRAM.py (this file)
│
├── cascade_*.py               ← Main Code
│   ├── cascade_segmentation_model.py  (Training)
│   ├── cascade_inference.py           (Inference)
│   └── test_cascade_model.py          (Testing)
│
├── train_cascade.*            ← Quick Start Scripts
│   ├── train_cascade.bat              (Windows)
│   └── train_cascade.sh               (Linux/Mac)
│
├── segmentation_data/         ← Stage 1 Data
│   ├── train_valid/
│   │   ├── fgt_seg/           (Images)
│   │   └── fgt_seg_labels/    (Masks)
│   └── test/
│
├── unified_segmentation_dataset.csv  ← Stage 2 Data
│
├── checkpoints_cascade/       ← Saved Models
│   ├── stage1/
│   └── stage2/
│
└── runs/cascade_segmentation/ ← TensorBoard Logs
    ├── stage1/
    └── stage2/


USAGE EXAMPLES
==============

# 1. Test everything works
python test_cascade_model.py

# 2. Train both stages
python cascade_segmentation_model.py --train-both

# 3. Train only Stage 1
python cascade_segmentation_model.py --train-stage1 --epochs-stage1 50

# 4. Train only Stage 2 (after Stage 1)
python cascade_segmentation_model.py --train-stage2 --epochs-stage2 100

# 5. Custom training parameters
python cascade_segmentation_model.py \
    --train-both \
    --batch-size-stage1 4 \
    --batch-size-stage2 6 \
    --lr-stage1 5e-4 \
    --lr-stage2 5e-4

# 6. Monitor training
tensorboard --logdir runs/cascade_segmentation

# 7. Run inference on single image
python cascade_inference.py \
    --stage1-weights checkpoints_cascade/stage1/best_stage1.pth \
    --stage2-weights checkpoints_cascade/stage2/best_stage2.pth \
    --image test_image.png

# 8. Run batch inference
python cascade_inference.py \
    --stage1-weights checkpoints_cascade/stage1/best_stage1.pth \
    --stage2-weights checkpoints_cascade/stage2/best_stage2.pth \
    --image-dir test_images/ \
    --output-dir predictions/


KEY ADVANTAGES
==============

✓ FOCUSED DETECTION
  Cancer detection only on relevant tissue (FGT)
  
✓ FEWER FALSE POSITIVES
  Adipose and pectoral regions excluded automatically
  
✓ BETTER FEATURE LEARNING
  Each stage specializes in its specific task
  
✓ CLINICAL RELEVANCE
  Mirrors radiologist workflow
  
✓ MODULAR TRAINING
  Train stages independently or together
  
✓ INTERPRETABLE RESULTS
  See both tissue types AND cancer locations
  
✓ SCALABLE
  Easy to add more stages or modify architecture
