#!/bin/bash
# Quick Start Script for Cascade Segmentation Training
# Linux/Mac Version

echo "========================================"
echo "Cascade Segmentation Model Training"
echo "========================================"
echo ""
echo "[3/3] Starting cascade training..."
echo ""

# Run training
python3 cascade_segmentation_model.py \
    --train-both \
    --tissue-data-dir segmentation_data/train_valid \
    --cancer-csv unified_segmentation_dataset.csv \
    --epochs-stage1 40 \
    --epochs-stage2 150 \
    --lr-stage1 5e-4 \
    --lr-stage2 1e-3 \
    --batch-size-stage1 64 \
    --batch-size-stage2 64 \
    --img-size-stage1 512 \
    --img-size-stage2 512 \
    --num-workers 4 \
    --stage1-checkpoint-dir checkpoints_cascade/stage1 \
    --stage2-checkpoint-dir checkpoints_cascade/stage2 \
    --logdir runs/cascade_segmentation \
    --l1-lambda 5e-5

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Checkpoints saved to:"
echo "  - Stage 1: checkpoints_cascade/stage1/"
echo "  - Stage 2: checkpoints_cascade/stage2/"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir runs/cascade_segmentation"
echo ""
echo "To run inference:"
echo "  python3 cascade_inference.py --stage1-weights checkpoints_cascade/stage1/best_stage1.pth --stage2-weights checkpoints_cascade/stage2/best_stage2.pth --image-dir path/to/images --output-dir predictions"
echo ""
