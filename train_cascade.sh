#!/bin/bash
# Quick Start Script for Cascade Segmentation Training
# Linux/Mac Version

echo "========================================"
echo "Cascade Segmentation Model Training"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found! Please install Python first."
    exit 1
fi

echo "[1/3] Checking required packages..."
python3 -c "import torch, cv2, albumentations, segmentation_models_pytorch, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install torch torchvision opencv-python albumentations segmentation-models-pytorch pandas numpy tqdm tensorboard matplotlib scikit-learn
else
    echo "All packages are installed!"
fi

echo ""
echo "[2/3] Training configuration:"
echo "  - Stage 1: Tissue Segmentation (30 epochs)"
echo "  - Stage 2: Cancer Segmentation (150 epochs)"
echo "  - Total training time: ~8-12 hours (GPU) or 24-36 hours (CPU)"
echo ""

# Check if CUDA is available
python3 -c "import torch; print('GPU Available!' if torch.cuda.is_available() else 'Using CPU (slower)')"
echo ""

echo "[3/3] Starting cascade training..."
echo ""

# Run training
python3 cascade_segmentation_model.py \
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
