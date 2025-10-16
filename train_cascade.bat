@echo off

REM Quick Start Script for Cascade Segmentation Training

REM Windows PowerShell Version

echo ========================================
echo Cascade Segmentation Model Training
echo ========================================
echo.


REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found! Please install Python first.
    pause
    exit /b 1
)

echo [1/3] Checking required packages...
python -c "import torch, cv2, albumentations, segmentation_models_pytorch, pandas" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install torch torchvision opencv-python albumentations segmentation-models-pytorch pandas numpy tqdm tensorboard matplotlib scikit-learn
) else (
    echo All packages are installed!
)

echo.
echo [2/3] Training configuration:
echo   - Stage 1: Tissue Segmentation (50 epochs)
echo   - Stage 2: Cancer Segmentation (120 epochs)
echo   - Total training time: ~8-12 hours (GPU) or 24-36 hours (CPU)
echo.


REM Check if CUDA is available
python -c "import torch; print('GPU Available!' if torch.cuda.is_available() else 'Using CPU (slower)')"
echo.

echo [3/3] Starting cascade training...
echo.


REM Run training
python cascade_segmentation_model.py ^
    --train-both ^
    --tissue-data-dir segmentation_data/train_valid ^
    --cancer-csv unified_segmentation_dataset.csv ^
    --epochs-stage1 50 ^
    --epochs-stage2 120 ^
    --lr-stage1 1e-3 ^
    --lr-stage2 3e-4 ^
    --batch-size-stage1 8 ^
    --batch-size-stage2 12 ^
    --img-size-stage1 512 ^
    --img-size-stage2 384 ^
    --l1-lambda 5e-5 ^
    --num-workers 4 ^
    --stage1-checkpoint-dir checkpoints_cascade/stage1 ^
    --stage2-checkpoint-dir checkpoints_cascade/stage2 ^
    --logdir runs/cascade_segmentation

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Checkpoints saved to:
echo   - Stage 1: checkpoints_cascade/stage1/
echo   - Stage 2: checkpoints_cascade/stage2/
echo.
echo To view training progress:
echo   tensorboard --logdir runs/cascade_segmentation
echo.
echo To run inference:
echo   python cascade_inference.py --stage1-weights checkpoints_cascade/stage1/best_stage1.pth --stage2-weights checkpoints_cascade/stage2/best_stage2.pth --image-dir path/to/images --output-dir predictions
echo.

pause
