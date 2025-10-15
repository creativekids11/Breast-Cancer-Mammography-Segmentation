#!/usr/bin/env python3
"""
Cascade Model Inference Script

Performs two-stage segmentation:
1. Stage 1: Segments tissue types (adipose, fibroglandular, pectoral)
2. Stage 2: Segments cancer within the fibroglandular region
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model classes
from cascade_segmentation_model import (
    ACAAtrousResUNet,
    CascadeSegmentationModel
)

def visualize_cascade_results(image, tissue_pred, cancer_pred, save_path=None):
    """
    Visualize cascade segmentation results
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Tissue segmentation
    tissue_colored = np.zeros((*tissue_pred.shape, 3), dtype=np.uint8)
    tissue_colored[tissue_pred == 1] = [255, 0, 0]      # Adipose - Red
    tissue_colored[tissue_pred == 2] = [0, 255, 0]      # Fibroglandular - Green
    tissue_colored[tissue_pred == 3] = [0, 0, 255]      # Pectoral - Blue
    
    axes[1].imshow(tissue_colored)
    axes[1].set_title('Stage 1: Tissue Segmentation\nRed=Adipose, Green=FGT, Blue=Pectoral')
    axes[1].axis('off')
    
    # FGT region only
    fgt_mask = (tissue_pred == 2).astype(np.uint8) * 255
    axes[2].imshow(fgt_mask, cmap='green')
    axes[2].set_title('Fibroglandular Region (FGT)')
    axes[2].axis('off')
    
    # Cancer prediction
    axes[3].imshow(image, cmap='gray')
    cancer_overlay = np.zeros((*cancer_pred.shape, 4))
    cancer_overlay[cancer_pred > 0.5] = [1, 0, 0, 0.5]  # Red transparent
    axes[3].imshow(cancer_overlay)
    axes[3].set_title('Stage 2: Cancer Segmentation\n(on FGT region)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def load_cascade_model(stage1_weights, stage2_weights, device='cuda'):
    """Load the cascade model with both stage weights"""
    model = CascadeSegmentationModel(
        stage1_weights=stage1_weights,
        stage2_weights=stage2_weights,
        device=device
    )
    model.eval()
    return model

def predict_single_image(model, image_path, device='cuda', img_size=512):
    """
    Perform cascade prediction on a single image
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    orig_shape = img.shape
    
    # Prepare for model
    img_resized = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # Get both stage predictions
        tissue_logits, cancer_logits, tissue_pred = model(img_tensor, return_stage1=True)
        
        # Process tissue prediction
        tissue_pred_np = tissue_pred.squeeze().cpu().numpy().astype(np.uint8)
        tissue_pred_resized = cv2.resize(
            tissue_pred_np, 
            (orig_shape[1], orig_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Process cancer prediction
        cancer_prob = torch.sigmoid(cancer_logits).squeeze().cpu().numpy()
        # Resize to original FGT crop size first, then to full image
        cancer_pred_resized = cv2.resize(
            cancer_prob,
            (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    return {
        'image': img,
        'tissue_segmentation': tissue_pred_resized,
        'cancer_probability': cancer_pred_resized,
        'fgt_mask': (tissue_pred_resized == 2).astype(np.uint8)
    }

def predict_batch(model, image_dir, output_dir, device='cuda', img_size=512):
    """
    Perform cascade prediction on a batch of images
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    results = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            result = predict_single_image(model, str(img_path), device, img_size)
            
            # Save results
            base_name = img_path.stem
            
            # Save tissue segmentation
            tissue_seg_path = output_dir / f"{base_name}_tissue_seg.png"
            cv2.imwrite(str(tissue_seg_path), result['tissue_segmentation'])
            
            # Save FGT mask
            fgt_mask_path = output_dir / f"{base_name}_fgt_mask.png"
            cv2.imwrite(str(fgt_mask_path), result['fgt_mask'] * 255)
            
            # Save cancer prediction
            cancer_pred_path = output_dir / f"{base_name}_cancer_prob.png"
            cv2.imwrite(str(cancer_pred_path), (result['cancer_probability'] * 255).astype(np.uint8))
            
            # Create visualization
            vis_path = output_dir / f"{base_name}_visualization.png"
            visualize_cascade_results(
                result['image'],
                result['tissue_segmentation'],
                result['cancer_probability'],
                save_path=str(vis_path)
            )
            
            results.append({
                'image_path': str(img_path),
                'tissue_seg_path': str(tissue_seg_path),
                'fgt_mask_path': str(fgt_mask_path),
                'cancer_pred_path': str(cancer_pred_path),
                'visualization_path': str(vis_path)
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\nProcessed {len(results)} images successfully!")
    print(f"Results saved to {output_dir}")
    
    return results

def get_args():
    parser = argparse.ArgumentParser(description="Cascade Model Inference")
    
    # Model weights
    parser.add_argument("--stage1-weights", type=str, required=True,
                       help="Path to Stage 1 model weights")
    parser.add_argument("--stage2-weights", type=str, required=True,
                       help="Path to Stage 2 model weights")
    
    # Input/Output
    parser.add_argument("--image", type=str, default=None,
                       help="Path to single image for inference")
    parser.add_argument("--image-dir", type=str, default=None,
                       help="Path to directory of images for batch inference")
    parser.add_argument("--output-dir", type=str, default="cascade_predictions",
                       help="Output directory for predictions")
    
    # Parameters
    parser.add_argument("--img-size", type=int, default=512,
                       help="Image size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for inference")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Check if GPU is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Check weights exist
    if not os.path.exists(args.stage1_weights):
        print(f"Error: Stage 1 weights not found: {args.stage1_weights}")
        return
    if not os.path.exists(args.stage2_weights):
        print(f"Error: Stage 2 weights not found: {args.stage2_weights}")
        return
    
    # Load cascade model
    print("Loading cascade model...")
    model = load_cascade_model(args.stage1_weights, args.stage2_weights, device)
    print("Model loaded successfully!")
    
    # Perform inference
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = predict_single_image(model, args.image, device, args.img_size)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save and visualize
        base_name = Path(args.image).stem
        vis_path = output_dir / f"{base_name}_cascade_result.png"
        
        visualize_cascade_results(
            result['image'],
            result['tissue_segmentation'],
            result['cancer_probability'],
            save_path=str(vis_path)
        )
        
        print(f"\nResults saved to {output_dir}")
        
    elif args.image_dir:
        print(f"\nProcessing images from: {args.image_dir}")
        results = predict_batch(model, args.image_dir, args.output_dir, device, args.img_size)
        
    else:
        print("Error: Please provide either --image or --image-dir")
        return

if __name__ == "__main__":
    main()
