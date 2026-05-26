"""
CLAHE Contrast Enhancement & Linear Percentile Stretching Module
Author: Antigravity AI Pair Programmer
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def lin_stretch_img(img: np.ndarray, low_prc: float = 0.1, high_prc: float = 99.9) -> np.ndarray:
    """
    Perform linear percentile stretching to enhance image dynamic range.
    
    Args:
        img: Input grayscale image.
        low_prc: Lower percentile to truncate.
        high_prc: Upper percentile to truncate.
        
    Returns:
        Stretched image scaled and clipped to [0, 255] uint8.
    """
    lo, hi = np.percentile(img, (low_prc, high_prc))
    if hi <= lo:
        # Avoid division by zero for uniform or blank images
        return img.astype(np.uint8)
        
    stretched = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        img: Input grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
        
    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Consolidated processing pipeline: linear stretching followed by CLAHE.
    
    Args:
        img: Input BGR/RGB or grayscale image.
        
    Returns:
        Enhanced grayscale image.
    """
    # Convert BGR/RGB to Grayscale if necessary
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
    # Apply linear stretching (0.1% to 99.9% percentile)
    stretched = lin_stretch_img(img, 0.1, 99.9)
    
    # Apply local adaptive histogram equalization (CLAHE)
    enhanced = apply_clahe(stretched, clip_limit=2.0, tile_grid_size=(8, 8))
    
    return enhanced

def process_folder(source_dir: str, dest_dir: str, target_size: int = 512) -> None:
    """
    Process an entire directory of X-ray images, applying CLAHE, resizing to target_size, and saving.
    
    Args:
        source_dir: Directory containing original images.
        dest_dir: Directory to save preprocessed images.
        target_size: Dimension to resize image to (standardizing to 512x512).
    """
    os.makedirs(dest_dir, exist_ok=True)
    supported_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(supported_extensions)]
    print(f"🔍 Found {len(files)} images in {source_dir}. Starting preprocessing...")
    
    for f in tqdm(files, desc="Preprocessing Images"):
        src_path = os.path.join(source_dir, f)
        img = cv2.imread(src_path)
        if img is None:
            print(f"⚠️ Failed to load {f}, skipping.")
            continue
            
        # Run enhancement pipeline
        enhanced = preprocess_image(img)
        
        # Resize to standard size (512x512)
        if target_size is not None and (enhanced.shape[0] != target_size or enhanced.shape[1] != target_size):
            enhanced = cv2.resize(enhanced, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
        dest_path = os.path.join(dest_dir, f)
        cv2.imwrite(dest_path, enhanced)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Image CLAHE Preprocessing Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to original images folder")
    parser.add_argument("--dest", type=str, required=True, help="Path to save processed images")
    parser.add_argument("--size", type=int, default=512, help="Target resize dimension (default: 512)")
    
    args = parser.parse_args()
    process_folder(args.source, args.dest, args.size)
