#!/usr/bin/env python3
"""
Utility functions for timeseries alignment framework.

This module provides performance measurement, transformation application,
and result comparison utilities for the Mother Machine registration system.
"""

# Standard library imports
import time
import os
import sys
import logging
import multiprocessing
import platform
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import psutil
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance.log'),
        logging.StreamHandler()
    ]
)

def get_system_info() -> Dict[str, str]:
    """Get basic system information"""
    return {
        'Platform': platform.platform(),
        'Python Version': sys.version,
        'CPU': platform.processor(),
        'CPU Cores (Physical)': str(psutil.cpu_count(logical=False)),
        'CPU Cores (Logical)': str(psutil.cpu_count(logical=True)),
        'Total RAM': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        'Available RAM': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
    }

def measure_runtime(method, *args, **kwargs):
    """
    Measure the runtime of a detection method.
    
    Parameters:
    -----------
    method : callable
        The detection method to evaluate.
    *args : tuple
        Positional arguments for the method.
    **kwargs : dict
        Keyword arguments for the method.
        
    Returns:
    --------
    float
        Execution time in seconds.
        
    Example:
    --------
    >>> runtime = measure_runtime(align_time_series, img_data, **params)
    >>> print(f"Runtime: {runtime:.4f} seconds")
    """
    start_time = time.time()
    method(*args, **kwargs)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.4f} seconds")
    return runtime

def measure_memory(method, *args, **kwargs):
    """
    Measure the peak memory usage of a detection method.
    
    Parameters:
    -----------
    method : callable
        The detection method to evaluate.
    *args : tuple
        Positional arguments for the method.
    **kwargs : dict
        Keyword arguments for the method.
        
    Returns:
    --------
    float
        Peak memory usage in MB.
        
    Example:
    --------
    >>> memory = measure_memory(align_time_series, img_data, **params)
    >>> print(f"Peak Memory Usage: {memory:.2f} MB")
    """
    import tracemalloc
    tracemalloc.start()
    method(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 10**6  # Convert to MB
    print(f"Peak Memory Usage: {peak_mb:.2f} MB")
    return peak_mb

def apply_transformations(
    img_data, file_path, detected_shifts, detected_rotations, channel=0,
    ome_output_path=None, dim_order="ZCTYX"
):
    """
    Apply detected transformations to the original image and save the result.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Original image data
    file_path : str
        Path to the original image file
    detected_shifts : list
        List of detected translation shifts as (y, x) tuples
    detected_rotations : list
        List of detected rotation angles in degrees
    channel : int, optional
        Channel to process (default: 0)
    ome_output_path : str, optional
        Custom output path for the transformed image
    dim_order : str, optional
        Dimension order for the output (default: "ZCTYX")
        
    Returns:
    --------
    str
        Path to the saved transformed image
    """
    from aicsimageio import AICSImage
    
    # Load the original image
    img = AICSImage(file_path)
    
    # Get image dimensions
    height, width = img_data.shape[-2:]
    
    # Process each timepoint
    for t in range(len(detected_rotations)):
        # Get current frame
        current_frame = img_data[t, channel, 0, :, :]
        
        # Calculate center for rotation
        center = (width / 2, height / 2)
        
        # Create transformation matrix combining rotation and translation
        rot_mat = cv2.getRotationMatrix2D(center, -detected_rotations[t], 1.0)
        rot_mat[0, 2] -= detected_shifts[t][1]  # Negative x translation
        rot_mat[1, 2] -= detected_shifts[t][0]  # Negative y translation
        
        # Apply transformation with zero padding
        transformed_frame = cv2.warpAffine(current_frame, rot_mat, (width, height))
        
        # Update the image data
        img_data[t, channel, 0, :, :] = transformed_frame
    
    # Save the transformed image
    if ome_output_path is None:
        base_path = os.path.splitext(file_path)[0]
        ome_output_path = f"{base_path}_aligned.tif"
    
    # Save using tifffile for better compatibility
    import tifffile
    tifffile.imwrite(ome_output_path, img_data)
    
    print(f"Transformed image saved to: {ome_output_path}")
    return ome_output_path

def diffimage_after_transform(image1, image2, shift=(0,0), angle=0, threshold=1e-2, plot=True, context=""):
    """
    Show the difference image after applying a transformation.
    
    Parameters:
    -----------
    image1 : numpy.ndarray
        Reference image
    image2 : numpy.ndarray
        Image to transform
    shift : tuple, optional
        Translation shift (y, x) (default: (0,0))
    angle : float, optional
        Rotation angle in degrees (default: 0)
    threshold : float, optional
        Threshold for difference visualization (default: 1e-2)
    plot : bool, optional
        Whether to display the plot (default: True)
    context : str, optional
        Context string to display (e.g., "HT t=5", "XCorr t=3") (default: "")
    """
    import cv2
    from skimage.transform import rotate
    
    # Apply transformation to image2
    if angle != 0:
        image2_transformed = rotate(image2, -angle, preserve_range=True)
    else:
        image2_transformed = image2.copy()
    
    if shift != (0, 0):
        # Create transformation matrix for translation
        M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
        image2_transformed = cv2.warpAffine(image2_transformed, M, (image2_transformed.shape[1], image2_transformed.shape[0]))
    
    # Calculate difference
    diff = image1.astype(float) - image2_transformed.astype(float)
    
    # Normalize difference for visualization (centered around 0)
    diff_max = max(abs(diff.min()), abs(diff.max()))
    diff_norm = diff / (diff_max + 1e-8)
    
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Add context to the main title if provided
        if context:
            fig.suptitle(f'Alignment Check: {context}', fontsize=14, fontweight='bold')
        
        # Original images
        axes[0].imshow(image1, cmap='gray')
        axes[0].set_title('Reference Image')
        axes[0].axis('off')
        
        axes[1].imshow(image2_transformed, cmap='gray')
        axes[1].set_title(f'Transformed Image\n(shift={shift}, angle={angle:.1f}°)')
        axes[1].axis('off')
        
        # Difference image
        im = axes[2].imshow(diff_norm, cmap='gray', vmin=-1, vmax=1)
        axes[2].set_title('Difference Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        # print(f"Mean difference: {diff.mean():.4f}")
        # print(f"Max difference: {diff.max():.4f}")
        # print(f"Std difference: {diff.std():.4f}")

def gradient_magnitude(img):
    """Calculate gradient magnitude using Sobel operators."""
    grad_x = sobel(img, axis=1)
    grad_y = sobel(img, axis=0)
    return np.sqrt(grad_x**2 + grad_y**2)

def calculate_similarity(image1, image2, mask=None):
    """
    Calculate similarity metrics between two images.
    
    Parameters:
    -----------
    image1 : numpy.ndarray
        First image
    image2 : numpy.ndarray
        Second image
    mask : numpy.ndarray, optional
        Mask to apply during calculation
        
    Returns:
    --------
    tuple
        (RMSE, SSIM, Edge-SSIM) similarity metrics
    """
    # Normalize images to [0, 1] range
    img1_norm = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8)
    img2_norm = (image2 - image2.min()) / (image2.max() - image2.min() + 1e-8)
    
    # Create mask if not provided
    if mask is None:
        mask = np.ones_like(img1_norm, dtype=bool)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(((img1_norm - img2_norm) * mask) ** 2))
    
    # Standard SSIM
    ssim_val = structural_similarity(img1_norm[mask], img2_norm[mask], data_range=1.0)
    
    # Edge-based SSIM (using Sobel gradients)
    grad1 = gradient_magnitude(img1_norm)
    grad2 = gradient_magnitude(img2_norm)
    edge_ssim = structural_similarity(
        grad1[mask], grad2[mask], data_range=grad1.max() - grad1.min()
    )
    
    return rmse, ssim_val, edge_ssim