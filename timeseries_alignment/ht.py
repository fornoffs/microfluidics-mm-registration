#!/usr/bin/env python3
"""
Hough Transform based rotation estimation for timeseries alignment.

This module implements Hough Transform methods for rotation estimation
in Mother Machine timeseries registration. The Hough Transform detects
lines in images and estimates rotation angles based on horizontal line detection.
"""

# Standard library imports
import numpy as np

# Third-party imports
import cv2
import matplotlib.pyplot as plt
from scipy import stats

# --- Constants for Hough transform parameters ---
# Gaussian blur parameters
GAUSSIAN_KERNEL_SIZE = (5, 5)  # Kernel size for Gaussian blur
GAUSSIAN_SIGMA = 1  # Standard deviation for Gaussian blur

# Canny edge detection parameters
CANNY_LOW_THRESHOLD = 50  # Lower threshold for Canny edge detection
CANNY_HIGH_THRESHOLD = 150  # Upper threshold for Canny edge detection
CANNY_APERTURE_SIZE = 3  # Aperture size for Canny edge detection

# Hough transform parameters
HOUGH_RHO = 1  # Distance resolution in pixels
HOUGH_THETA = np.pi/180  # Angle resolution in radians
HOUGH_THRESHOLD = 100  # Minimum number of votes
HOUGH_MIN_LINE_LENGTH = 100  # Minimum line length
HOUGH_MAX_LINE_GAP = 10  # Maximum gap between line segments

# Angle processing parameters
ANGLE_NORMALIZATION_RANGE = (-45, 45)  # Range for angle normalization
ANGLE_FILTER_THRESHOLD = 40  # Maximum angle to consider for horizontal lines

def ht_rotation(single_img, plot=False):
    """
    Detect the rotation angle of an image by finding horizontal lines using Hough transform.
    
    Args:
        single_img (ndarray): Input image to analyze
        plot (bool): If True, displays intermediate plots for visualization
        
    Returns:
        float: Detected rotation angle in degrees, rounded to 0.1 precision
    """
    # Ensure the image is in the correct format for Canny edge detection
    if single_img.ndim == 3:
        single_img = cv2.cvtColor(single_img, cv2.COLOR_BGR2GRAY)
    elif single_img.ndim != 2:
        raise ValueError("Input image must be 2D or 3D with color channels.")

    # Convert image to 8-bit if necessary, also normalize
    if single_img.dtype != np.uint8:
        single_img = cv2.normalize(single_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Gaussian blur to reduce noise
    smoothed_image = cv2.GaussianBlur(single_img, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(
        smoothed_image,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
        apertureSize=CANNY_APERTURE_SIZE
    )
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(edges, cmap='gray')
        plt.title("Edge Detection Result")
        plt.axis('off')
        plt.show()

    # Apply Probabilistic Hough transform to detect lines
    lines = cv2.HoughLinesP(
        edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        print("No lines detected")
        return 0.0  # Return 0 if no lines detected

    # Find the most common angle among detected lines
    angles = []
    valid_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Normalize angle to -45 to 45 range
        if angle < ANGLE_NORMALIZATION_RANGE[0]:
            angle += 180
        elif angle > ANGLE_NORMALIZATION_RANGE[1]:
            angle -= 180
            
        # Only consider angles that are close to horizontal
        if abs(angle) < ANGLE_FILTER_THRESHOLD:
            angles.append(angle)
            valid_lines.append((x1, y1, x2, y2))  # Store the valid lines for plotting

    if not angles:
        print("No suitable angles found, returning 0.0")
        return 0.0  # Return 0.0 if no suitable angles are found
    
    # Find the best angle
    
    # Calculate IQR to filter outliers
    q1 = np.percentile(angles, 25)
    q3 = np.percentile(angles, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out outliers
    filtered_angles = [angle for angle in angles if lower_bound <= angle <= upper_bound]
    
    # Use median of filtered angles
    best_angle = np.median(filtered_angles)
    #best_angle = stats.mode(filtered_angles, keepdims=False)[0]
    if plot and valid_lines:  # Only plot if we have valid lines
        plt.figure(figsize=(10, 6))
        plt.imshow(edges, cmap='gray')
        
        # Plot only the valid lines
        for x1, y1, x2, y2 in valid_lines:
            plt.plot([x1, x2], [y1, y2], 'r-')
        
        plt.title(f"Detected Lines using Hough Transform\n(Only showing lines with angles < {ANGLE_FILTER_THRESHOLD}°)")
        plt.axis('off')
        plt.show()
        print(f"Best angle: {best_angle}")
        print("all angles: ", angles)
    #print(f"Best angle: {best_angle}")
    #print("all angles: ", angles)
    return float(-best_angle)
