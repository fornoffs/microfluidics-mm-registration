#!/usr/bin/env python3
"""
ORB (Oriented FAST and Rotated BRIEF) feature-based registration methods.

This module implements ORB-based rotation and translation estimation for
Mother Machine timeseries registration. ORB provides rotation-invariant
feature detection and matching for robust alignment.
"""

# Standard library imports
import numpy as np

# Third-party imports
import cv2
import matplotlib.pyplot as plt

# Local imports
from timeseries_alignment.utils import diffimage_after_transform
from timeseries_alignment.timeseries_alignment_framework import DEFAULT_CROP_FRACTION
from timeseries_alignment.timeseries_alignment_framework import DEFAULT_CROP_SIDE

# --- Constants for ORB feature detection and matching ---
# Lowe's ratio test parameters
LOWE_RATIO_THRESHOLD = 0.7  # Threshold for Lowe's ratio test in feature matching

# RANSAC parameters for robust transformation estimation
RANSAC_REPROJ_THRESHOLD = 3.0  # Maximum allowed reprojection error in pixels
RANSAC_MAX_ITERS = 2000  # Maximum number of RANSAC iterations
RANSAC_CONFIDENCE = 0.99  # Required confidence for RANSAC

def crop_roi(image, crop_fraction=DEFAULT_CROP_FRACTION, side=DEFAULT_CROP_SIDE):
    width = image.shape[1]
    if side == 'right':
        start_x = int((1 - crop_fraction) * width)
        return image[:, start_x:], start_x
    elif side == 'left':
        end_x = int(crop_fraction * width)
        return image[:, :end_x], 0
    else:
        raise ValueError("side must be 'right' or 'left'")
    
def detect_and_match_keypoints(
    image1, image2, n_features=1000, 
    estimate_matrix=False, matrix_type='affine', ransac_params=None,
    filter_method='none', filter_kwargs=None
):
    """
    Detects and matches keypoints between two images using ORB and BFMatcher with Lowe's ratio test.
    Optionally estimates and returns the transformation matrix and mask.
    """
    # Convert images to uint8 if needed
    if image1.dtype != np.uint8:
        image1_8bit = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        image1_8bit = image1.copy()
    if image2.dtype != np.uint8:
        image2_8bit = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        image2_8bit = image2.copy()
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints1, descriptors1 = orb.detectAndCompute(image1_8bit, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2_8bit, None)
    if descriptors1 is None or descriptors2 is None or len(keypoints1) < 4 or len(keypoints2) < 4:
        if estimate_matrix:
            return image1_8bit, image2_8bit, keypoints1, keypoints2, [], descriptors1, descriptors2, None, None
        else:
            return image1_8bit, image2_8bit, keypoints1, keypoints2, [], descriptors1, descriptors2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < LOWE_RATIO_THRESHOLD * n.distance] if matches else []
    if not estimate_matrix:
        return image1_8bit, image2_8bit, keypoints1, keypoints2, good_matches, descriptors1, descriptors2
    if len(good_matches) < 4:
        return image1_8bit, image2_8bit, keypoints1, keypoints2, good_matches, descriptors1, descriptors2, None, None
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    if ransac_params is None:
        ransac_params = dict(
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE
        )
    if matrix_type == 'affine':
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, **ransac_params)
    elif matrix_type == 'full_affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, **ransac_params)
    elif matrix_type == 'translation':
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, **ransac_params)
        if M is not None:
            M[0,0] = 1.0  # scale
            M[1,1] = 1.0  # scale
            M[0,1] = 0.0  # rotation
            M[1,0] = 0.0  # rotation
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")
    return image1_8bit, image2_8bit, keypoints1, keypoints2, good_matches, descriptors1, descriptors2, M, mask