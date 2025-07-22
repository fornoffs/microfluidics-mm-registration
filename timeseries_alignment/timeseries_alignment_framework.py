#!/usr/bin/env python3
"""
Timeseries alignment framework for Mother Machine registration.

This module provides wrapper functions and the main alignment pipeline
for HT, ORB, and XCorr registration methods.
"""

# Standard library imports
import numpy as np

# Third-party imports
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rotate

# --- Constants for image processing ---
# ORB feature detection parameters
DEFAULT_CROP_FRACTION = 1/3  # Crop to 1/3 of image width to focus on relevant features
DEFAULT_CROP_SIDE = 'right'  # Default side to crop from
DEFAULT_ORB_FEATURES = 1000  # Number of ORB features to detect for reliable matching
# XCorr parameters
DEFAULT_UPSAMPLING_FACTOR = 10  # Factor to upsample image for more precise angle detection # subpixel accuracy (0.1 pixel precision)
# Visualization parameters
DIFF_IMAGE_THRESHOLD = 1e-2  # Threshold for difference image visualization

# Local imports
from timeseries_alignment.xcorr import xcorr_translation
from timeseries_alignment.ht import ht_rotation
from timeseries_alignment.orb import detect_and_match_keypoints, crop_roi
from timeseries_alignment.utils import diffimage_after_transform

# --- Wrappers for alignment methods ---
def ht_rotation_wrapper(img, t, last_angle, template, **kwargs):
    """
    Wrapper for HT-based rotation estimation that applies rotation at each step.
    
    Parameters:
    -----------
    img : ndarray
        Current image to align
    t : int
        Current timepoint (not used in HT rotation)
    last_angle : float
        Previous rotation angle (not used in HT rotation)
    template : ndarray
        Reference image to align against (not used in HT rotation)
    
    Additional Parameters:
    --------------------
    upsample_factor : int, optional
        Factor to upsample image for more precise angle detection (default: 1)
        
    Returns:
    --------
    float
        Detected rotation angle in degrees
        
    Raises:
    ------
    ValueError
        If input image is None or has invalid shape
    """
    # Input validation
    if img is None:
        raise ValueError("Input image cannot be None")
    
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a numpy array")
    
    if img.ndim != 2:
        raise ValueError(f"Input image must be 2D, got shape {img.shape}")
    
    if img.size == 0:
        raise ValueError("Input image cannot be empty")
    
    plot = kwargs.get('plot', False)
    
    try:
        angle = ht_rotation(img, plot=plot)
    except Exception as e:
        print(f"Error in HT rotation at timepoint {t}: {e}")
        return 0.0  # Return 0 angle on error

    if plot:
        print("Note: HT does not use a template.")
        print(f"HT. t = {t}, rotation angle = {angle:.2f}°")
        try:
            diffimage_after_transform(template, img, (0, 0), angle, DIFF_IMAGE_THRESHOLD, context=f"HT t={t}")
        except Exception as e:
            print(f"Warning: Could not display difference image: {e}")
    
    return angle

def ht_prev_rotation_wrapper(img, t, last_angle, template, **kwargs):
    """
    Wrapper for HT-based rotation estimation with temporal initialization.
    First rotates the current image by the previous angle, then estimates rotation
    relative to the template.
    
    Parameters:
    -----------
    img : ndarray
        Current image to align
    t : int
        Current timepoint
    last_angle : float
        Previous rotation angle to apply to current image
    template : ndarray
        Reference image to align against
    
    Additional Parameters:
    --------------------
    upsample_factor : int, optional
        Factor to upsample image for more precise angle detection (default: 1)
        
    Returns:
    --------
    float
        Combined rotation angle (previous + current) in degrees
        
    Raises:
    ------
    ValueError
        If input image is None or has invalid shape
    """
    # Input validation
    if img is None:
        raise ValueError("Input image cannot be None")
    
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a numpy array")
    
    if img.ndim != 2:
        raise ValueError(f"Input image must be 2D, got shape {img.shape}")
    
    if not isinstance(t, int) or t < 0:
        raise ValueError(f"Timepoint must be non-negative integer, got {t}")
    
    if not isinstance(last_angle, (int, float)):
        raise ValueError(f"Last angle must be numeric, got {type(last_angle)}")
    
    plot = kwargs.get('plot', False)
    
    try:
        # Only rotate if we have a previous angle
        if t > 0:        
            img = rotate(img, -last_angle, preserve_range=True)

        current_angle = ht_rotation(img, plot=plot)
        
        if plot:
            print("Note: HT does not use a template.")
            print(f"HT-Prev. t = {t}, rotation angle = {current_angle:.2f}°")
            try:
                img_original = rotate(img, last_angle, preserve_range=True)
                diffimage_after_transform(template, img_original, (0, 0), last_angle+current_angle, DIFF_IMAGE_THRESHOLD, context=f"HT-Prev t={t}")
            except Exception as e:
                print(f"Warning: Could not display difference image: {e}")
        
        return last_angle + current_angle
        
    except Exception as e:
        print(f"Error in HT-Prev rotation at timepoint {t}: {e}")
        return last_angle  # Return previous angle on error

def orb_rotation_wrapper(img, t, last_angle, template, **kwargs):
    """
    Wrapper for ORB-based rotation estimation that applies rotation at each step.
    
    Additional Parameters:
    --------------------
    crop_fraction : float, optional
        Fraction of image to use for ORB detection (default: 1/3)
    crop_side : str, optional
        Which side to crop from ('right' or 'left') (default: 'right')
    n_features : int, optional
        Number of ORB features to detect (default: 1000)
    """
    plot = kwargs.get('plot', False)

    # Crop both images
    crop_fraction = kwargs.get('crop_fraction', DEFAULT_CROP_FRACTION)
    crop_side = kwargs.get('crop_side', DEFAULT_CROP_SIDE)
    ref_crop, _ = crop_roi(template, crop_fraction, side=crop_side)
    img_crop, _ = crop_roi(img, crop_fraction, side=crop_side)
    
    # Get keypoints and matches
    (
        ref_8bit, mov_8bit, kp1, kp2, good_matches,
        desc1, desc2, M_rot, mask_rot
    ) = detect_and_match_keypoints(
        ref_crop, img_crop, n_features=kwargs.get('n_features', DEFAULT_ORB_FEATURES), estimate_matrix=True, matrix_type='affine'
    )
    
    # Check if we have enough good matches
    if M_rot is None or len(good_matches) < 4:
        if plot:
            print(f"Not enough good matches for rotation estimation (found {len(good_matches)})")
        return 0
        
    # Extract rotation angle
    angle = np.degrees(np.arctan2(M_rot[0, 1], M_rot[0, 0]))

    if plot:
        inlier_matches_rot = [m for i, m in enumerate(good_matches) if mask_rot[i]]
        img_matches_rot = cv2.drawMatches(
            ref_8bit, kp1, mov_8bit, kp2, inlier_matches_rot[:150], None,
            matchColor=(240, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(20, 10))
        plt.imshow(img_matches_rot)
        plt.title(f'Rotation Step: Inlier Matches (showing top 50 of {len(inlier_matches_rot)} matches)')
        plt.show()
        print(f"ORB. t = {t}, rotation angle = {angle:.2f}°")
        diffimage_after_transform(template, img, (0, 0), angle, DIFF_IMAGE_THRESHOLD, context=f"ORB t={t}")
        
    return angle

def orb_prev_rotation_wrapper(img, t, last_angle, template, **kwargs):
    """
    Wrapper for ORB-based rotation estimation with temporal initialization.
    First rotates the current image by the previous angle, then estimates rotation
    relative to the template.

    Parameters:
    -----------
    img : ndarray
        Current image to align
    t : int
        Current timepoint
    last_angle : float
        Previous rotation angle
    template : ndarray
        Reference image to align against
    
    Additional Parameters:
    --------------------
    crop_fraction : float, optional
        Fraction of image to use for ORB detection (default: 1/3)
    crop_side : str, optional
        Which side to crop from ('right' or 'left') (default: 'right')
    n_features : int, optional
        Number of ORB features to detect (default: 1000)
    """
    from timeseries_alignment.orb import detect_and_match_keypoints, crop_roi
    import matplotlib.pyplot as plt
    import cv2
    from timeseries_alignment.utils import diffimage_after_transform
    plot = kwargs.get('plot', False)

    # For t > 0, rotate the image by the previous angle
    if t > 0:
        img = rotate(img, -last_angle, preserve_range=True)
    
    # Crop both images
    crop_fraction = kwargs.get('crop_fraction', DEFAULT_CROP_FRACTION)
    crop_side = kwargs.get('crop_side', DEFAULT_CROP_SIDE)
    ref_crop, _ = crop_roi(template, crop_fraction, side=crop_side)
    img_crop, _ = crop_roi(img, crop_fraction, side=crop_side)
    
    # Get keypoints and matches
    (
        ref_8bit, mov_8bit, kp1, kp2, good_matches,
        desc1, desc2, M_rot, mask_rot
    ) = detect_and_match_keypoints(
        ref_crop, img_crop, n_features=kwargs.get('n_features', DEFAULT_ORB_FEATURES), estimate_matrix=True, matrix_type='affine'
    )
    
    # Check if we have enough good matches
    if M_rot is None or len(good_matches) < 4:
        if plot:
            print(f"Not enough good matches for rotation estimation (found {len(good_matches)})")
        return 0
        
    # Extract rotation angle
    current_angle = np.degrees(np.arctan2(M_rot[0, 1], M_rot[0, 0]))

    if plot:
        inlier_matches_rot = [m for i, m in enumerate(good_matches) if mask_rot[i]]
        img_matches_rot = cv2.drawMatches(
            ref_8bit, kp1, mov_8bit, kp2, inlier_matches_rot[:150], None,
            matchColor=(240, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(20, 10))
        plt.imshow(img_matches_rot)
        plt.title(f'Rotation Step: Inlier Matches (showing top 50 of {len(inlier_matches_rot)} matches)')
        plt.show()
        print(f"ORB-Prev. t = {t}, rotation angle = {current_angle+last_angle}°")
        print(f"current angle: {current_angle}°", f"last angle: {last_angle}°")
        # Undo the rotation to get back to the original image
        img_original = rotate(img, last_angle, preserve_range=True)
        diffimage_after_transform(template, img_original, (0, 0), current_angle+last_angle, DIFF_IMAGE_THRESHOLD, context=f"ORB-Prev t={t}")
        
    return current_angle+last_angle

def xcorr_translation_wrapper(img, t, last_angle, template, **kwargs):
    """
    Wrapper for XCorr-based translation estimation.
    
    Parameters:
    -----------
    img : ndarray
        Current image to align
    t : int
        Current timepoint
    last_angle : float
        Previous rotation angle (not used in XCorr translation)
    template : ndarray
        Reference image to align against
    
    Additional Parameters:
    --------------------
    upsample_factor : int, optional
        Upsampling factor for sub-pixel accuracy (default: 10)
        
    Returns:
    --------
    tuple
        Translation shift in (y, x) format
    """
    plot = kwargs.get('plot', False)
    shift = xcorr_translation(img, t, last_angle, template, **kwargs)
    
    if plot:
        print(f"XCorr. t = {t}, translation = {shift}")
        
    return shift

def orb_translation_wrapper(img, t, last_angle, template, **kwargs):
    """
    Wrapper for ORB-based translation estimation.
    
    Parameters:
    -----------
    img : ndarray
        Current image to align
    t : int
        Current timepoint (not used in ORB translation)
    last_angle : float
        Previous rotation angle (not used in ORB translation)
    template : ndarray
        Reference image to align against
    
    Additional Parameters:
    --------------------
    crop_fraction : float, optional
        Fraction of image to use for ORB detection (default: 1/3)
    crop_side : str, optional
        Which side to crop from ('right' or 'left') (default: 'right')
    n_features : int, optional
        Number of ORB features to detect (default: 1000)
        
    Returns:
    --------
    tuple
        Translation shift in (y, x) format
    """
    from timeseries_alignment.orb import detect_and_match_keypoints, crop_roi
    import matplotlib.pyplot as plt
    import cv2
    from timeseries_alignment.utils import diffimage_after_transform
    plot = kwargs.get('plot', False)

    # Crop both images
    crop_fraction = kwargs.get('crop_fraction', DEFAULT_CROP_FRACTION)
    crop_side = kwargs.get('crop_side', DEFAULT_CROP_SIDE)
    ref_crop, _ = crop_roi(template, crop_fraction, side=crop_side)
    img_crop, _ = crop_roi(img, crop_fraction, side=crop_side)
    
    # Get keypoints and matches
    (
        ref_8bit, mov_8bit, kp1, kp2, good_matches,
        desc1, desc2, M_trans, mask_trans
    ) = detect_and_match_keypoints(
        ref_crop, img_crop, n_features=kwargs.get('n_features', DEFAULT_ORB_FEATURES), estimate_matrix=True, matrix_type='translation'
    )
    
    # Check if we have enough good matches
    if M_trans is None or len(good_matches) < 4:
        if plot:
            print(f"Not enough good matches for translation estimation (found {len(good_matches)})")
        return (0, 0)
        
    # Extract translation in (y, x) format to match original implementation
    shift = (M_trans[1, 2], M_trans[0, 2])  # Changed from (M_trans[0, 2], M_trans[1, 2])
    
    if plot:
        inlier_matches_trans = [m for i, m in enumerate(good_matches) if mask_trans[i]]
        img_matches_trans = cv2.drawMatches(
            ref_8bit, kp1, mov_8bit, kp2, inlier_matches_trans[:150], None,
            matchColor=(240, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(20, 10))
        plt.imshow(img_matches_trans)
        plt.title(f'Translation Step: Inlier Matches (showing top 50 of {len(inlier_matches_trans)} matches)')
        plt.show()
        print(f"ORB. t = {t}, translation = {shift}")
        diffimage_after_transform(template, img, shift, 0, DIFF_IMAGE_THRESHOLD, context=f"ORB-Trans t={t}")
        
    return shift

# --- Template creation ---
def create_template_hough(img_data, channel=0, plot=False, t=0):
    """
    Create a template by detecting and correcting rotation in the first timepoint.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input image data array
    channel : int, optional
        Channel index to process (default: 0)
    plot : bool, optional
        Whether to display plots (default: False)
    t : int, optional
        Timepoint to use for template creation (default: 0)
        
    Returns:
    --------
    tuple
        (template, angle) where template is the rotated image and angle is the detected rotation
        
    Raises:
    ------
    ValueError
        If input data is invalid or channel/timepoint out of bounds
    """
    # Input validation
    if img_data is None:
        raise ValueError("Input image data cannot be None")
    
    if not isinstance(img_data, np.ndarray):
        raise ValueError("Input image data must be a numpy array")
    
    if img_data.ndim != 5:
        raise ValueError(f"Input image data must be 5D (T,C,Z,Y,X), got shape {img_data.shape}")
    
    if channel < 0 or channel >= img_data.shape[1]:
        raise ValueError(f"Channel {channel} out of bounds for data with {img_data.shape[1]} channels")
    
    if t < 0 or t >= img_data.shape[0]:
        raise ValueError(f"Timepoint {t} out of bounds for data with {img_data.shape[0]} timepoints")
    
    try:
        # Get first timepoint image
        first_frame = img_data[t, channel, 0, :, :]
        
        if first_frame.size == 0:
            raise ValueError("Selected frame is empty")
        
        if plot:
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(first_frame, cmap='gray')
            plt.title(f'Original Image (t={t})')
            plt.axis('off')
        
        # Get rotation angle using HT
        angle = ht_rotation(first_frame, plot=plot)
        
        # Apply rotation to create template
        template = rotate(first_frame, -angle, preserve_range=True)
        
        if plot:
            plt.subplot(122)
            plt.imshow(template, cmap='gray')
            plt.title(f'Rotated Template (angle={angle:.2f}°)')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return template, angle
        
    except Exception as e:
        print(f"Error creating template at timepoint {t}, channel {channel}: {e}")
        # Return a fallback template (original frame) and 0 angle
        return img_data[t, channel, 0, :, :], 0.0

from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


# --- General time-series alignment function ---
def align_time_series(
    img_data,
    channel=0,
    rotation_method=None,      # function(img, t, prev_angle, template, **kwargs) -> angle
    translation_method=None,   # function(img, t, prev_angle,template, **kwargs) -> shift
    template=None,
    plot=False,                # Whether to print information and show plots during alignment
    use_template_prev=False,   # only applies to cc: use previous aligned image as template
    **kwargs
):
    """
    Main function to align a timeseries using specified rotation and translation methods.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input image data with shape (T, C, Z, Y, X)
    channel : int, optional
        Channel to process (default: 0)
    rotation_method : callable, optional
        Function for rotation estimation
    translation_method : callable, optional
        Function for translation estimation
    template : numpy.ndarray, optional
        Reference template (if None, will be created from first timepoint)
    plot : bool, optional
        Whether to show debug plots (default: False)
    use_template_prev : bool, optional
        Whether to use previous aligned image as template (default: False)
    **kwargs : dict
        Additional parameters passed to methods
        
    Returns:
    --------
    tuple
        (detected_shifts, detected_angles) - lists of transformations for each timepoint
        
    Raises:
    ------
    ValueError
        If input data is invalid or required parameters are missing
    """
    # Input validation
    if img_data is None:
        raise ValueError("Input image data cannot be None")
    
    if not isinstance(img_data, np.ndarray):
        raise ValueError("Input image data must be a numpy array")
    
    if img_data.ndim != 5:
        raise ValueError(f"Input image data must be 5D (T,C,Z,Y,X), got shape {img_data.shape}")
    
    if channel < 0 or channel >= img_data.shape[1]:
        raise ValueError(f"Channel {channel} out of bounds for data with {img_data.shape[1]} channels")
    
    if img_data.shape[0] == 0:
        raise ValueError("Input data has no timepoints")
    
    # Check if at least one method is provided
    if rotation_method is None and translation_method is None:
        raise ValueError("At least one of rotation_method or translation_method must be provided")
    """
    General time-series alignment function.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Image data array with shape (T, C, Z, Y, X)
    channel : int, optional
        Channel index to process (default: 0)
    rotation_method : function, optional
        Function for rotation estimation (default: None)
    translation_method : function, optional
        Function for translation estimation (default: None)
    template : numpy.ndarray, optional
        Reference template image (default: None). If None, the first timepoint image is used as template.
    plot : bool, optional
        Whether to show plots and print information (default: False)
    use_template_prev : bool, optional
        Whether to use the previous aligned image as template, only applies to cc (default: False)
        
        Additional Parameters:
    --------------------
    crop_fraction : float, optional
        Fraction of image to use for ORB detection (default: 1/3)
    crop_side : str, optional
        Which side to crop from ('right' or 'left') (default: 'right')
    n_features : int, optional
        Number of ORB features to detect (default: 1000)
    upsample_factor : int, optional
        Precision factor for subpixel alignment in CC (default: 10)
        
    Returns:
    --------
    tuple
        (shifts, angles) where shifts is a list of (y, x) shifts and angles is a list of rotation angles
    """
    import matplotlib.pyplot as plt
    import cv2
    n_timepoints = img_data.shape[0]
    angles = []
    shifts = []
    last_angle = 0

    # Check if input data has correct shape
    if len(img_data.shape) != 5:
        raise ValueError(f"Expected 5D array (T, C, Z, Y, X), got shape {img_data.shape}")
    
    # Check if channel index is valid
    if channel >= img_data.shape[1]:
        raise ValueError(f"Channel index {channel} out of range (0-{img_data.shape[1]-1})")

    # Prepare template
    if template is None:
        template_img = img_data[0, channel, 0, :, :]
    else:
        # Handle case where template might be a tuple (template, angle)
        if isinstance(template, tuple) and len(template) == 2:
            template_img = template[0]  # Extract just the image from (template, angle)
        else:
            template_img = template
    
    if use_template_prev:
        template_prev = template_img

    # loop through all timepoints
    for t in range(n_timepoints):
        try:
            img = img_data[t, channel, 0, :, :]
            
            # Validate current frame
            if img is None or img.size == 0:
                print(f"Warning: Empty frame at timepoint {t}, skipping...")
                angles.append(0.0)
                shifts.append((0.0, 0.0))
                continue
            
            # Rotation
            if rotation_method is not None:
                try:
                    # Pass plot parameter to the rotation method
                    angle = rotation_method(img, t, last_angle, template=template_img, plot=plot, **kwargs)
                    if not isinstance(angle, (int, float)):
                        print(f"Warning: Invalid angle type at timepoint {t}, using 0.0")
                        angle = 0.0
                except Exception as e:
                    print(f"Error in rotation method at timepoint {t}: {e}")
                    angle = 0.0
            else:
                angle = 0.0
                
            angles.append(angle)
            last_angle = angle
            
        except Exception as e:
            print(f"Error processing timepoint {t}: {e}")
            angles.append(0.0)
            shifts.append((0.0, 0.0))
            continue

        # Only apply rotation if we're doing translation (since we need the rotated image for that)
        if translation_method is not None and angle != 0:
            img = rotate(img, -angle, preserve_range=True)
            if plot:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.imshow(img, cmap='gray')
                plt.title(f'Image at t={t} (after rotation)')
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(template_img, cmap='gray')
                plt.title('Template')
                plt.axis('off')
                plt.tight_layout()
                plt.show()


     # Translation
        if translation_method is not None: 
            try:
                if use_template_prev and t > 0:
                    # Get previous image and transform it by previous rotation and translation
                    prev_img = img_data[t-1, channel, 0, :, :]
                    height, width = prev_img.shape
                    center = (width / 2, height / 2)
                    
                    # Create transformation matrix combining rotation and translation
                    rot_mat = cv2.getRotationMatrix2D(center, -angles[-1], 1.0)
                    rot_mat[0, 2] -= shifts[-1][1]  # Negative x translation
                    rot_mat[1, 2] -= shifts[-1][0]  # Negative y translation
                    
                    # Apply transformation with zero padding
                    prev_img = cv2.warpAffine(
                        prev_img, rot_mat, (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                    template_prev = prev_img
                    shift = translation_method(img, t, angle, template=template_prev, plot=plot, **kwargs)
                else:
                    shift = translation_method(img, t, angle, template=template_img, plot=plot, **kwargs)
                
                # Validate shift result
                if not isinstance(shift, (tuple, list)) or len(shift) != 2:
                    print(f"Warning: Invalid shift type at timepoint {t}, using (0, 0)")
                    shift = (0.0, 0.0)
                elif not all(isinstance(x, (int, float)) for x in shift):
                    print(f"Warning: Invalid shift values at timepoint {t}, using (0, 0)")
                    shift = (0.0, 0.0)
                    
            except Exception as e:
                print(f"Error in translation method at timepoint {t}: {e}")
                shift = (0.0, 0.0)
        else:
            shift = (0.0, 0.0)
        shifts.append(shift)

        # Print information if plot
        if plot:
            print(f"at the end of the general alignment function: t={t}, angle={angle}, shift={shift}")

    return shifts, angles

def process_rotation(t, img, template_img, plot, rotation_method, reference_angle, **kwargs):
    """Process rotation for a single timepoint"""
    if rotation_method is not None:
        # Pass plot parameter to the rotation method
        angle = rotation_method(img, t, reference_angle, template=template_img, plot=plot, **kwargs)
    else:
        angle = 0
    return t, angle

def process_translation(t, img, angle, template_img, plot, translation_method, **kwargs):
    """Process translation for a single timepoint"""
    img_rotated = rotate(img, -angle, preserve_range=True)
    if translation_method is not None:
        shift = translation_method(img_rotated, t, angle, template=template_img, plot=plot, **kwargs)
    else:
        shift = (0, 0)
    return t, shift

def process_batch_rotation(batch_indices, img_data, channel, template_img, plot, rotation_method, first_angle, **kwargs):
    """Process a batch of rotations"""
    batch_results = []
    for t in batch_indices:
        img = img_data[t, channel, 0, :, :]
        _, angle = process_rotation(t, img, template_img, plot, rotation_method, first_angle, **kwargs)
        batch_results.append((t, angle))
    return batch_results

def process_batch_translation(batch_indices, img_data, channel, angles, shifts, template_img, plot, translation_method, use_template_prev, **kwargs):
    """Process a batch of translations"""
    batch_results = []
    for t in batch_indices:
        img = img_data[t, channel, 0, :, :]
        img_rotated = rotate(img, -angles[t], preserve_range=True)
        if use_template_prev and t > 0:
            # Get previous image and transform it
            prev_img = img_data[t-1, channel, 0, :, :]
            height, width = prev_img.shape
            center = (width / 2, height / 2)
            
            # Create transformation matrix
            rot_mat = cv2.getRotationMatrix2D(center, -angles[-1], 1.0)
            rot_mat[0, 2] -= shifts[-1][1]
            rot_mat[1, 2] -= shifts[-1][0]
            
            # Apply transformation
            prev_img = cv2.warpAffine(
                prev_img, rot_mat, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            template_prev = prev_img
            shift = translation_method(img_rotated, t, angles[t], template=template_prev, plot=plot, **kwargs)
        else:
            shift = translation_method(img_rotated, t, angles[t], template=template_img, plot=plot, **kwargs)
        batch_results.append((t, shift))
    return batch_results

def align_time_series_multiprocessing(
    img_data,
    channel=0,
    rotation_method=None,
    translation_method=None,
    template=None,
    plot=False,
    use_template_prev=False,
    n_processes=None,
    **kwargs
):
    """Time series alignment with multiprocessing support."""
    import matplotlib.pyplot as plt
    import cv2
    import multiprocessing as mp
    from functools import partial
    n_timepoints = img_data.shape[0]
    angles = []
    shifts = []
    #print("hello")
    #print(f"upsample_factor: {kwargs.get('upsample_factor', DEFAULT_UPSAMPLING_FACTOR)}")

    # Check if input data has correct shape
    if len(img_data.shape) != 5:
        raise ValueError(f"Expected 5D array (T, C, Z, Y, X), got shape {img_data.shape}")
    
    # Check if channel index is valid
    if channel >= img_data.shape[1]:
        raise ValueError(f"Channel index {channel} out of range (0-{img_data.shape[1]-1})")
    
    # Prepare template
    if template is None:
        template_img = img_data[0, channel, 0, :, :]
    else:
        # Handle case where template might be a tuple (template, angle)
        if isinstance(template, tuple) and len(template) == 2:
            template_img = template[0]  # Extract just the image from (template, angle)
        else:
            template_img = template
    
    if use_template_prev:
        template_prev = template_img
        # Can't use multiprocessing with use_template_prev as it depends on previous results
        n_processes = 1

    if n_processes is None:
        n_processes = mp.cpu_count()

    # Process first timepoint sequentially to get reference angle
    first_img = img_data[0, channel, 0, :, :]
    _, first_angle = process_rotation(0, first_img, template_img, plot, rotation_method, 0, **kwargs)
    angles.append(first_angle)

    # Process remaining rotations in parallel
    if n_processes > 1 and not use_template_prev:
        with mp.Pool(processes=n_processes) as pool:
            results = []
            for t in range(1, n_timepoints):
                img = img_data[t, channel, 0, :, :]
                results.append(pool.apply_async(
                    process_rotation,
                    args=(t, img, template_img, plot, rotation_method, first_angle),
                    kwds=kwargs
                ))
            
            # Get results in order
            for t in range(1, n_timepoints):
                _, angle = results[t-1].get()
                angles.append(angle)
    else:
        # Sequential processing for rotations
        for t in range(1, n_timepoints):
            img = img_data[t, channel, 0, :, :]
            _, angle = process_rotation(t, img, template_img, plot, rotation_method, first_angle, **kwargs)
            angles.append(angle)

    # Process translations in parallel
    if translation_method is not None:
        if n_processes > 1 and not use_template_prev:
            with mp.Pool(processes=n_processes) as pool:
                results = []
                for t in range(n_timepoints):
                    img = img_data[t, channel, 0, :, :]
                    results.append(pool.apply_async(
                        process_translation,
                        args=(t, img, angles[t], template_img, plot, translation_method),
                        kwds=kwargs
                    ))
                
                # Get results in order
                for t in range(n_timepoints):
                    _, shift = results[t].get()
                    shifts.append(shift)
        else:
            # Sequential processing for translations
            for t in range(n_timepoints):
                img = img_data[t, channel, 0, :, :]
                img_rotated = rotate(img, -angles[t], preserve_range=True)
                if use_template_prev and t > 0:
                    # Get previous image and transform it by previous rotation and translation
                    prev_img = img_data[t-1, channel, 0, :, :]
                    height, width = prev_img.shape
                    center = (width / 2, height / 2)
                    
                    # Create transformation matrix combining rotation and translation
                    rot_mat = cv2.getRotationMatrix2D(center, -angles[-1], 1.0)
                    rot_mat[0, 2] -= shifts[-1][1]  # Negative x translation
                    rot_mat[1, 2] -= shifts[-1][0]  # Negative y translation
                    
                    # Apply transformation with zero padding
                    prev_img = cv2.warpAffine(
                        prev_img, rot_mat, (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                    template_prev = prev_img
                    shift = translation_method(img_rotated, t, angles[t], template=template_prev, plot=plot, **kwargs)
                else:
                    shift = translation_method(img_rotated, t, angles[t], template=template_img, plot=plot, **kwargs)
                shifts.append(shift)

                # Print information if plot
                if plot:
                    print(f"at the end of the general alignment function: t={t}, angle={angles[t]}, shift={shift}")

    return shifts, angles
