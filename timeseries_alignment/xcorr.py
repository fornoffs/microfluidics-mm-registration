#!/usr/bin/env python3
"""
Cross-correlation based translation estimation for timeseries alignment.

This module implements phase cross-correlation methods for translation estimation
in Mother Machine timeseries registration. Cross-correlation operates in the
Fourier domain and is particularly effective for translational alignment.
"""

# Standard library imports
from typing import Tuple

# Third-party imports
import numpy as np
from skimage.registration import phase_cross_correlation

# Default parameters
DEFAULT_UPSAMPLING_FACTOR = 10
DIFF_IMAGE_THRESHOLD = 1e-2

# Local imports
from timeseries_alignment.utils import diffimage_after_transform


def xcorr_translation(img: np.ndarray, 
                      t: int, 
                      last_angle: float, 
                      template: np.ndarray, 
                      **kwargs) -> Tuple[float, float]:
    """
    Cross-correlation based translation estimation using phase cross-correlation.
    
    This method estimates translation by computing the phase cross-correlation
    between the template and current image in the Fourier domain. It's particularly
    effective for translational alignment and can achieve sub-pixel accuracy.
    
    Parameters:
    -----------
    img : np.ndarray
        Current image to align (2D array)
    t : int
        Current timepoint (for logging/debugging)
    last_angle : float
        Previous rotation angle (not used in cross-correlation)
    template : np.ndarray
        Reference template image (2D array)
    **kwargs : dict
        Additional parameters:
        - plot : bool, optional
            Whether to show debug plots (default: False)
        - upsample_factor : int, optional
            Upsampling factor for sub-pixel accuracy (default: 10)
    
    Returns:
    --------
    tuple
        Translation shift in (y, x) format
        
    Notes:
    ------
    - Uses phase cross-correlation from scikit-image
    - Returns negative shift to align current image to template
    - Supports sub-pixel accuracy through upsampling
    - Works best with similar intensity distributions
    """
    plot = kwargs.get('plot', False)
    upsample_factor = kwargs.get('upsample_factor', DEFAULT_UPSAMPLING_FACTOR)
    
    # Compute phase cross-correlation
    shift, error, diffphase = phase_cross_correlation(
        template, img, upsample_factor=upsample_factor
    )
    
    # Return negative shift to align current image to template
    # shift[0] is y shift, shift[1] is x shift
    result_shift = (-shift[0], -shift[1])
    
    if plot:
        print(f"XCorr. t = {t}, translation = {result_shift}")
        print(f"upsample_factor: {upsample_factor}")
        
        # Show difference image after transformation
        diffimage_after_transform(template, img, result_shift, 0, DIFF_IMAGE_THRESHOLD, context=f"XCorr t={t}")
    
    return result_shift 