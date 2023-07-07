__all__ = ["segment"]

import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.transform import probabilistic_hough_line

from .MSLD import MSLD

DEG2RAD = np.pi / 180
orientation = np.linspace(-50 * DEG2RAD, +50 * DEG2RAD, 15)
orientation_limited_msld = MSLD(33, [17, 21, 25, 27, 31], orientation)

broad_mlsd = MSLD(33, [17, 21, 25, 27, 31], orientation=15)


def segment(roi, roi_mask, refine_small_vessels=True, large_vessels=False, return_all_masks=False):
    """
    Segment vessels in a ROI. Expect the resolution of the image to be roughly 3µm/px.

    Args:
        roi (np.ndarray): ROI to segment. Should be a 2D array of shape (h, w).

    Returns:
        np.ndarray: A 2D array of shape (h, w) with the same shape as the input ROI, where each pixel is either 0 or 1.
    """

    igc = 1 - roi[..., 1].astype(float) / 255
    masks = []

    # Detect small vessels
    # - Apply MSLD
    small_vessels_p = orientation_limited_msld.multiScaleLineDetector(igc)
    small_vessels = small_vessels_p > 0.65
    small_vessels_clean = small_vessels * roi_mask
    # - Cleanup vessels mask with morphological operations
    small_vessels_clean = remove_small_objects(small_vessels_clean, 200)
    small_vessels_clean = remove_small_holes(small_vessels_clean, 100)

    final_mask = small_vessels_clean
    masks += [small_vessels_clean]

    if refine_small_vessels:
        # - Refine vessels mask using probabilistic Hough transform
        lines = probabilistic_hough_line(
            small_vessels_clean, threshold=40, line_length=70, theta=np.linspace(-60 * DEG2RAD, +60 * DEG2RAD, 60)
        )
        small_vessels_refined = np.zeros(small_vessels_clean.shape, dtype=np.uint8)
        for line in lines:
            small_vessels_refined = cv2.line(small_vessels_refined, line[0], line[1], color=1, thickness=2)
        masks += [small_vessels_refined]
        final_mask = small_vessels_refined
    else:
        small_vessels_refined = small_vessels_clean

    # - Large vessels
    if large_vessels:
        resampling_f = 1 / 3
        downscaled_igc = cv2.resize(igc, None, fx=resampling_f, interpolation=cv2.INTER_AREA)
        large_vessels_p = broad_mlsd.multiScaleLineDetector(downscaled_igc)
        large_vessels_p = cv2.resize(large_vessels_p, igc.shape, interpolation=cv2.INTER_CUBIC)
        large_vessels = large_vessels_p > 0.65

        masks += [large_vessels]
        final_mask = final_mask | large_vessels

    # Assemble vessels mask and return
    if return_all_masks:
        return masks
    return final_mask
