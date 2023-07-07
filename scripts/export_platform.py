__all__ = ["process_img"]

import argparse
import os
from os.path import join as pjoin

import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.transform import probabilistic_hough_line

from lib.MSLD import MSLD
from lib.preprocessing import enhance_microscopy
from lib.roi_utilities import extract_ROIs, extract_tumor_ellipse, find_scale, load_img_mask

DEG2RAD = np.pi / 180
orientation = np.linspace(-50 * DEG2RAD, +50 * DEG2RAD, 15)
msld_detector = MSLD(33, [17, 21, 25, 27, 31], orientation)


def process_img(path: str, output_path: str):
    output_folders = ("Raw", "Preprocessed", "Vessels")
    for folder_name in output_folders:
        folder_path = pjoin(output_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    if os.path.isdir(path):
        files = [_ for _ in os.listdir(path) if _.endswith(".png")]
        for i, file in enumerate(files):
            print(f"[{i}/{len(files)}] Processing {file}...")
            try:
                process_img(pjoin(path, file), output_path)
            except Exception as e:
                print(f"  Crashed processing {file}: {e}")
        return
    filename = os.path.basename(path)[:-4]

    # Load image and mask
    img, mask = load_img_mask(path)

    # Extract tumor ellipse contour and its normals
    roi_yx, roi_normals = extract_tumor_ellipse(img, with_normals=True)

    # Find scale if possible
    scale, scale_mask = find_scale(img, mask, return_scale_mask=True)

    # Extract ROIs
    h = 1500 / scale  # ROI height: 1.5mm
    resampling_f = scale / 3  # Resample ROIs to 3um/px
    if 0.9 < resampling_f < 1.2:
        resampling_f = 1

    ROIs = extract_ROIs(roi_yx, roi_normals, mask, roi_h=h)

    print(f" => Scale: {scale:.2f}µm/px, ROI count: {len(ROIs)}")

    # Enhance microscopy and unwarp ROIs
    preprocessed = enhance_microscopy(img, denoise=5)
    unwarped_raws = [roi.unwarp(img, resample=resampling_f) for roi in ROIs]
    unwarped_pres = [roi.unwarp(preprocessed, resample=resampling_f) for roi in ROIs]
    unwarped_masks = [roi.unwarp(mask, resample=resampling_f) for roi in ROIs]

    # Detect vessels in ROIs
    for i, (raw, roi, roi_mask) in enumerate(zip(unwarped_raws, unwarped_pres, unwarped_masks)):
        # Compute the inverted green channel
        igc = 1 - roi[..., 1].astype(float) / 255

        # Detect vessels using MSLD
        vessels_p = msld_detector.multiScaleLineDetector(igc)
        vessels = vessels_p > 0.65

        # Cleanup vessels mask with morphological operations
        vessels_clean = vessels * roi_mask
        vessels_clean = remove_small_objects(vessels_clean, 200)
        vessels_clean = remove_small_holes(vessels_clean, 100)

        # Refine vessels mask using probabilistic Hough transform
        orientation = np.linspace(-60 * DEG2RAD, +60 * DEG2RAD, 60)
        lines = probabilistic_hough_line(vessels_clean, threshold=40, line_length=70, theta=orientation)
        vessels_lines = np.zeros(vessels_clean.shape, dtype=np.uint8)
        for l in lines:
            vessels_lines = cv2.line(vessels_lines, l[0], l[1], color=1, thickness=2)

        # Save images
        cv2.imwrite(pjoin(output_path, "Raw", f"{filename}_{i}.jpg"), raw[::-1])
        cv2.imwrite(pjoin(output_path, "Preprocessed", f"{filename}_{i}.jpg"), roi[::-1])

        igc_rgb = np.ones(roi.shape, dtype=np.uint8)
        igc_rgb *= (igc[..., None] * 255).astype(np.uint8)

        cv2.imwrite(pjoin(output_path, "Vessels", f"{filename}_{i}.png"), vessels_lines[::-1])


def process_img_cmd(args: argparse.Namespace):
    pass


def main():
    args = argparse.ArgumentParser(
        prog="Tumoral Vessel Detector", description="Extract ROI in a tumor image and detect vessels in it"
    )
    args.add_argument("-p", "--path", type=str, help="Path to the image", default="./")
    args.add_argument("-o", "--output", type=str, help="Path to the output folder", default="output")
    args = args.parse_args()

    process_img(args.path, args.output)


if __name__ == "__main__":
    main()
