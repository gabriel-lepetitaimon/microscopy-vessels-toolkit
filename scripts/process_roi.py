__all__ = ["process_img"]

import argparse
import os
from datetime import datetime
from os.path import join as pjoin
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.transform import probabilistic_hough_line

from microscopy_vessels_toolkit.analysis.vessel_density import label_1d
from microscopy_vessels_toolkit.preprocess import enhance_microscopy
from microscopy_vessels_toolkit.preprocess.roi import (
    AnnotationNotFound,
    extract_ROIs,
    extract_tumor_ellipse,
    find_scale,
    load_img_mask,
)
from microscopy_vessels_toolkit.segment.MSLD import MSLD
from microscopy_vessels_toolkit.utilities.color import catpuccin_colors

DEG2RAD = np.pi / 180
orientation = np.linspace(-50 * DEG2RAD, +50 * DEG2RAD, 15)
msld_detector = MSLD(33, [17, 21, 25, 27, 31], orientation)


def process_img(
    path: str,
    output_path: str,
    debug_img: bool = False,
    vessels_stats: list[Dict] | None = None,
):
    output_folders = ("Preprocessed", "Vessels", "RefinedVessels")
    if debug_img:
        output_folders = output_folders + ("debug",)
    for folder_name in output_folders:
        folder_path = pjoin(output_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    if os.path.isdir(path):
        files = [_ for _ in os.listdir(path) if _.endswith(".png")]
        vessels_stats = []
        for i, file in enumerate(files):
            print(f"[{i}/{len(files)}] Processing {file}...")
            try:
                process_img(
                    pjoin(path, file),
                    output_path,
                    debug_img=debug_img,
                    vessels_stats=vessels_stats,
                )
            except AnnotationNotFound as e:
                print(f"  Crashed processing {file}: {e}")
        # Export vessels stats
        vessels_stats = pd.DataFrame(vessels_stats)
        vessels_stats.to_csv(pjoin(output_path, "vessels_stats.csv"))
        return
    filename = os.path.basename(path)[:-4]

    # Load image and mask
    img, mask = load_img_mask(path)
    debug_img = img.copy() if debug_img else None

    t_start = datetime.now()
    # Extract tumor ellipse contour and its normals
    roi_yx, roi_normals = extract_tumor_ellipse(img, with_normals=True)

    if debug_img is not None:
        debug_img[roi_yx[..., 0], roi_yx[..., 1]] = [255, 255, 255]

    # Find scale if possible
    scale, scale_mask = find_scale(img, mask, return_scale_mask=True)

    t_find_annotations = datetime.now() - t_start

    if debug_img is not None:
        scale_y, scale_x = np.where(scale_mask)
        debug_img[scale_y, scale_x] = [255, 0, 255]
        label_y, label_x = np.max([scale_y, scale_x], axis=1)
        cv2.putText(
            debug_img,
            f"Scale: {scale:.2f}um/px",
            (label_x + 15, label_y + 5),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 0, 255),
            1,
        )

    # Extract ROIs
    t = datetime.now()
    h = 1500 / scale  # ROI height: 1.5mm
    resampled_scale = 3
    resampling_f = scale / resampled_scale  # Resample ROIs to 3um/px
    if 0.9 < resampling_f < 1.2:
        resampling_f = 1
        resampled_scale = scale

    ROIs = extract_ROIs(roi_yx, roi_normals, mask, roi_h=h)
    print(f" => Scale: {scale:.2f}µm/px, ROI count: {len(ROIs)}")

    if debug_img is not None:
        for i, roi in enumerate(ROIs):
            roi.draw_polygon(
                debug_img,
                color=catpuccin_colors[i % len(catpuccin_colors)],
                thickness=4,
                label=f"{i+1}",
            )

    # Enhance microscopy and unwarp ROIs
    t1 = datetime.now()
    preprocessed = enhance_microscopy(img, mask, denoise=5)
    t_preprocess = datetime.now() - t1
    unwarped_ROIs = [roi.unwarp(preprocessed, resample=resampling_f) for roi in ROIs]
    unwarped_masks = [roi.unwarp(mask, resample=resampling_f) for roi in ROIs]
    t_unwarp_ROI = datetime.now() - t

    # Detect vessels in ROIs
    for i, (roi, roi_mask) in enumerate(zip(unwarped_ROIs, unwarped_masks, strict=True)):
        t = datetime.now()
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
        vessels_enhanced = np.zeros(vessels_clean.shape, dtype=np.uint8)
        for line in lines:
            vessels_enhanced = cv2.line(vessels_enhanced, line[0], line[1], color=1, thickness=2)

        t_detect_vessels = datetime.now() - t

        # Save images
        cv2.imwrite(pjoin(output_path, "Preprocessed", f"{filename}_{i}.png"), roi[::-1])

        igc_rgb = np.ones(roi.shape, dtype=np.uint8)
        igc_rgb *= (igc[..., None] * 255).astype(np.uint8)

        vessels_output = np.where(vessels_clean[:, :, None], [[[28, 25, 197]]], igc_rgb)
        cv2.imwrite(pjoin(output_path, "Vessels", f"{filename}_{i}.png"), vessels_output[::-1])

        vessels_output = np.where(vessels_enhanced[:, :, None], [[[28, 25, 197]]], igc_rgb)
        cv2.imwrite(
            pjoin(output_path, "RefinedVessels", f"{filename}_{i}.png"),
            vessels_output[::-1],
        )

        if i == 0 and vessels_stats is not None:
            # Compute vessel density
            w = vessels_enhanced.shape[1]
            width_mm = w * resampled_scale / 1000
            lines_row = np.linspace(0, vessels_enhanced.shape[0], 25 + 2, dtype=int)[1:-1]
            lines_nb_vessels = [label_1d(vessels_enhanced[row, :]).max() for row in lines_row]

            # print(" => Vessel count per line:", nb_vessels_per_line)
            vessel_med_density = np.median(lines_nb_vessels) / width_mm

            print(f" => Vessel density: {vessel_med_density:.1f} vessels/mm")
            vessels_stats.append(
                {
                    "file": filename,
                    "Scale (µm/px)": scale,
                    "Vessel Density (vessels/mm)": vessel_med_density,
                    "Total Duration (s)": (datetime.now() - t_start).total_seconds(),
                    "Find Annotations (s)": t_find_annotations.total_seconds(),
                    "Unwarp ROIs (s)": t_unwarp_ROI.total_seconds(),
                    "... dont preprocessing (s)": t_preprocess.total_seconds(),
                    "Detect Vessels (s)": t_detect_vessels.total_seconds(),
                }
            )

    if debug_img is not None:
        cv2.imwrite(pjoin(output_path, "debug", f"{filename}.png"), debug_img)


def argmedian(x):
    i = np.argpartition(x, len(x) // 2)[len(x) // 2]
    return i, x[i]


def main():
    args = argparse.ArgumentParser(
        prog="Tumoral Vessel Detector",
        description="Extract ROI in a tumor image and detect vessels in it",
    )
    args.add_argument("-p", "--path", type=str, help="Path to the image", default="./")
    args.add_argument("-o", "--output", type=str, help="Path to the output folder", default="output")
    args.add_argument("-d", "--debug", action="store_true", help="Save debug images")
    args = args.parse_args()

    process_img(args.path, args.output, debug_img=args.debug)


if __name__ == "__main__":
    main()
