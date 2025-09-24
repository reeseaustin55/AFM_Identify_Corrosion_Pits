"""Feature extraction utilities for pit characterization."""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
from skimage import measure, morphology

from .utils import mask_from_contour


def extract_pit_profile(contour: np.ndarray, image: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute geometric and intensity-based descriptors for ``contour``."""
    mask = mask_from_contour(contour, image.shape)
    if not mask.any():
        return None

    labeled = measure.label(mask)
    regions = measure.regionprops(labeled, intensity_image=image)
    if not regions:
        return None
    region = regions[0]

    edge_buffer = 5
    dilated_mask = cv2.dilate(mask, np.ones((edge_buffer * 2, edge_buffer * 2)), iterations=1)
    edge_mask = cv2.subtract(dilated_mask, mask)

    pit_heights = image[mask > 0] if np.any(mask > 0) else np.array([0])
    edge_heights = image[edge_mask > 0] if np.any(edge_mask > 0) else np.array([0])

    pit_bottom = np.percentile(pit_heights, 10) if len(pit_heights) else 0.0
    edge_top = np.percentile(edge_heights, 90) if len(edge_heights) else 0.0
    depth = float(edge_top - pit_bottom)

    contour_points = contour[::5]
    slopes = []
    for i in range(len(contour_points)):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % len(contour_points)]
        direction = np.array([p2[1] - p1[1], p2[0] - p1[0]], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction /= norm
        perpendicular = np.array([-direction[1], direction[0]])
        inside_point = p1 - perpendicular * 3
        outside_point = p1 + perpendicular * 3
        y_in, x_in = int(round(inside_point[0])), int(round(inside_point[1]))
        y_out, x_out = int(round(outside_point[0])), int(round(outside_point[1]))
        if (0 <= y_in < image.shape[0] and 0 <= x_in < image.shape[1] and
                0 <= y_out < image.shape[0] and 0 <= x_out < image.shape[1]):
            slope = (float(image[y_out, x_out]) - float(image[y_in, x_in])) / 6.0
            slopes.append(slope)

    avg_slope = float(np.mean(slopes)) if slopes else 0.0

    profile = {
        "area": float(region.area),
        "perimeter": float(cv2.arcLength(np.array([(int(x), int(y)) for y, x in contour], dtype=np.int32), True)),
        "depth": depth,
        "pit_bottom": float(pit_bottom),
        "edge_height": float(edge_top),
        "avg_slope": avg_slope,
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "equivalent_diameter": float(region.equivalent_diameter),
        "aspect_ratio": float(region.major_axis_length / (region.minor_axis_length + 1e-6)),
        "mean_intensity": float(region.mean_intensity),
        "intensity_std": float(np.std(pit_heights) if len(pit_heights) else 0.0),
    }
    return profile
