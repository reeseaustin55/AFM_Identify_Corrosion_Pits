"""Utilities for aligning pit boundaries to image gradients."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .utils import mask_from_contour, mask_to_closed_contour


def _bilinear_interpolate(image: np.ndarray, y: float, x: float) -> float:
    h, w = image.shape
    if y < 0 or x < 0 or y > h - 1 or x > w - 1:
        return 0.0
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    return float(Ia * wa + Ib * wb + Ic * wc + Id * wd)


def align_contour_to_gradient(
    contour: np.ndarray,
    image: np.ndarray,
    max_outward: float = 25.0,
    inward: float = 2.0,
    smooth_sigma: float = 1.5,
) -> Optional[np.ndarray]:
    """Align ``contour`` with the strongest gradient along radial search rays."""
    if contour is None or len(contour) < 3:
        return None

    grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    centroid = np.mean(contour, axis=0)
    search_distances = np.linspace(-inward, max_outward, 48)
    h, w = image.shape
    new_points = []

    for (y, x) in contour:
        direction = np.array([y - centroid[0], x - centroid[1]], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm < 1e-3:
            new_points.append([y, x])
            continue
        direction /= norm

        best_score = -np.inf
        best_point = np.array([y, x], dtype=np.float32)
        for dist in search_distances:
            sy = y + direction[0] * dist
            sx = x + direction[1] * dist
            if sy < 0 or sy >= h or sx < 0 or sx >= w:
                continue
            score = _bilinear_interpolate(grad_mag, sy, sx)
            if score > best_score:
                best_score = score
                best_point = np.array([sy, sx], dtype=np.float32)
        new_points.append(best_point)

    new_contour = np.array(new_points, dtype=np.float32)
    if smooth_sigma > 0:
        ys = gaussian_filter1d(new_contour[:, 0], sigma=smooth_sigma, mode="wrap")
        xs = gaussian_filter1d(new_contour[:, 1], sigma=smooth_sigma, mode="wrap")
        new_contour = np.column_stack([ys, xs])

    mask = mask_from_contour(new_contour, image.shape)
    closed = mask_to_closed_contour(mask)
    return closed if closed is not None else new_contour
