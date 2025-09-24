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
    contour = np.asarray(contour, dtype=np.float32)
    span = max(float(np.ptp(contour[:, 0])), float(np.ptp(contour[:, 1])))
    effective_outward = max(max_outward, span * 0.12)
    effective_inward = max(inward, span * 0.04)
    search_distances = np.linspace(-effective_inward, effective_outward, 80)
    h, w = image.shape
    new_points = []

    prev_pts = np.roll(contour, 1, axis=0)
    next_pts = np.roll(contour, -1, axis=0)

    for idx, (y, x) in enumerate(contour):
        tangent = next_pts[idx] - prev_pts[idx]
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-6:
            direction = np.array([y - centroid[0], x - centroid[1]], dtype=np.float32)
            norm = np.linalg.norm(direction)
            if norm < 1e-3:
                new_points.append([y, x])
                continue
            normal = direction / norm
        else:
            tangent /= t_norm
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            radial = np.array([y - centroid[0], x - centroid[1]], dtype=np.float32)
            if np.dot(normal, radial) < 0:
                normal *= -1.0
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            new_points.append([y, x])
            continue
        normal /= normal_norm

        scores = []
        candidates = []
        for dist in search_distances:
            sy = y + normal[0] * dist
            sx = x + normal[1] * dist
            if sy < 0 or sy >= h or sx < 0 or sx >= w:
                scores.append(-np.inf)
                candidates.append(np.array([sy, sx], dtype=np.float32))
                continue
            grad_score = _bilinear_interpolate(grad_mag, sy, sx)
            gx = _bilinear_interpolate(grad_x, sy, sx)
            gy = _bilinear_interpolate(grad_y, sy, sx)
            gvec = np.array([gy, gx], dtype=np.float32)
            gnorm = np.linalg.norm(gvec) + 1e-6
            alignment = max(0.0, float(np.dot(gvec, normal)) / gnorm)
            # Encourage near, well-aligned edges. Penalize large jumps more heavily
            # as the search moves farther away from the original contour.
            penalty = 0.02 * (max(dist, 0.0) ** 2) + 0.01 * (abs(dist) ** 1.5)
            scores.append((grad_score * (0.6 + 0.4 * alignment)) - penalty)
            candidates.append(np.array([sy, sx], dtype=np.float32))

        if not scores:
            new_points.append([y, x])
            continue

        smooth_scores = gaussian_filter1d(np.array(scores, dtype=np.float32), sigma=1.5, mode="reflect")
        outward_mask = search_distances >= 0
        best_idx = int(np.argmax(smooth_scores))
        if outward_mask.any():
            outward_scores = smooth_scores[outward_mask]
            outward_idx = np.argmax(outward_scores)
            # Prefer the best outward candidate unless the global optimum is
            # significantly stronger (prevents jumping across the pit interior).
            if outward_scores[outward_idx] > smooth_scores[best_idx] - 0.15:
                best_idx = int(np.nonzero(outward_mask)[0][outward_idx])
        best_point = candidates[best_idx]
        # Fallback if the search wandered outside the image.
        if not (0 <= best_point[0] < h and 0 <= best_point[1] < w):
            best_point = np.array([y, x], dtype=np.float32)
        new_points.append(best_point)

    new_contour = np.array(new_points, dtype=np.float32)
    if smooth_sigma > 0:
        ys = gaussian_filter1d(new_contour[:, 0], sigma=smooth_sigma, mode="wrap")
        xs = gaussian_filter1d(new_contour[:, 1], sigma=smooth_sigma, mode="wrap")
        new_contour = np.column_stack([ys, xs])

    mask = mask_from_contour(new_contour, image.shape)
    closed = mask_to_closed_contour(mask)
    return closed if closed is not None else new_contour
