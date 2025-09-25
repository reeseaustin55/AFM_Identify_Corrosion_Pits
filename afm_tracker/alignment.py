"""Utilities for aligning pit boundaries to image gradients."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .utils import mask_from_contour, mask_to_closed_contour


def _bilinear_interpolate(image: np.ndarray, y: float, x: float) -> float:
    """Sample ``image`` at fractional coordinates using bilinear blending."""
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
    distance_penalty: float = 0.4,
    curvature_weight: float = 0.0,
) -> Optional[np.ndarray]:
    """Align ``contour`` with the strongest gradient along radial search rays."""
    if contour is None or len(contour) < 3:
        return None

    grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    centroid = np.mean(contour, axis=0)
    search_distances = np.linspace(-inward, max_outward, 36)
    h, w = image.shape
    new_points = []

    curvature_weight = max(0.0, float(curvature_weight))

    for (y, x) in contour:
        direction = np.array([y - centroid[0], x - centroid[1]], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm < 1e-3:
            new_points.append([y, x])
            continue
        direction /= norm

        samples = []
        for dist in search_distances:
            sy = y + direction[0] * dist
            sx = x + direction[1] * dist
            if sy < 0 or sy >= h or sx < 0 or sx >= w:
                samples.append((dist, None, None, 0.0))
                continue
            gx = _bilinear_interpolate(grad_x, sy, sx)
            gy = _bilinear_interpolate(grad_y, sy, sx)
            normal_strength = gx * direction[1] + gy * direction[0]
            samples.append((dist, sy, sx, normal_strength if np.isfinite(normal_strength) else 0.0))

        dists = np.array([s[0] for s in samples], dtype=np.float32)
        normal_vals = np.array([s[3] for s in samples], dtype=np.float32)
        if normal_vals.size == 0:
            new_points.append([y, x])
            continue

        # Smooth the response so jagged gradients do not cause large jumps.
        smoothed = gaussian_filter1d(normal_vals, sigma=1.0, mode="nearest")
        origin_idx = int(np.argmin(np.abs(dists)))
        origin_val = smoothed[origin_idx]

        # Penalise distant moves so we stay near the current edge when the
        # gradient strength is comparable.
        penalties = distance_penalty * np.abs(dists)
        scored = smoothed - penalties

        # Search outward (positive distances) for the first prominent peak.
        outward_idx = origin_idx
        outward_vals = scored[origin_idx:]
        if outward_vals.size > 1:
            peaks, _ = find_peaks(outward_vals, prominence=1e-3)
            if peaks.size > 0:
                outward_idx = origin_idx + peaks[0]
            else:
                outward_idx = origin_idx + int(np.argmax(outward_vals))

        # Search inward (negative distances) similarly.
        inward_idx = origin_idx
        inward_vals = scored[: origin_idx + 1]
        if inward_vals.size > 1:
            peaks, _ = find_peaks(inward_vals[::-1], prominence=1e-3)
            if peaks.size > 0:
                inward_idx = origin_idx - peaks[0]
            else:
                inward_idx = int(np.argmax(inward_vals))

        candidate_indices = [origin_idx, outward_idx, inward_idx]
        # Prefer candidates that genuinely improve the directional gradient.
        candidate_scores = []
        for idx in candidate_indices:
            if idx == origin_idx:
                candidate_scores.append(scored[idx])
            elif smoothed[idx] <= origin_val:
                candidate_scores.append(-np.inf)
            else:
                candidate_scores.append(scored[idx])
        best_choice = int(np.argmax(candidate_scores))
        best_idx = int(candidate_indices[best_choice])
        if candidate_scores[best_choice] <= scored[origin_idx] + 1e-3:
            best_idx = origin_idx
        best = samples[best_idx]
        if best[1] is None:
            new_points.append([y, x])
        else:
            new_points.append([best[1], best[2]])

    new_contour = np.array(new_points, dtype=np.float32)
    if curvature_weight > 0.0 and len(new_contour) >= 3:
        prev_pts = np.roll(new_contour, 1, axis=0)
        next_pts = np.roll(new_contour, -1, axis=0)
        laplacian = prev_pts + next_pts - 2.0 * new_contour
        new_contour = new_contour + curvature_weight * laplacian

    if smooth_sigma > 0:
        ys = gaussian_filter1d(new_contour[:, 0], sigma=smooth_sigma, mode="wrap")
        xs = gaussian_filter1d(new_contour[:, 1], sigma=smooth_sigma, mode="wrap")
        new_contour = np.column_stack([ys, xs])

    mask = mask_from_contour(new_contour, image.shape)
    closed = mask_to_closed_contour(mask)
    return closed if closed is not None else new_contour
