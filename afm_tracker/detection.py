"""Pit detection and refinement helpers."""
from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage import morphology, measure
from skimage.segmentation import watershed, active_contour

from .profile import extract_pit_profile
from .utils import (
    component_touches_edge,
    mask_from_contour,
    mask_to_closed_contour,
    point_in_contour,
    roi_bounds,
)


class DetectionMixin:
    """Provides detection routines for :class:`SmartPitTracker`."""

    roi_halves: Sequence[int] = (80, 120, 160, 220)

    def detect_pit_edge(self, image: np.ndarray, seed_point, method: int = 0):
        x, y = int(seed_point[0]), int(seed_point[1])
        method_order = ((method % 3), (method + 1) % 3, (method + 2) % 3)

        for half in self.roi_halves:
            bounds = roi_bounds(x, y, half=half, shape=image.shape)
            roi = image[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]

            for m in method_order:
                if m == 0:
                    comp = self._height_threshold_component(roi, x - bounds.x0, y - bounds.y0)
                elif m == 1:
                    comp = self._watershed_component(roi, x - bounds.x0, y - bounds.y0)
                else:
                    comp = self._snake_component(roi, x - bounds.x0, y - bounds.y0)

                if comp is None or not comp.any():
                    continue
                if component_touches_edge(comp):
                    continue

                contour_roi = mask_to_closed_contour(comp)
                if contour_roi is None:
                    continue
                contour = contour_roi.copy()
                contour[:, 1] += bounds.x0
                contour[:, 0] += bounds.y0

                if point_in_contour(contour, x, y, image.shape):
                    return contour
        return None

    # --- Component generation -------------------------------------------------
    def _height_threshold_component(self, roi: np.ndarray, x: int, y: int):
        seed_value = roi[y, x]
        hist, bins = np.histogram(roi, bins=64)
        smooth_hist = gaussian_filter1d(hist.astype(float), sigma=2)
        valleys = find_peaks(-smooth_hist, distance=5)[0]

        if len(valleys) > 0:
            seed_bin = np.digitize(seed_value, bins) - 1
            valid_valleys = valleys[valleys > seed_bin]
            if len(valid_valleys) > 0:
                threshold = bins[valid_valleys[0]]
            else:
                threshold = seed_value + 0.10 * (roi.max() - seed_value)
        else:
            threshold = seed_value + 0.12 * (roi.max() - seed_value)

        pit_mask = (roi < threshold).astype(np.uint8)
        pit_mask = morphology.binary_opening(pit_mask, morphology.disk(2))
        pit_mask = morphology.binary_closing(pit_mask, morphology.disk(3))

        labeled = measure.label(pit_mask)
        lbl = labeled[y, x] if 0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1] else 0
        if lbl == 0:
            return None
        comp = labeled == lbl
        comp = morphology.remove_small_holes(comp, area_threshold=64)
        return comp

    def _watershed_component(self, roi: np.ndarray, x: int, y: int):
        den = cv2.bilateralFilter(roi.astype(np.uint8), 9, 75, 75)
        gx = cv2.Sobel(den, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(den, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

        markers = np.zeros_like(roi, dtype=np.int32)
        r = 2
        y0 = max(0, y - r)
        y1 = min(roi.shape[0], y + r + 1)
        x0 = max(0, x - r)
        x1 = min(roi.shape[1], x + r + 1)
        markers[y0:y1, x0:x1] = 1
        yy, xx = np.indices(roi.shape)
        dist = np.hypot(xx - x, yy - y)
        markers[(grad > np.percentile(grad, 80)) & (dist > 15)] = 2

        labels = watershed(grad, markers)
        comp = morphology.binary_closing(labels == 1, morphology.disk(2))
        return comp if comp.any() else None

    def _snake_component(self, roi: np.ndarray, x: int, y: int):
        n_points, radius = 80, 20
        theta = np.linspace(0, 2 * np.pi, n_points)
        ic = np.column_stack([y + radius * np.sin(theta), x + radius * np.cos(theta)])
        from skimage.filters import gaussian

        sm = gaussian(roi, 2, preserve_range=True)
        gx = cv2.Sobel(sm.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sm.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        try:
            snake = active_contour(edge, ic, alpha=0.02, beta=8, gamma=0.01, max_iterations=400)
        except Exception:
            return None
        mask = np.zeros(roi.shape, dtype=np.uint8)
        pts = np.array([[int(p[1]), int(p[0])] for p in snake])
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool) if mask.any() else None

    # --- Similar pit search ---------------------------------------------------
    def _compute_ref_stats(self, reference_profiles: List[dict]):
        ref_stats = {}
        for key in reference_profiles[0].keys():
            values = [p[key] for p in reference_profiles]
            ref_stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)) if len(values) > 1 else float(np.mean(values) * 0.2),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        return ref_stats

    def _candidate_regions(self, image: np.ndarray, ref_stats):
        threshold = ref_stats["pit_bottom"]["mean"] + (
            ref_stats["edge_height"]["mean"] - ref_stats["pit_bottom"]["mean"]
        ) * 0.5
        pit_mask = (image < threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pit_mask = cv2.morphologyEx(pit_mask, cv2.MORPH_CLOSE, kernel)
        pit_mask = cv2.morphologyEx(pit_mask, cv2.MORPH_OPEN, kernel)
        labeled = measure.label(pit_mask)
        return labeled, measure.regionprops(labeled, intensity_image=image)

    def _contour_from_region(self, labeled: np.ndarray, label_id: int):
        comp = labeled == label_id
        return mask_to_closed_contour(comp.astype(np.uint8))

    def find_similar_pits(self, image: np.ndarray, reference_profiles: List[dict]):
        if not reference_profiles:
            return []
        ref_stats = self._compute_ref_stats(reference_profiles)
        labeled, regions = self._candidate_regions(image, ref_stats)

        detected_pits = []
        h, w = image.shape
        edge_margin = 40

        for region in regions:
            cy, cx = region.centroid
            if cx < edge_margin or cx > w - edge_margin or cy < edge_margin or cy > h - edge_margin:
                continue

            contour = self._contour_from_region(labeled, region.label)
            if contour is None:
                continue

            profile = extract_pit_profile(contour, image)
            if not profile:
                continue

            overlap = False
            for ex_id, ex_contour in self.pits.get(self.current_image_idx, {}).items():
                iou_val = self._iou(contour, ex_contour, image.shape)
                if iou_val > 0.15 or self._contains(ex_contour, contour, image.shape) or self._contains(contour, ex_contour, image.shape):
                    overlap = True
                    break
            if overlap:
                continue

            score = 0.0
            weights = 0.0
            area_ratio = profile["area"] / (ref_stats["area"]["mean"] + 1e-6)
            if 0.25 < area_ratio < 4.0:
                score += (1.0 - min(1.0, abs(1.0 - area_ratio))) * 3
                weights += 3
            else:
                continue
            depth_ratio = profile["depth"] / (ref_stats["depth"]["mean"] + 1e-6)
            if 0.3 < depth_ratio < 3.5:
                score += (1.0 - min(1.0, abs(1.0 - depth_ratio) * 0.5)) * 4
                weights += 4
            else:
                continue
            if ref_stats["avg_slope"]["mean"] != 0:
                slope_diff = abs(profile["avg_slope"] - ref_stats["avg_slope"]["mean"])
                score += np.exp(-slope_diff / (abs(ref_stats["avg_slope"]["mean"]) + 0.1)) * 2
                weights += 2
            ecc_diff = abs(profile["eccentricity"] - ref_stats["eccentricity"]["mean"])
            if ecc_diff < 0.35:
                score += (0.35 - ecc_diff) / 0.35
                weights += 1
            ar_diff = abs(profile["aspect_ratio"] - ref_stats["aspect_ratio"]["mean"])
            if ar_diff < 1.2:
                score += (1.2 - ar_diff) / 1.2
                weights += 1

            if weights > 0:
                score /= weights
            if score > 0.4:
                detected_pits.append({"contour": contour, "score": score, "profile": profile})

        detected_pits.sort(key=lambda x: x["score"], reverse=True)
        return detected_pits

    # Proxy methods for mixin compatibility -----------------------------------
    def _iou(self, *args, **kwargs):  # pragma: no cover - forwarded to utils
        return self.iou(*args, **kwargs)

    def _contains(self, *args, **kwargs):  # pragma: no cover - forwarded to utils
        return self.contains(*args, **kwargs)
