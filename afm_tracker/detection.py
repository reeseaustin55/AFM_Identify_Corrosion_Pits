"""Pit detection and refinement helpers."""
from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_fill_holes
from scipy.signal import find_peaks
from skimage import morphology, measure
from skimage.segmentation import (
    watershed,
    active_contour,
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
)

from .alignment import align_contour_to_gradient
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

    roi_halves: Sequence[int] = (80, 120, 160, 220, 280, 360, 460)

    def detect_pit_edge(self, image: np.ndarray, seed_point, method: int = 0):
        x, y = int(seed_point[0]), int(seed_point[1])
        method_order = ((method % 3), (method + 1) % 3, (method + 2) % 3)

        h, w = image.shape
        max_half = int(min(h, w) / 2) - 5
        min_dim = min(h, w)
        dynamic_halves = [int(round(min_dim * frac)) for frac in np.linspace(0.1, 0.48, 8)]
        roi_candidates = []
        for half in (*self.roi_halves, *dynamic_halves):
            if half <= 0:
                continue
            roi_candidates.append(min(max_half, int(half)))
        # Preserve ordering while removing duplicates and ensuring increasing size.
        seen = set()
        candidate_halves = []
        for half in roi_candidates:
            if half <= 20 or half in seen:
                continue
            seen.add(half)
            candidate_halves.append(half)
        if max_half not in seen and max_half > 20:
            candidate_halves.append(max_half)

        for half in candidate_halves:
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
                touches_edge = component_touches_edge(comp)
                can_expand = (
                    bounds.x0 > 0
                    and bounds.x1 < image.shape[1]
                    and bounds.y0 > 0
                    and bounds.y1 < image.shape[0]
                    and half < candidate_halves[-1]
                )
                if touches_edge and can_expand:
                    continue
                if touches_edge and not can_expand:
                    comp = morphology.binary_erosion(comp, morphology.disk(1))
                    if comp is None or not comp.any():
                        continue

                contour_roi = mask_to_closed_contour(comp)
                if contour_roi is None:
                    continue
                contour = contour_roi.copy()
                contour[:, 1] += bounds.x0
                contour[:, 0] += bounds.y0

                contour = self._polish_contour(contour, image)
                if contour is None:
                    continue

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
        scale = max(1, int(round(max(roi.shape) / 90)))
        open_r = max(1, scale)
        close_r = max(open_r + 1, int(round(max(roi.shape) / 60)))
        pit_mask = morphology.binary_opening(pit_mask, morphology.disk(open_r))
        pit_mask = morphology.binary_closing(pit_mask, morphology.disk(close_r))

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
        kernel_radius = max(2, int(round(min(image.shape) / 140)))
        ksize = kernel_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
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

    # --- Large pit refinement helpers ---------------------------------------
    def _polish_contour(self, contour: np.ndarray, image: np.ndarray, keep_mask=None):
        if contour is None or len(contour) < 3:
            return None

        contour = np.asarray(contour, dtype=np.float32)
        h, w = image.shape
        shape = (h, w)

        keep_mask_bool = None
        if keep_mask is not None:
            keep_mask_bool = np.asarray(keep_mask, dtype=bool)
            if keep_mask_bool.shape != shape:
                raise ValueError("keep_mask must match the image shape")
            if not keep_mask_bool.any():
                keep_mask_bool = None

        span_y = float(np.ptp(contour[:, 0])) if len(contour) else 0.0
        span_x = float(np.ptp(contour[:, 1])) if len(contour) else 0.0
        span = max(span_x, span_y, 1.0)

        outward = max(max(shape) * 0.18, span * 0.55)
        inward = max(6.0, span * 0.25)
        smooth_sigma = 1.2 if span < 90 else 1.8

        refined = align_contour_to_gradient(
            contour,
            image,
            max_outward=outward,
            inward=inward,
            smooth_sigma=smooth_sigma,
        )
        if refined is None:
            refined = contour

        refined_mask = mask_from_contour(refined, shape).astype(bool)

        if keep_mask_bool is not None and not np.all(keep_mask_bool <= refined_mask):
            buffer_r = max(2, int(round(span / 35)))
            union = np.logical_or(refined_mask, keep_mask_bool)
            union = morphology.binary_closing(union, morphology.disk(buffer_r))
            union = binary_fill_holes(union)
            union = morphology.binary_opening(union, morphology.disk(max(1, buffer_r // 2)))
            fallback = mask_to_closed_contour(union.astype(np.uint8))
            if fallback is not None and len(fallback) >= 3:
                refined = fallback
                refined_mask = mask_from_contour(refined, shape).astype(bool)

        large_refined = self._refine_large_contour(refined, image)
        if large_refined is not None and len(large_refined) >= 3:
            refined = large_refined
            refined_mask = mask_from_contour(refined, shape).astype(bool)
            if keep_mask_bool is not None and not np.all(keep_mask_bool <= refined_mask):
                union = binary_fill_holes(refined_mask | keep_mask_bool)
                fallback = mask_to_closed_contour(union.astype(np.uint8))
                if fallback is not None and len(fallback) >= 3:
                    refined = fallback
                    refined_mask = mask_from_contour(refined, shape).astype(bool)

        if keep_mask_bool is not None:
            keep_area = int(keep_mask_bool.sum())
            final_area = int(refined_mask.sum())
            if final_area < max(keep_area * 0.75, keep_area - 40):
                return None

        return refined

    def _refine_large_contour(self, contour: np.ndarray, image: np.ndarray):
        if contour is None or len(contour) < 3:
            return contour

        mask = mask_from_contour(contour, image.shape).astype(bool)
        area = int(mask.sum())
        if area == 0:
            return contour

        h, w = image.shape
        area_ratio = area / float(h * w)
        ys = np.any(mask, axis=1)
        xs = np.any(mask, axis=0)
        height = int(ys.sum())
        width = int(xs.sum())
        max_span = max(height, width)
        if area_ratio < 0.04 and max_span < int(min(h, w) * 0.38):
            return contour

        refined_mask = self._geodesic_refine_mask(mask, image)
        if refined_mask is None or not refined_mask.any():
            return contour

        new_contour = mask_to_closed_contour(refined_mask.astype(np.uint8))
        if new_contour is None or len(new_contour) < 3:
            return contour

        return new_contour

    def _geodesic_refine_mask(self, mask: np.ndarray, image: np.ndarray):
        coords = np.column_stack(np.where(mask))
        if coords.size == 0:
            return None

        y0 = max(0, int(coords[:, 0].min()))
        y1 = min(image.shape[0], int(coords[:, 0].max()) + 1)
        x0 = max(0, int(coords[:, 1].min()))
        x1 = min(image.shape[1], int(coords[:, 1].max()) + 1)
        pad = max(10, int(round(max(y1 - y0, x1 - x0) * 0.08)))
        y0 = max(0, y0 - pad)
        y1 = min(image.shape[0], y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(image.shape[1], x1 + pad)

        roi_img = image[y0:y1, x0:x1].astype(np.float32)
        if roi_img.size == 0:
            return None
        roi_mask = mask[y0:y1, x0:x1].astype(bool)
        if not roi_mask.any():
            return None

        roi_min = float(roi_img.min())
        roi_ptp = float(roi_img.max() - roi_min)
        if roi_ptp < 1e-3:
            return None
        roi_norm = (roi_img - roi_min) / (roi_ptp + 1e-6)

        scale = max(roi_norm.shape)
        down_factor = max(1, int(np.ceil(scale / 380)))
        if down_factor > 1:
            small_size = (
                max(1, roi_norm.shape[1] // down_factor),
                max(1, roi_norm.shape[0] // down_factor),
            )
            roi_norm_small = cv2.resize(roi_norm, small_size, interpolation=cv2.INTER_AREA)
            roi_mask_small = cv2.resize(
                roi_mask.astype(np.uint8),
                small_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            roi_norm_small = roi_norm
            roi_mask_small = roi_mask

        alpha = float(np.clip(45 + scale * 0.35, 60, 170))
        sigma = float(np.clip(scale / 220, 0.8, 4.0))
        iterations = int(np.clip(scale * 1.1, 180, 520))
        smoothing = 3 if scale < 240 else 5
        balloon = 1

        gimage = inverse_gaussian_gradient(roi_norm_small, alpha=alpha, sigma=sigma)
        dilate_r = max(2, int(round(max(roi_mask_small.shape) / 90)))
        init_ls = morphology.binary_dilation(roi_mask_small, morphology.disk(dilate_r))
        mgac = morphological_geodesic_active_contour(
            gimage,
            iterations,
            init_level_set=init_ls,
            smoothing=smoothing,
            balloon=balloon,
        )
        mgac = mgac.astype(bool)
        if down_factor > 1:
            mgac = cv2.resize(
                mgac.astype(np.uint8),
                (roi_mask.shape[1], roi_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        close_r = max(2, int(round(scale / 120)))
        open_r = max(1, int(round(scale / 220)))
        mgac = morphology.binary_closing(mgac, morphology.disk(close_r))
        mgac = binary_fill_holes(mgac)
        mgac = morphology.binary_opening(mgac, morphology.disk(open_r))
        mgac = morphology.remove_small_objects(
            mgac,
            min_size=max(32, int(0.01 * roi_mask.sum())),
        )
        if not np.any(mgac):
            return None
        if not np.all(roi_mask <= mgac):
            mgac = np.logical_or(mgac, roi_mask)

        refined = np.zeros_like(mask, dtype=bool)
        refined[y0:y1, x0:x1] = mgac
        return refined
