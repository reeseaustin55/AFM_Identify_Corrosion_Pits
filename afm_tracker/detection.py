"""Pit detection and refinement helpers."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage import measure, morphology
from skimage.filters import gaussian
from skimage.segmentation import active_contour, watershed

from .profile import extract_pit_profile
from .alignment import align_contour_to_gradient
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

    def _roi_halves_for_image(self, image: np.ndarray) -> List[int]:
        """Return ROI half-widths to try for the given image.

        When ``large_pit_mode`` is enabled on the tracker we extend the search
        window to cover substantially larger pits by appending progressively
        larger half-widths up to ~45% of the frame's smaller dimension.  The
        default behaviour is preserved when the toggle is off.
        """

        base = list(self.roi_halves)
        if getattr(self, "large_pit_mode", False):
            max_half = max(base) if base else 0
            limit = int(min(image.shape) * 0.4)
            step = max(30, int(limit * 0.08))
            extra = [half for half in range(max_half + step, limit + 1, step)]
            base.extend(extra)
        # Preserve order but remove duplicates
        seen: List[int] = []
        for half in base:
            if half not in seen:
                seen.append(half)
        return seen

    def detect_pit_edge(self, image: np.ndarray, seed_point, method: int = 0):
        x, y = int(seed_point[0]), int(seed_point[1])
        base_methods = ((method % 3), (method + 1) % 3, (method + 2) % 3)
        method_order = list(base_methods)
        if getattr(self, "large_pit_mode", False):
            # Large pits often benefit from the adaptive component grower, try it first
            method_order = [3] + [m for m in method_order if m != 3]

        for half in self._roi_halves_for_image(image):
            bounds = roi_bounds(x, y, half=half, shape=image.shape)
            roi = image[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]

            for m in method_order:
                if m == 0:
                    comp = self._height_threshold_component(roi, x - bounds.x0, y - bounds.y0)
                elif m == 1:
                    comp = self._watershed_component(roi, x - bounds.x0, y - bounds.y0)
                elif m == 3:
                    comp = self._adaptive_large_component(
                        roi,
                        x - bounds.x0,
                        y - bounds.y0,
                        touches_image_edge=(
                            bounds.x0 == 0
                            or bounds.y0 == 0
                            or bounds.x1 == image.shape[1]
                            or bounds.y1 == image.shape[0]
                        ),
                    )
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
                    refined = self._post_process_contour(
                        contour,
                        image,
                        seed_point=(x, y),
                    )
                    return refined if refined is not None else contour
        return None

    def _current_blur_sigma(self) -> float:
        return float(max(0.0, getattr(self, "detection_blur_sigma", 1.6)))

    def _post_process_contour(self, contour, image, seed_point):
        """Refine ``contour`` to better follow edges while keeping the seed inside."""

        if contour is None or len(contour) < 3:
            return contour

        mask = mask_from_contour(contour, image.shape).astype(bool)
        area = int(mask.sum())
        if area == 0:
            return contour

        refined_mask = self._geodesic_refine_mask(mask, image, seed_point)
        if refined_mask is not None and refined_mask.any():
            new_contour = mask_to_closed_contour(refined_mask.astype(np.uint8))
            if (
                new_contour is not None
                and point_in_contour(new_contour, seed_point[0], seed_point[1], image.shape)
            ):
                contour = new_contour

        aligned = align_contour_to_gradient(
            contour,
            image,
            max_outward=min(40.0, max(image.shape) * 0.08),
            inward=4.0,
            smooth_sigma=1.0,
            distance_penalty=0.6,
        )
        if aligned is not None and len(aligned) >= 3:
            if point_in_contour(aligned, seed_point[0], seed_point[1], image.shape):
                contour = aligned

        pinch_masks = self._pinch_components(
            mask_from_contour(contour, image.shape).astype(bool), image
        )
        finalized = self._components_to_contours(
            pinch_masks,
            image,
            seed_point=seed_point,
            allow_multiple=False,
        )
        if finalized:
            return finalized[0]
        return contour

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
        sigma = self._current_blur_sigma()
        if sigma > 0:
            sm = gaussian(roi, sigma, preserve_range=True)
        else:
            sm = roi
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

    def _adaptive_large_component(
        self,
        roi: np.ndarray,
        x: int,
        y: int,
        touches_image_edge: bool = False,
    ):
        """Grow a smooth component around ``(x, y)`` that can span large pits."""

        if not (0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]):
            return None

        roi_float = roi.astype(np.float32)
        sigma = self._current_blur_sigma()
        if sigma > 0:
            blur = cv2.GaussianBlur(roi_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            blur = roi_float
        norm = cv2.normalize(blur, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        if not np.isfinite(norm).all():
            norm = blur

        seed_val = norm[y, x]
        if not np.isfinite(seed_val):
            return None

        candidate = None
        for offset in np.linspace(0.08, 0.3, 6):
            thr = seed_val + offset
            mask = norm <= thr
            if mask.sum() < 50:
                continue
            mask = morphology.binary_opening(mask, morphology.disk(3))
            mask = morphology.binary_closing(mask, morphology.disk(5))
            mask = morphology.remove_small_holes(mask, area_threshold=256)
            mask = morphology.remove_small_objects(mask, 196)
            comp = self._seed_component(mask, x, y)
            if comp is None or not comp.any():
                continue
            if component_touches_edge(comp):
                if touches_image_edge:
                    continue
                candidate = comp
                continue
            candidate = comp
            break

        if candidate is None or not candidate.any():
            return None

        return candidate

    def _seed_component(self, mask: np.ndarray, x: int, y: int):
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not mask.any():
            return None
        labeled = measure.label(mask)
        if not (0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]):
            return None
        lbl = labeled[y, x]
        if lbl == 0:
            return None
        return labeled == lbl

    def _geodesic_refine_mask(
        self, mask: np.ndarray, image: np.ndarray, seed_point
    ) -> np.ndarray:
        """Quickly clean up ``mask`` in a neighbourhood around the seed point."""

        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not mask.any():
            return mask

        area = int(mask.sum())
        if area < 120:
            return mask

        seed_x = int(round(seed_point[0]))
        seed_y = int(round(seed_point[1]))
        if not (0 <= seed_x < image.shape[1] and 0 <= seed_y < image.shape[0]):
            return mask

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return mask

        y0 = max(0, rows[0] - 6)
        y1 = min(image.shape[0], rows[-1] + 7)
        x0 = max(0, cols[0] - 6)
        x1 = min(image.shape[1], cols[-1] + 7)

        submask = mask[y0:y1, x0:x1]
        subimg = image[y0:y1, x0:x1].astype(np.float32)
        sigma = self._current_blur_sigma()
        if sigma > 0:
            subimg = cv2.GaussianBlur(subimg, (0, 0), sigmaX=sigma, sigmaY=sigma)

        seed_local = (seed_y - y0, seed_x - x0)
        if not (0 <= seed_local[0] < submask.shape[0] and 0 <= seed_local[1] < submask.shape[1]):
            return mask

        seed_val = subimg[seed_local]
        local_min = float(subimg.min())
        local_max = float(subimg.max())
        dynamic = max(1e-3, local_max - local_min)
        threshold = min(seed_val + 0.12 * dynamic, local_max)

        candidate = subimg <= threshold
        candidate = morphology.binary_opening(candidate, morphology.disk(2))
        candidate = morphology.binary_closing(candidate, morphology.disk(3))

        comp = self._seed_component(candidate, seed_local[1], seed_local[0])
        if comp is None or not comp.any():
            comp = submask

        refined = comp | submask
        refined = morphology.binary_closing(refined, morphology.disk(2))
        refined = morphology.binary_opening(refined, morphology.disk(1))
        hole_thresh = max(96, int(area * 0.03))
        refined = morphology.remove_small_holes(refined, area_threshold=hole_thresh)
        refined = morphology.remove_small_objects(refined, min_size=hole_thresh)

        result = np.zeros_like(mask, dtype=bool)
        result[y0:y1, x0:x1] = refined
        return result

    # --- Pinch-off helpers ---------------------------------------------------

    def _score_component(self, component_mask: np.ndarray, image: np.ndarray) -> float:
        if component_mask.dtype != bool:
            component_mask = component_mask.astype(bool)
        if not component_mask.any():
            return float("-inf")
        values = image[component_mask]
        mean_val = float(np.mean(values)) if values.size else 0.0
        area = float(component_mask.sum())
        return -mean_val + 0.03 * np.sqrt(area)

    def _current_pinch_radius(self) -> int:
        """Return the active pinch radius (in pixels)."""

        try:
            value = float(getattr(self, "pinch_distance_px", 3.0))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            value = 3.0
        return max(0, int(round(value)))

    def _pinch_components(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        pinch_radius: Optional[int] = None,
        min_area: int = 120,
    ) -> List[Tuple[np.ndarray, float]]:
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.sum() < min_area:
            return [(mask, self._score_component(mask, image))]

        if pinch_radius is None:
            pinch_radius = self._current_pinch_radius()

        if pinch_radius <= 0:
            return [(mask, self._score_component(mask, image))]

        opened = morphology.binary_opening(mask, morphology.disk(pinch_radius))
        labeled = measure.label(opened)
        if labeled.max() <= 1:
            return [(mask, self._score_component(mask, image))]

        selem = morphology.disk(max(1, pinch_radius))
        components: List[Tuple[np.ndarray, float]] = []
        for lbl in range(1, labeled.max() + 1):
            core = labeled == lbl
            if not core.any():
                continue
            expanded = morphology.binary_dilation(core, selem) & mask
            expanded = morphology.remove_small_objects(expanded, min_size=min_area)
            if not expanded.any():
                continue
            score = self._score_component(expanded, image)
            components.append((expanded, score))

        if not components:
            return [(mask, self._score_component(mask, image))]

        components.sort(key=lambda item: item[1], reverse=True)
        best_score = components[0][1]
        threshold = max(4.0, abs(best_score) * 0.1)
        filtered = [item for item in components if item[1] >= best_score - threshold]
        return filtered if filtered else components[:1]

    def _components_to_contours(
        self,
        components: List[Tuple[np.ndarray, float]],
        image: np.ndarray,
        seed_point=None,
        allow_multiple: bool = False,
    ) -> List[np.ndarray]:
        if not components:
            return []

        entries: List[Tuple[float, np.ndarray]] = []
        seed_entry: Optional[Tuple[float, np.ndarray]] = None
        for comp_mask, score in components:
            contour = mask_to_closed_contour(comp_mask.astype(np.uint8))
            if contour is None or len(contour) < 3:
                continue
            aligned = align_contour_to_gradient(contour, image, smooth_sigma=0.8)
            if aligned is not None and len(aligned) >= 3:
                contour = aligned
            entries.append((score, contour))
            if seed_point is not None and point_in_contour(
                contour, seed_point[0], seed_point[1], image.shape
            ):
                seed_entry = (score, contour)

        if not entries:
            return []

        entries.sort(key=lambda item: item[0], reverse=True)

        if seed_point is not None and not allow_multiple:
            if seed_entry is not None:
                return [seed_entry[1]]
            return [entries[0][1]]

        if not allow_multiple:
            return [entries[0][1]]

        best_score = entries[0][0]
        threshold = max(4.0, abs(best_score) * 0.1)
        return [
            contour
            for score, contour in entries
            if score >= best_score - threshold and len(contour) >= 3
        ]

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
