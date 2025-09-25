"""Utility helpers for working with pit contours and regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class RoiBounds:
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def roi_bounds(x: float, y: float, half: int, shape: Tuple[int, int]) -> RoiBounds:
    """Return the ROI bounds around ``(x, y)`` clipped to ``shape``."""
    h, w = shape
    x0 = max(0, int(x - half))
    x1 = min(w, int(x + half))
    y0 = max(0, int(y - half))
    y1 = min(h, int(y + half))
    return RoiBounds(x0=x0, x1=x1, y0=y0, y1=y1)


def mask_from_contour(contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Create a binary mask for ``contour`` in ``shape``."""
    mask = np.zeros(shape, dtype=np.uint8)
    if contour is None or len(contour) < 3:
        return mask
    pts = np.array([(int(x), int(y)) for y, x in contour], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_closed_contour(mask: np.ndarray) -> np.ndarray | None:
    """Extract the largest closed contour from ``mask`` (Y, X ordering)."""
    if mask is None:
        return None
    m = mask.astype(np.uint8)
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)
    return np.array([[p[0][1], p[0][0]] for p in contour], dtype=np.float32)


def iou(contour_a: np.ndarray, contour_b: np.ndarray, shape: Tuple[int, int]) -> float:
    """Intersection over union between two contours."""
    if contour_a is None or contour_b is None:
        return 0.0
    m1 = mask_from_contour(contour_a, shape)
    m2 = mask_from_contour(contour_b, shape)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union else 0.0


def contains(big: np.ndarray, small: np.ndarray, shape: Tuple[int, int]) -> bool:
    """Return True if ``small`` is completely contained within ``big``."""
    if big is None or small is None:
        return False
    mb = mask_from_contour(big, shape).astype(bool)
    ms = mask_from_contour(small, shape).astype(bool)
    if not ms.any():
        return False
    return bool((ms & ~mb).sum() == 0)


def point_in_contour(contour: np.ndarray, x: float, y: float, shape: Tuple[int, int]) -> bool:
    """Return True if ``(x, y)`` lies inside ``contour``."""
    mask = mask_from_contour(contour, shape)
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or yi >= mask.shape[0] or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def component_touches_edge(component_mask: np.ndarray) -> bool:
    """Return True if the component touches the ROI boundary."""
    if component_mask is None or not component_mask.any():
        return False
    top = component_mask[0, :].any()
    bottom = component_mask[-1, :].any()
    left = component_mask[:, 0].any()
    right = component_mask[:, -1].any()
    return bool(top or bottom or left or right)


def smooth_contour(contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray | None:
    """Project the contour onto a binary mask and re-extract a clean boundary."""
    mask = mask_from_contour(contour, shape)
    return mask_to_closed_contour(mask)
