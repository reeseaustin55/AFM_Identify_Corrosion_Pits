"""Batch processing utilities for quantifying AFM pit growth."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from afm_tracker.loader import load_image_series
from afm_tracker.profile import extract_pit_profile
from afm_tracker.utils import iou, mask_to_closed_contour


@dataclass
class PitCandidate:
    """Light-weight container for pit measurements within a frame."""

    contour: np.ndarray
    centroid: Tuple[float, float]
    profile: Dict[str, float]


@dataclass
class PitObservation:
    """Persistent measurements for a tracked pit instance."""

    pit_id: int
    frame_index: int
    image_name: str
    timestamp: Optional[datetime]
    time_seconds: Optional[float]
    area_px: float
    area_nm2: Optional[float]
    depth: float
    mean_intensity: float
    centroid_x: float
    centroid_y: float
    equivalent_diameter_px: float
    perimeter_px: float
    solidity: float
    aspect_ratio: float

    def to_record(self) -> Dict[str, object]:
        ts_iso = self.timestamp.isoformat() if self.timestamp is not None else None
        return {
            "pit_id": self.pit_id,
            "frame_index": self.frame_index,
            "image_name": self.image_name,
            "timestamp_iso": ts_iso,
            "time_seconds": self.time_seconds,
            "area_px": self.area_px,
            "area_nm2": self.area_nm2,
            "depth": self.depth,
            "mean_intensity": self.mean_intensity,
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "equivalent_diameter_px": self.equivalent_diameter_px,
            "perimeter_px": self.perimeter_px,
            "solidity": self.solidity,
            "aspect_ratio": self.aspect_ratio,
        }


class PitGrowthAnalyzer:
    """Utility for extracting pit growth metrics from AFM image sequences."""

    def __init__(
        self,
        min_area_pixels: int = 60,
        min_area_nm2: Optional[float] = None,
        max_match_distance_px: float = 40.0,
        iou_threshold: float = 0.08,
    ) -> None:
        self.min_area_pixels = max(3, int(min_area_pixels))
        self.min_area_nm2 = min_area_nm2
        self.max_match_distance_px = max_match_distance_px
        self.iou_threshold = iou_threshold

        self._active_tracks: Dict[int, PitCandidate] = {}
        self._next_track_id: int = 0
        self._observations: List[PitObservation] = []
        self._nm_per_pixel: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image_folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process ``image_folder`` and return per-frame and summary tables."""

        (
            images,
            image_files,
            timestamps,
            nm_per_pixel,
        ) = load_image_series(Path(image_folder))
        if not images:
            raise FileNotFoundError(f"No PNG images found in {image_folder}")

        self._nm_per_pixel = nm_per_pixel
        min_area_px = self._resolve_min_area_pixels()
        print(
            f"Resolved minimum pit area: {min_area_px} pixels"
            + (" (from physical threshold)" if self.min_area_nm2 else "")
        )

        start_time = timestamps[0] if timestamps else None

        output_dir = Path.cwd() / "pit_growth_outputs"
        output_dir.mkdir(exist_ok=True)

        for frame_idx, (image, img_path) in enumerate(zip(images, image_files)):
            timestamp = timestamps[frame_idx] if frame_idx < len(timestamps) else None
            time_seconds = None
            if start_time is not None and timestamp is not None:
                time_seconds = (timestamp - start_time).total_seconds()

            candidates = self._segment_frame(image, min_area_px)
            assignments = self._associate_tracks(candidates, image.shape)

            for pit_id, candidate in assignments.items():
                profile = candidate.profile
                area_px = float(profile.get("area", 0.0))
                area_nm2 = (
                    float(area_px * (nm_per_pixel ** 2))
                    if nm_per_pixel is not None
                    else None
                )
                observation = PitObservation(
                    pit_id=pit_id,
                    frame_index=frame_idx,
                    image_name=img_path.name,
                    timestamp=timestamp,
                    time_seconds=time_seconds,
                    area_px=area_px,
                    area_nm2=area_nm2,
                    depth=float(profile.get("depth", 0.0)),
                    mean_intensity=float(profile.get("mean_intensity", 0.0)),
                    centroid_x=float(candidate.centroid[0]),
                    centroid_y=float(candidate.centroid[1]),
                    equivalent_diameter_px=float(profile.get("equivalent_diameter", 0.0)),
                    perimeter_px=float(profile.get("perimeter", 0.0)),
                    solidity=float(profile.get("solidity", 0.0)),
                    aspect_ratio=float(profile.get("aspect_ratio", 0.0)),
                )
                self._observations.append(observation)

            overlay_path = output_dir / f"frame_{frame_idx:03d}_overlay.png"
            self._write_overlay(image, assignments, overlay_path)

        per_frame_df = pd.DataFrame([obs.to_record() for obs in self._observations])
        summary_df = self._summarize_growth(per_frame_df)

        per_frame_csv = output_dir / "pit_observations.csv"
        summary_csv = output_dir / "pit_growth_summary.csv"
        summary_excel = output_dir / "pit_growth_summary.xlsx"
        per_frame_df.to_csv(per_frame_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        summary_df.to_excel(summary_excel, index=False)

        print(f"Saved per-frame observations to {per_frame_csv}")
        print(f"Saved pit growth summary to {summary_csv}")
        print(f"Saved corrosion rate workbook to {summary_excel}")

        return per_frame_df, summary_df

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    def _segment_frame(self, image: np.ndarray, min_area_px: int) -> List[PitCandidate]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        equalized = clahe.apply(image)

        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        enhanced = cv2.morphologyEx(equalized, cv2.MORPH_BLACKHAT, blackhat_kernel)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        _, thresh = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        candidates: List[PitCandidate] = []
        for label_idx in range(1, num_labels):
            area_px = stats[label_idx, cv2.CC_STAT_AREA]
            if area_px < min_area_px:
                continue

            x0 = stats[label_idx, cv2.CC_STAT_LEFT]
            y0 = stats[label_idx, cv2.CC_STAT_TOP]
            width = stats[label_idx, cv2.CC_STAT_WIDTH]
            height = stats[label_idx, cv2.CC_STAT_HEIGHT]

            component_mask = (labels[y0 : y0 + height, x0 : x0 + width] == label_idx).astype(
                np.uint8
            )
            contour_local = mask_to_closed_contour(component_mask)
            if contour_local is None:
                continue

            contour = contour_local.copy()
            contour[:, 0] += y0
            contour[:, 1] += x0

            profile = extract_pit_profile(contour, image)
            if not profile:
                continue

            centroid = tuple(centroids[label_idx])
            candidates.append(PitCandidate(contour=contour, centroid=centroid, profile=profile))

        return candidates

    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------
    def _associate_tracks(
        self, candidates: Iterable[PitCandidate], image_shape: Tuple[int, int]
    ) -> Dict[int, PitCandidate]:
        candidates = list(candidates)
        if not self._active_tracks:
            assignments = {
                self._spawn_track(): candidate for candidate in candidates
            }
            self._active_tracks = assignments.copy()
            return assignments

        prev_ids = list(self._active_tracks.keys())
        cost = np.zeros((len(prev_ids), len(candidates)), dtype=np.float32)
        diag = float(np.hypot(image_shape[0], image_shape[1]))

        for i, pid in enumerate(prev_ids):
            prev_candidate = self._active_tracks[pid]
            prev_centroid = np.array(prev_candidate.centroid)
            for j, candidate in enumerate(candidates):
                curr_centroid = np.array(candidate.centroid)
                dist = np.linalg.norm(curr_centroid - prev_centroid)
                iou_val = iou(prev_candidate.contour, candidate.contour, image_shape)
                cost[i, j] = (1.0 - float(iou_val)) + 0.1 * float(dist / diag)

        row_ind, col_ind = linear_sum_assignment(cost) if candidates else ([], [])

        assignments: Dict[int, PitCandidate] = {}
        matched_candidates = set()
        for r, c in zip(row_ind, col_ind):
            pid = prev_ids[r]
            candidate = candidates[c]
            curr_centroid = np.array(candidate.centroid)
            prev_centroid = np.array(self._active_tracks[pid].centroid)
            dist = np.linalg.norm(curr_centroid - prev_centroid)
            overlap = iou(self._active_tracks[pid].contour, candidate.contour, image_shape)
            if overlap >= self.iou_threshold or dist <= self.max_match_distance_px:
                assignments[pid] = candidate
                matched_candidates.add(c)

        for idx, candidate in enumerate(candidates):
            if idx in matched_candidates:
                continue
            pid = self._spawn_track()
            assignments[pid] = candidate

        self._active_tracks = assignments.copy()
        return assignments

    def _spawn_track(self) -> int:
        pid = self._next_track_id
        self._next_track_id += 1
        return pid

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _resolve_min_area_pixels(self) -> int:
        if self.min_area_nm2 is None or self._nm_per_pixel is None:
            return self.min_area_pixels
        area_px = int(round(self.min_area_nm2 / (self._nm_per_pixel ** 2)))
        return max(self.min_area_pixels, area_px)

    def _write_overlay(
        self,
        image: np.ndarray,
        assignments: Dict[int, PitCandidate],
        output_path: Path,
    ) -> None:
        overlay = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for pit_id, candidate in assignments.items():
            contour = candidate.contour.astype(np.int32)
            pts = np.column_stack([contour[:, 1], contour[:, 0]])
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
            cx, cy = int(round(candidate.centroid[0])), int(round(candidate.centroid[1]))
            cv2.putText(
                overlay,
                str(pit_id),
                (cx + 3, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(output_path), overlay)

    def _summarize_growth(self, per_frame_df: pd.DataFrame) -> pd.DataFrame:
        nm_per_pixel = self._nm_per_pixel
        conversion_factor = (
            float(nm_per_pixel**2) if nm_per_pixel is not None else None
        )

        if per_frame_df.empty:
            return pd.DataFrame(
                columns=[
                    "pit_id",
                    "frames_observed",
                    "start_frame",
                    "end_frame",
                    "initial_area_px",
                    "young_area_px",
                    "initial_area_nm2",
                    "final_area_nm2",
                    "initial_perimeter_nm",
                    "final_perimeter_nm",
                    "conversion_factor_px_to_nm2",
                    "net_growth_nm2",
                    "average_perimeter_nm",
                    "corrosion_nm",
                    "duration_seconds",
                    "corrosion_rate_nm_per_s",
                    "corrosion_rate_nm_per_min",
                    "max_depth",
                ]
            )

        def _safe_float(value: object) -> Optional[float]:
            if value is None or pd.isna(value):
                return None
            if isinstance(value, (int, np.integer)):
                return float(value)
            if isinstance(value, (float, np.floating)):
                return float(value)
            return float(value)

        grouped = per_frame_df.groupby("pit_id")
        records: List[Dict[str, object]] = []

        for pit_id, group in grouped:
            group = group.sort_values("frame_index")
            frames_observed = len(group)
            start_frame = int(group.iloc[0]["frame_index"])
            end_frame = int(group.iloc[-1]["frame_index"])

            initial_area_px = _safe_float(group.iloc[0]["area_px"])
            young_area_px = _safe_float(group.iloc[-1]["area_px"])
            initial_area_nm2 = _safe_float(group.iloc[0]["area_nm2"])
            final_area_nm2 = _safe_float(group.iloc[-1]["area_nm2"])

            initial_perimeter_px = _safe_float(group.iloc[0]["perimeter_px"])
            final_perimeter_px = _safe_float(group.iloc[-1]["perimeter_px"])

            initial_perimeter_nm = (
                float(initial_perimeter_px * nm_per_pixel)
                if nm_per_pixel is not None and initial_perimeter_px is not None
                else None
            )
            final_perimeter_nm = (
                float(final_perimeter_px * nm_per_pixel)
                if nm_per_pixel is not None and final_perimeter_px is not None
                else None
            )

            net_growth_nm2 = (
                float(final_area_nm2 - initial_area_nm2)
                if initial_area_nm2 is not None and final_area_nm2 is not None
                else None
            )

            average_perimeter_nm = (
                float((initial_perimeter_nm + final_perimeter_nm) / 2.0)
                if initial_perimeter_nm is not None and final_perimeter_nm is not None
                else None
            )

            corrosion_nm = (
                float(net_growth_nm2 / average_perimeter_nm)
                if net_growth_nm2 is not None
                and average_perimeter_nm not in (None, 0.0)
                else None
            )

            time_start = _safe_float(group.iloc[0]["time_seconds"])
            time_end = _safe_float(group.iloc[-1]["time_seconds"])
            duration_seconds = (
                float(time_end - time_start)
                if time_start is not None and time_end is not None
                else None
            )

            corrosion_rate_nm_per_s = (
                float(corrosion_nm / duration_seconds)
                if corrosion_nm is not None
                and duration_seconds not in (None, 0.0)
                else None
            )
            corrosion_rate_nm_per_min = (
                float(corrosion_rate_nm_per_s * 60.0)
                if corrosion_rate_nm_per_s is not None
                else None
            )

            max_depth = float(group["depth"].max())

            records.append(
                {
                    "pit_id": pit_id,
                    "frames_observed": frames_observed,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "initial_area_px": initial_area_px,
                    "young_area_px": young_area_px,
                    "initial_area_nm2": initial_area_nm2,
                    "final_area_nm2": final_area_nm2,
                    "initial_perimeter_nm": initial_perimeter_nm,
                    "final_perimeter_nm": final_perimeter_nm,
                    "conversion_factor_px_to_nm2": conversion_factor,
                    "net_growth_nm2": net_growth_nm2,
                    "average_perimeter_nm": average_perimeter_nm,
                    "corrosion_nm": corrosion_nm,
                    "duration_seconds": duration_seconds,
                    "corrosion_rate_nm_per_s": corrosion_rate_nm_per_s,
                    "corrosion_rate_nm_per_min": corrosion_rate_nm_per_min,
                    "max_depth": max_depth,
                }
            )

        return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify AFM corrosion pit growth over time.")
    parser.add_argument("image_folder", type=Path, help="Folder containing sequential AFM PNG images")
    parser.add_argument(
        "--min-area-nm2",
        type=float,
        default=None,
        help="Optional physical area threshold (nm^2) for filtering pits.",
    )
    parser.add_argument(
        "--min-area-px",
        type=int,
        default=60,
        help="Minimum connected-component area in pixels when physical scale is unknown.",
    )
    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=40.0,
        help="Maximum centroid displacement (pixels) allowed when tracking pits across frames.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.08,
        help="Minimum IoU required to keep an existing track association.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = PitGrowthAnalyzer(
        min_area_pixels=args.min_area_px,
        min_area_nm2=args.min_area_nm2,
        max_match_distance_px=args.max_match_distance,
        iou_threshold=args.iou_threshold,
    )
    analyzer.analyze(args.image_folder)


if __name__ == "__main__":
    main()
