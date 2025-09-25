from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, LassoSelector, Slider
from skimage import morphology

from .alignment import align_contour_to_gradient
from .detection import DetectionMixin
from .loader import load_image_series
from .profile import extract_pit_profile
from .tracking import TrackingMixin
from .utils import (
    contains,
    iou,
    mask_from_contour,
    mask_to_closed_contour,
    point_in_contour,
)

warnings.filterwarnings("ignore")
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


@dataclass
class InteractiveState:
    lasso: Optional[LassoSelector] = None
    circle_press: Optional[tuple] = None
    circle_artist: Optional[Line2D] = None
    manual_active: bool = False
    manual_pts: Optional[List[tuple]] = None
    cid_press: Optional[int] = None
    cid_release: Optional[int] = None
    cid_motion: Optional[int] = None


class SmartPitTracker(DetectionMixin, TrackingMixin):
    def __init__(self, image_folder: Optional[str] = None):
        if image_folder is None:
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                image_folder = filedialog.askdirectory(title="Select folder containing AFM images")
                root.destroy()
                if not image_folder:
                    print("No folder selected. Exiting.")
                    self.images = []
                    return
            except Exception:
                print("Please provide a folder path containing the AFM images")
                self.images = []
                return

        self.image_folder = Path(image_folder)
        (
            self.images,
            self.image_files,
            self.timestamps,
            self.nm_per_pixel,
        ) = load_image_series(self.image_folder)

        if not self.images:
            print(f"No PNG files found in {self.image_folder}")
            return

        if self.nm_per_pixel is None:
            self.nm_per_pixel = 250.0 / 1024.0
            print(f"Using default scale: {self.nm_per_pixel:.3f} nm/pixel")

        print(f"Loaded {len(self.images)} images")
        self.pits: Dict[int, Dict[int, np.ndarray]] = {}
        self.pit_profiles: Dict[int, dict] = {}
        self.pit_lineage: Dict[int, Dict[int, List[int]]] = {}
        self.current_image_idx = 0
        self.selected_pit_ids: List[int] = []
        self.next_pit_id = 0
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.mode = "select"
        self.trace_method = 0
        self.state = InteractiveState(manual_pts=[])
        self.large_pit_mode = False
        self.detection_blur_sigma = 1.6

    # ------------------------------------------------------------------
    # Basic geometry helpers (wrappers around utils)
    # ------------------------------------------------------------------
    def iou(self, contour_a: np.ndarray, contour_b: np.ndarray, shape) -> float:
        return iou(contour_a, contour_b, shape)

    def contains(self, contour_big: np.ndarray, contour_small: np.ndarray, shape) -> bool:
        return contains(contour_big, contour_small, shape)

    def point_in_contour(self, contour: np.ndarray, x: float, y: float, shape) -> bool:
        return point_in_contour(contour, x, y, shape)

    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------
    def run_interactive_gui(self):
        if not self.images:
            print("No images to display!")
            return

        print("\n" + "=" * 70)
        print("AFM PIT TRACKING - Manual, Smart, and Interactive Tools")
        print("=" * 70)

        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.38)

        # Buttons row 1
        ax_add = plt.axes([0.06, 0.22, 0.10, 0.05])
        ax_refit = plt.axes([0.17, 0.22, 0.10, 0.05])
        ax_delete = plt.axes([0.28, 0.22, 0.10, 0.05])
        ax_auto = plt.axes([0.39, 0.22, 0.12, 0.05])
        ax_clear = plt.axes([0.52, 0.22, 0.10, 0.05])
        ax_confirm = plt.axes([0.63, 0.22, 0.10, 0.05])

        # Buttons row 2
        ax_prev = plt.axes([0.06, 0.14, 0.10, 0.05])
        ax_next = plt.axes([0.17, 0.14, 0.10, 0.05])
        ax_process = plt.axes([0.28, 0.14, 0.12, 0.05])
        ax_export = plt.axes([0.41, 0.14, 0.10, 0.05])
        ax_lasso = plt.axes([0.52, 0.14, 0.10, 0.05])
        ax_circle = plt.axes([0.63, 0.14, 0.10, 0.05])
        ax_manual = plt.axes([0.74, 0.14, 0.12, 0.05])

        # Buttons row 3 (advanced tools)
        ax_align = plt.axes([0.06, 0.08, 0.12, 0.05])
        ax_select_all = plt.axes([0.19, 0.08, 0.12, 0.05])
        ax_large_mode = plt.axes([0.32, 0.08, 0.18, 0.05])
        ax_sigma = plt.axes([0.55, 0.02, 0.35, 0.03])

        self.btn_add = Button(ax_add, "Add Mode")
        self.btn_refit = Button(ax_refit, "Refit+")
        self.btn_delete = Button(ax_delete, "Delete")
        self.btn_auto = Button(ax_auto, "Auto Similar")
        self.btn_clear = Button(ax_clear, "Clear Sel")
        self.btn_confirm = Button(ax_confirm, "Confirm")

        self.btn_prev = Button(ax_prev, "\u2190 Prev")
        self.btn_next = Button(ax_next, "Next \u2192")
        self.btn_process = Button(ax_process, "Process All")
        self.btn_export = Button(ax_export, "Export")
        self.btn_lasso = Button(ax_lasso, "Lasso")
        self.btn_circle = Button(ax_circle, "Circle ROI")
        self.btn_manual = Button(ax_manual, "Manual Trace")
        self.btn_align = Button(ax_align, "Align Edge")
        self.btn_select_all = Button(ax_select_all, "Select All")
        self.btn_large_mode = Button(ax_large_mode, "Large Fit: Off")
        self.blur_slider = Slider(
            ax_sigma,
            "Blur σ",
            valmin=0.0,
            valmax=5.0,
            valinit=getattr(self, "detection_blur_sigma", 1.6),
            valstep=0.1,
        )

        self.btn_add.on_clicked(self.toggle_add_mode)
        self.btn_refit.on_clicked(self.refit_selected_robust)
        self.btn_delete.on_clicked(self.delete_selected)
        self.btn_auto.on_clicked(self.auto_detect_similar)
        self.btn_clear.on_clicked(self.clear_selection)
        self.btn_confirm.on_clicked(self.confirm_frame)
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_process.on_clicked(self.process_all_frames)
        self.btn_export.on_clicked(self.export_results)
        self.btn_lasso.on_clicked(self.activate_lasso)
        self.btn_circle.on_clicked(self.activate_circle)
        self.btn_manual.on_clicked(self.activate_manual)
        self.btn_align.on_clicked(self.align_selected_edges)
        self.btn_select_all.on_clicked(self.select_all_pits)
        self.btn_large_mode.on_clicked(self.toggle_large_fit_mode)
        self.blur_slider.on_changed(self._on_blur_sigma_change)

        self.display_current_image()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        print(
            "\nINSTRUCTIONS (tools): Lasso/Circle/Manual create pits from your input. "
            "Auto Similar scans for more like your seeds. Align Edge snaps the boundary to the strongest nearby slope. "
            "Toggle Large Fit when working with bigger pits."
        )
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def display_current_image(self):
        if self.ax is None:
            return
        self.ax.clear()
        self.ax.imshow(self.images[self.current_image_idx], cmap="hot")

        if self.current_image_idx in self.pits:
            for pit_id, contour in self.pits[self.current_image_idx].items():
                color = "yellow" if pit_id in self.selected_pit_ids else "cyan"
                lw = 3 if pit_id in self.selected_pit_ids else 2
                self.ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=lw)

        self.ax.set_xlim(0, self.images[0].shape[1])
        self.ax.set_ylim(self.images[0].shape[0], 0)

        info_text = f"Pits: {len(self.pits.get(self.current_image_idx, {}))}"
        if self.selected_pit_ids:
            info_text += f" | Selected: {len(self.selected_pit_ids)}"
        if self.large_pit_mode:
            info_text += " | Large Fit"
        self.ax.text(
            10,
            30,
            info_text,
            color="white",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )
        plt.draw()

    # ------------------------------------------------------------------
    # Mode toggles
    # ------------------------------------------------------------------
    def toggle_add_mode(self, _event=None):
        self._deactivate_tools()
        if self.mode == "add":
            self.mode = "select"
            self.btn_add.label.set_text("Add Mode")
            print("SELECT mode")
        else:
            self.mode = "add"
            self.btn_add.label.set_text("Sel Mode")
            print("ADD mode: click pit centers to add")
        self.display_current_image()

    def activate_lasso(self, _event=None):
        self._deactivate_tools()
        self.mode = "lasso"
        print("LASSO mode: draw freehand region")
        self.state.lasso = LassoSelector(self.ax, onselect=self._on_lasso_select, useblit=True)
        self.display_current_image()

    def activate_circle(self, _event=None):
        self._deactivate_tools()
        self.mode = "circle"
        print("CIRCLE mode: click-drag to draw circle")
        self.state.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_circle_press)
        self.state.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_circle_motion)
        self.state.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_circle_release)

    def activate_manual(self, _event=None):
        self._deactivate_tools()
        self.mode = "manual"
        self.state.manual_active = False
        self.state.manual_pts = []
        print("MANUAL TRACE mode: click-drag to draw boundary")
        self.state.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_manual_press)
        self.state.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_manual_motion)
        self.state.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_manual_release)

    def _deactivate_tools(self):
        if self.state.lasso is not None:
            self.state.lasso.disconnect_events()
            self.state.lasso = None
        for cid in [self.state.cid_press, self.state.cid_motion, self.state.cid_release]:
            if cid is not None and self.fig is not None:
                self.fig.canvas.mpl_disconnect(cid)
        self.state.cid_press = self.state.cid_motion = self.state.cid_release = None
        if self.state.circle_artist is not None:
            self.state.circle_artist.remove()
            self.state.circle_artist = None
        self.state.circle_press = None
        self.state.manual_active = False
        self.state.manual_pts = []

    # ------------------------------------------------------------------
    # Lasso logic
    # ------------------------------------------------------------------
    def _on_lasso_select(self, verts):
        img = self.images[self.current_image_idx]
        ny, nx = img.shape
        ygrid, xgrid = np.mgrid[0:ny, 0:nx]
        pts = np.vstack((xgrid.ravel(), ygrid.ravel())).T
        path = MplPath(verts)
        mask = path.contains_points(pts).reshape((ny, nx))
        mask = morphology.binary_opening(mask, morphology.disk(2))
        mask = morphology.binary_closing(mask, morphology.disk(2))
        if not mask.any():
            print("Lasso region empty")
            return
        contours = self._fit_drawn_mask(mask)
        if not contours:
            print("Lasso failed to fit a pit to the selection")
            return
        added = 0
        for contour in contours:
            before = len(self.pits.get(self.current_image_idx, {}))
            self._add_contour_as_pit(contour)
            after = len(self.pits.get(self.current_image_idx, {}))
            if after > before:
                added += 1
        if added == 0:
            print("Lasso selection overlapped existing pits; nothing added")
        self._deactivate_tools()
        self.mode = "select"
        self.display_current_image()

    # ------------------------------------------------------------------
    # Circle logic
    # ------------------------------------------------------------------
    def _on_circle_press(self, event):
        if event.inaxes != self.ax:
            return
        self.state.circle_press = (event.xdata, event.ydata)

    def _on_circle_motion(self, event):
        if self.state.circle_press is None or event.inaxes != self.ax:
            return
        cx0, cy0 = self.state.circle_press
        r = np.hypot(event.xdata - cx0, event.ydata - cy0)
        theta = np.linspace(0, 2 * np.pi, 200)
        xs = cx0 + r * np.cos(theta)
        ys = cy0 + r * np.sin(theta)
        if self.state.circle_artist is not None:
            self.state.circle_artist.remove()
        self.state.circle_artist = self.ax.plot(xs, ys, "w--", linewidth=1.5)[0]
        self.fig.canvas.draw_idle()

    def _on_circle_release(self, event):
        if self.state.circle_press is None or event.inaxes != self.ax:
            return
        cx0, cy0 = self.state.circle_press
        r = np.hypot(event.xdata - cx0, event.ydata - cy0)
        self.state.circle_press = None
        if self.state.circle_artist is not None:
            self.state.circle_artist.remove()
            self.state.circle_artist = None

        img = self.images[self.current_image_idx]
        ny, nx = img.shape
        ygrid, xgrid = np.mgrid[0:ny, 0:nx]
        mask = (xgrid - cx0) ** 2 + (ygrid - cy0) ** 2 <= r ** 2
        mask = morphology.binary_closing(mask, morphology.disk(2))
        contours = self._fit_drawn_mask(mask)
        if not contours:
            print("Circle ROI failed to fit a pit")
            return
        for contour in contours:
            self._add_contour_as_pit(contour)
        self._deactivate_tools()
        self.mode = "select"
        self.display_current_image()

    # ------------------------------------------------------------------
    # Manual trace logic
    # ------------------------------------------------------------------
    def _on_manual_press(self, event):
        if event.inaxes != self.ax:
            return
        self.state.manual_active = True
        self.state.manual_pts = [(event.xdata, event.ydata)]

    def _on_manual_motion(self, event):
        if not self.state.manual_active or event.inaxes != self.ax:
            return
        self.state.manual_pts.append((event.xdata, event.ydata))
        xs = [p[0] for p in self.state.manual_pts]
        ys = [p[1] for p in self.state.manual_pts]
        self.ax.plot(xs, ys, "w-", linewidth=1)
        self.fig.canvas.draw_idle()

    def _on_manual_release(self, event):
        if not self.state.manual_active or event.inaxes != self.ax:
            return
        self.state.manual_active = False
        if len(self.state.manual_pts) < 10:
            print("Manual trace too short")
            self.state.manual_pts = []
            return
        pts = np.array(self.state.manual_pts, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3:
            print("Manual trace failed to capture a region")
            self.state.manual_pts = []
            return

        contour = np.array([[p[1], p[0]] for p in pts], dtype=np.float32)
        contour = np.vstack([contour, contour[:1]])
        self._add_contour_as_pit(contour)
        self._deactivate_tools()
        self.mode = "select"
        self.display_current_image()

    def _on_blur_sigma_change(self, value):
        self.detection_blur_sigma = max(0.0, float(value))

    # ------------------------------------------------------------------
    # Pit management helpers
    # ------------------------------------------------------------------
    def _add_contour_as_pit(self, contour: np.ndarray):
        if self.current_image_idx not in self.pits:
            self.pits[self.current_image_idx] = {}
        shape = self.images[self.current_image_idx].shape
        new_mask = mask_from_contour(contour, shape)
        if new_mask.sum() < 20:
            print("Pit too small to add.")
            return
        for ex_id, ex_contour in self.pits[self.current_image_idx].items():
            existing_mask = mask_from_contour(ex_contour, shape)
            if (new_mask & existing_mask).any():
                print("Adjusting pit to avoid crossing existing pits.")
                return
            if self.contains(ex_contour, contour, shape) or self.contains(contour, ex_contour, shape):
                print("Adjusting pit to avoid crossing existing pits.")
                return
        pit_id = self.next_pit_id
        self.next_pit_id += 1
        self.pits[self.current_image_idx][pit_id] = contour
        profile = extract_pit_profile(contour, self.images[self.current_image_idx])
        if profile:
            self.pit_profiles[pit_id] = profile
        self.selected_pit_ids = [pit_id]
        print(f"Added pit {pit_id} via interactive tool")

    def _fit_drawn_mask(self, mask: np.ndarray):
        """Refine a drawn mask and return one or more pit contours covering it."""

        image = self.images[self.current_image_idx]
        contour = mask_to_closed_contour(mask.astype(np.uint8))
        if contour is None:
            return []

        cy = float(np.mean(contour[:, 0]))
        cx = float(np.mean(contour[:, 1]))
        base_mask = mask.astype(bool)
        best = None
        best_score = -np.inf

        mask_rows = np.any(base_mask, axis=1)
        mask_cols = np.any(base_mask, axis=0)
        if mask_rows.any() and mask_cols.any():
            min_row, max_row = np.where(mask_rows)[0][[0, -1]]
            min_col, max_col = np.where(mask_cols)[0][[0, -1]]
            bbox_half = max(max_row - min_row, max_col - min_col) / 2.0
        else:
            bbox_half = 0.0
        need_large = bbox_half > max(self.roi_halves, default=0) * 0.9 or base_mask.sum() > 5000

        methods = [0, 1, 2]
        if need_large or getattr(self, "large_pit_mode", False):
            methods = [3] + methods

        prev_large = self.large_pit_mode
        if need_large and not prev_large:
            self.large_pit_mode = True

        try:
            for method in methods:
                candidate = self.detect_pit_edge(image, (cx, cy), method=method)
                if candidate is None:
                    continue
                cand_mask = mask_from_contour(candidate, image.shape)
                if cand_mask.sum() == 0:
                    continue
                overlap = (base_mask & cand_mask).sum() / (base_mask.sum() + 1e-6)
                if overlap < 0.6:
                    continue
                spill = (cand_mask & ~base_mask).sum() / (base_mask.sum() + 1e-6)
                score = overlap - 0.2 * spill
                if score > best_score:
                    best_score = score
                    best = candidate
                if best_score > 0.85:
                    break
        finally:
            if need_large and not prev_large:
                self.large_pit_mode = prev_large

        if best is not None:
            best_mask = mask_from_contour(best, image.shape).astype(bool)
            combined = base_mask | best_mask
            refined = self._geodesic_refine_mask(combined, image, (cx, cy))
            if refined is not None and refined.any():
                refined_contour = mask_to_closed_contour(refined.astype(np.uint8))
                if refined_contour is not None:
                    best = refined_contour

        if best is None:
            refined_mask = self._geodesic_refine_mask(base_mask, image, (cx, cy))
            if refined_mask is not None and refined_mask.any():
                refined_contour = mask_to_closed_contour(refined_mask.astype(np.uint8))
                if refined_contour is not None:
                    best = refined_contour

        if best is None:
            best = contour

        aligned = align_contour_to_gradient(best, image)
        if aligned is not None:
            aligned_mask = mask_from_contour(aligned, image.shape)
            if aligned_mask.sum() > 0:
                overlap = (base_mask & aligned_mask).sum() / (base_mask.sum() + 1e-6)
                if overlap >= 0.6:
                    best = aligned

        pinch_masks = self._pinch_components(
            mask_from_contour(best, image.shape).astype(bool), image
        )
        candidate_contours = self._components_to_contours(
            pinch_masks,
            image,
            seed_point=(cx, cy),
            allow_multiple=True,
        )
        if not candidate_contours:
            return [best]

        filtered: List[np.ndarray] = []
        for cand in candidate_contours:
            cand_mask = mask_from_contour(cand, image.shape)
            if cand_mask.sum() == 0:
                continue
            overlap = (base_mask & cand_mask).sum() / (base_mask.sum() + 1e-6)
            if overlap >= 0.45:
                filtered.append(cand)

        return filtered if filtered else candidate_contours

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if self.mode == "add":
            contour = self.detect_pit_edge(self.images[self.current_image_idx], (x, y), 0)
            if contour is not None:
                self._add_contour_as_pit(contour)
                self.display_current_image()
            else:
                print("No pit detected at click (or seed not contained / too large).")
        elif self.mode == "select":
            min_dist = float("inf")
            selected_pit = None
            if self.current_image_idx in self.pits:
                for pit_id, contour in self.pits[self.current_image_idx].items():
                    cy = np.mean(contour[:, 0])
                    cx = np.mean(contour[:, 1])
                    dist = np.hypot(cx - x, cy - y)
                    if dist < min_dist and dist < 50:
                        min_dist = dist
                        selected_pit = pit_id
            if selected_pit is not None:
                if selected_pit in self.selected_pit_ids:
                    self.selected_pit_ids.remove(selected_pit)
                    print(f"Deselected pit {selected_pit}")
                else:
                    self.selected_pit_ids.append(selected_pit)
                    print(f"Selected pit {selected_pit}")
                self.display_current_image()

    # ------------------------------------------------------------------
    # Higher level actions
    # ------------------------------------------------------------------
    def refit_selected_robust(self, _event=None):
        if not self.selected_pit_ids:
            print("No pits selected for refitting")
            return
        print("Refitting (robust): trying multiple methods & ROI sizes, enforcing seed containment.")
        for pit_id in self.selected_pit_ids:
            if self.current_image_idx not in self.pits or pit_id not in self.pits[self.current_image_idx]:
                continue
            contour = self.pits[self.current_image_idx][pit_id]
            cy = np.mean(contour[:, 0])
            cx = np.mean(contour[:, 1])
            best = None
            best_score = -1.0
            for m in [0, 1, 2]:
                c = self.detect_pit_edge(self.images[self.current_image_idx], (cx, cy), method=m)
                if c is None:
                    continue
                iou_val = self.iou(contour, c, self.images[self.current_image_idx].shape)
                area0 = mask_from_contour(contour, self.images[self.current_image_idx].shape).sum()
                area1 = mask_from_contour(c, self.images[self.current_image_idx].shape).sum()
                area_sim = 1.0 - min(1.0, abs(area1 - area0) / (area0 + 1e-6))
                score = iou_val + 0.1 * area_sim
                if score > best_score:
                    best_score = score
                    best = c
            if best is not None:
                self.pits[self.current_image_idx][pit_id] = best
                profile = extract_pit_profile(best, self.images[self.current_image_idx])
                if profile:
                    self.pit_profiles[pit_id] = profile
                print(f"Refit pit {pit_id}: score={best_score:.3f}")
            else:
                print(f"Refit failed for pit {pit_id}")
        self.display_current_image()

    def delete_selected(self, _event=None):
        if not self.selected_pit_ids:
            print("No pits selected for deletion")
            return
        for pit_id in self.selected_pit_ids:
            if self.current_image_idx in self.pits and pit_id in self.pits[self.current_image_idx]:
                del self.pits[self.current_image_idx][pit_id]
                if pit_id in self.pit_profiles:
                    del self.pit_profiles[pit_id]
        print(f"Deleted {len(self.selected_pit_ids)} pits")
        self.selected_pit_ids = []
        self.display_current_image()

    def clear_selection(self, _event=None):
        self.selected_pit_ids = []
        print("Selection cleared")
        self.display_current_image()

    def select_all_pits(self, _event=None):
        frame_pits = self.pits.get(self.current_image_idx, {})
        if not frame_pits:
            self.selected_pit_ids = []
            print("No pits to select.")
        else:
            self.selected_pit_ids = list(frame_pits.keys())
            print(f"Selected all {len(self.selected_pit_ids)} pits")
        self.display_current_image()

    def toggle_large_fit_mode(self, _event=None):
        self.large_pit_mode = not self.large_pit_mode
        if hasattr(self, "btn_large_mode"):
            self.btn_large_mode.label.set_text("Large Fit: On" if self.large_pit_mode else "Large Fit: Off")
        state = "enabled" if self.large_pit_mode else "disabled"
        print(f"Large pit fitting {state}.")
        self.display_current_image()

    def auto_detect_similar(self, _event=None):
        frame_pits = self.pits.get(self.current_image_idx, {})
        if not frame_pits:
            print("Auto Similar: add a few manual/interactive pits first.")
            return

        ref_ids = self.selected_pit_ids if self.selected_pit_ids else list(frame_pits.keys())
        reference_profiles = []
        for pid in ref_ids:
            contour = frame_pits.get(pid)
            if contour is None:
                continue
            profile = extract_pit_profile(contour, self.images[self.current_image_idx])
            if profile:
                reference_profiles.append(profile)

        if len(reference_profiles) < 2:
            print("Auto Similar: need at least two valid reference pits (select more or add a couple).")
            return

        print(f"Auto Similar: scanning with {len(reference_profiles)} reference profiles...")
        sims = self.find_similar_pits(self.images[self.current_image_idx], reference_profiles)

        added = 0
        existing = list(frame_pits.values())
        for cand in sims:
            contour = cand["contour"]
            dup = False
            for ex in existing:
                if (
                    self.iou(contour, ex, self.images[self.current_image_idx].shape) > 0.2
                    or self.contains(ex, contour, self.images[self.current_image_idx].shape)
                    or self.contains(contour, ex, self.images[self.current_image_idx].shape)
                ):
                    dup = True
                    break
            if dup:
                continue
            pid = self.next_pit_id
            self.next_pit_id += 1
            self.pits[self.current_image_idx][pid] = contour
            self.pit_profiles[pid] = cand["profile"]
            existing.append(contour)
            added += 1
        print(f"Auto Similar: added {added} pits.")
        self.display_current_image()

    def align_selected_edges(self, _event=None):
        if not self.selected_pit_ids:
            print("Align Edge: select at least one pit first.")
            return
        if self.current_image_idx not in self.pits:
            print("Align Edge: no pits on this frame.")
            return
        aligned = 0
        image = self.images[self.current_image_idx]
        shape = image.shape
        for pit_id in list(self.selected_pit_ids):
            contour = self.pits[self.current_image_idx].get(pit_id)
            if contour is None:
                continue
            new_contour = align_contour_to_gradient(contour, image)
            if new_contour is None:
                continue
            pinch_masks = self._pinch_components(
                mask_from_contour(new_contour, shape).astype(bool), image
            )
            finalized = self._components_to_contours(
                pinch_masks,
                image,
                seed_point=(np.mean(new_contour[:, 1]), np.mean(new_contour[:, 0])),
                allow_multiple=False,
            )
            if not finalized:
                continue
            new_contour = finalized[0]
            mask = mask_from_contour(new_contour, shape)
            if mask.sum() < 20:
                continue
            self.pits[self.current_image_idx][pit_id] = new_contour
            profile = extract_pit_profile(new_contour, image)
            if profile:
                self.pit_profiles[pit_id] = profile
            aligned += 1
        if aligned:
            print(f"Align Edge: updated {aligned} pit(s).")
            self.display_current_image()
        else:
            print("Align Edge: unable to improve selected pits.")

    def confirm_frame(self, _event=None):
        if self.current_image_idx not in self.pits or not self.pits[self.current_image_idx]:
            print("No pits to confirm!")
            return
        print(f"Frame {self.current_image_idx + 1} confirmed with {len(self.pits[self.current_image_idx])} pits")
        self.selected_pit_ids = []
        self.display_current_image()

    def prev_frame(self, _event=None):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.selected_pit_ids = []
            self.display_current_image()

    def next_frame(self, _event=None):
        if self.current_image_idx < len(self.images) - 1:
            self.current_image_idx += 1
            self.selected_pit_ids = []
            if self.current_image_idx not in self.pits and self.current_image_idx > 0:
                if self.current_image_idx - 1 in self.pits:
                    print(f"Auto-tracking frame {self.current_image_idx + 1}...")
                    self.pits[self.current_image_idx] = self.track_pits_to_next_frame(
                        self.current_image_idx - 1, self.current_image_idx
                    )
            self.display_current_image()

    def process_all_frames(self, _event=None):
        if 0 not in self.pits:
            print("Please set up frame 1 first!")
            return
        print("Processing all frames...")
        for frame_idx in range(1, len(self.images)):
            if frame_idx not in self.pits:
                print(f"Processing frame {frame_idx + 1}...")
                self.pits[frame_idx] = self.track_pits_to_next_frame(frame_idx - 1, frame_idx)
        print("All frames processed!")
        self.display_current_image()

    def export_results(self, _event=None):
        print("Exporting results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.image_folder / f"pit_analysis_{timestamp}"
        output_dir.mkdir(exist_ok=True)

        for frame_idx in range(len(self.images)):
            if frame_idx not in self.pits:
                continue
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.images[frame_idx], cmap="hot")
            for _, contour in self.pits[frame_idx].items():
                ax.plot(contour[:, 1], contour[:, 0], "cyan", linewidth=2)
            ax.set_title(f"Frame {frame_idx + 1}")
            ax.axis("off")
            output_file = output_dir / f"frame_{frame_idx + 1:03d}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

        rates = self.calculate_corrosion_rates()
        if rates:
            df = pd.DataFrame(rates)
            df.to_csv(output_dir / "corrosion_rates.csv", index=False)
            print(f"Saved corrosion rates for {len(df)} transitions.")
        print(f"Results exported to: {output_dir}")

    # Placeholder - to be implemented or overridden as needed
    def calculate_corrosion_rates(self):  # pragma: no cover
        return []
