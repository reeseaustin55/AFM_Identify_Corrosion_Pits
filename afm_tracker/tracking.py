"""Drift correction and pit tracking helpers."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .profile import extract_pit_profile
from .utils import contains, iou, mask_from_contour


class TrackingMixin:
    images: list
    pits: Dict[int, Dict[int, np.ndarray]]
    pit_profiles: Dict[int, dict]
    next_pit_id: int

    def correct_drift(self, from_idx: int, to_idx: int) -> Tuple[float, float]:
        """Estimate the translation that best aligns ``from_idx`` with ``to_idx``."""

        if from_idx == to_idx:
            return (0.0, 0.0)
        A = self.images[from_idx].astype(np.float32)
        B = self.images[to_idx].astype(np.float32)

        def norm(z):
            z = z - np.mean(z)
            s = np.std(z) + 1e-6
            return z / s

        A = norm(A)
        B = norm(B)
        warp_mode = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            for scale in [4, 2, 1]:
                As = cv2.resize(A, (A.shape[1] // scale, A.shape[0] // scale), interpolation=cv2.INTER_AREA)
                Bs = cv2.resize(B, (B.shape[1] // scale, B.shape[0] // scale), interpolation=cv2.INTER_AREA)
                wrp = warp.copy()
                wrp[:, :2] /= scale
                _, wrp = cv2.findTransformECC(As, Bs, wrp, warp_mode, criteria, None, 5)
                warp = wrp
            dx, dy = warp[0, 2], warp[1, 2]
            return (float(dx), float(dy))
        except Exception:
            hann = cv2.createHanningWindow((A.shape[1], A.shape[0]), cv2.CV_32F)
            s, _ = cv2.phaseCorrelate(A * hann, B * hann)
            return (float(s[0]), float(s[1]))

    def track_pits_to_next_frame(self, from_idx: int, to_idx: int):
        """Track pit contours from ``from_idx`` into ``to_idx`` using detection."""

        if from_idx not in self.pits or len(self.pits[from_idx]) == 0:
            return {}
        to_img = self.images[to_idx]
        H, W = to_img.shape[:2]
        dx, dy = self.correct_drift(from_idx, to_idx)
        print(f"Drift (ECC): ({dx:.2f}, {dy:.2f}) px")

        proposals, proposal_ids = [], []
        for pit_id, contour in self.pits[from_idx].items():
            cy = float(np.mean(contour[:, 0])) + dy
            cx = float(np.mean(contour[:, 1])) + dx
            if 0 <= cx < W and 0 <= cy < H:
                new_contour = self.detect_pit_edge(to_img, (cx, cy), method=0)
                if new_contour is not None and len(new_contour) >= 10:
                    proposals.append(new_contour)
                    proposal_ids.append(pit_id)

        kept, kept_ids = [], []
        for i, c in enumerate(proposals):
            if any(iou(c, kc, to_img.shape) > 0.3 for kc in kept):
                continue
            kept.append(c)
            kept_ids.append(proposal_ids[i])

        candidates = {kept_ids[i]: kept[i] for i in range(len(kept))}
        if to_idx in self.pits:
            for pid, cnt in self.pits[to_idx].items():
                candidates[pid] = cnt

        prev_ids = list(self.pits[from_idx].keys())
        cand_ids = list(candidates.keys())
        if not prev_ids or not cand_ids:
            return {pid: candidates[pid] for pid in cand_ids}

        alpha, beta = 0.7, 0.3
        cost = np.zeros((len(prev_ids), len(cand_ids)), dtype=np.float32)
        for i, pid in enumerate(prev_ids):
            pc = self.pits[from_idx][pid]
            pcy = np.mean(pc[:, 0]) + dy
            pcx = np.mean(pc[:, 1]) + dx
            for j, cid in enumerate(cand_ids):
                cc = candidates[cid]
                iou_val = iou(pc, cc, to_img.shape)
                ccy = np.mean(cc[:, 0])
                ccx = np.mean(cc[:, 1])
                dist = np.hypot(ccx - pcx, ccy - pcy)
                cost[i, j] = alpha * (1.0 - iou_val) + beta * (dist / 50.0)

        row_ind, col_ind = linear_sum_assignment(cost)

        new_pits, used_cands = {}, set()
        for r, c in zip(row_ind, col_ind):
            pid = prev_ids[r]
            cid = cand_ids[c]
            if cost[r, c] < 0.9:
                new_pits[pid] = candidates[cid]
                used_cands.add(cid)

        for cid in cand_ids:
            if cid in used_cands:
                continue
            c_contour = candidates[cid]
            if any(
                iou(c_contour, ex_contour, to_img.shape) > 0.2
                or contains(ex_contour, c_contour, to_img.shape)
                or contains(c_contour, ex_contour, to_img.shape)
                for ex_contour in new_pits.values()
            ):
                continue
            nid = self.next_pit_id
            self.next_pit_id += 1
            new_pits[nid] = c_contour

        if to_idx not in self.pit_lineage:
            self.pit_lineage[to_idx] = {}
        for nid, cnt in new_pits.items():
            if nid in self.pits.get(from_idx, {}):
                self.pit_lineage[to_idx][nid] = [nid]
            else:
                best_parent = None
                best_iou = 0.0
                for pid, pc in self.pits[from_idx].items():
                    iou_val = iou(pc, cnt, to_img.shape)
                    if iou_val > best_iou:
                        best_iou, best_parent = iou_val, pid
                self.pit_lineage[to_idx][nid] = [best_parent] if best_parent is not None else [nid]
        return new_pits
