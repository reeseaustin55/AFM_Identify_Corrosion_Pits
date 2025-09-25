"""Image loading utilities for AFM pit analysis."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


def find_ibw_sidecar(png_path: Path) -> Optional[Path]:
    """Return the IBW file that shares a stem with ``png_path``.

    Some export pipelines flip the case of the extension when copying files
    between systems, so we perform a case-insensitive search that still prefers
    the straightforward ``.ibw`` match when available.
    """

    primary = png_path.with_suffix(".ibw")
    if primary.exists():
        return primary

    alt = png_path.with_suffix(".IBW")
    if alt.exists():
        return alt

    stem = png_path.stem
    for sibling in png_path.parent.iterdir():
        if sibling.stem == stem and sibling.suffix.lower() == ".ibw":
            return sibling
    return None


def load_image_series(image_folder: Path):
    """Load grayscale PNG images and associated metadata."""
    images: List = []
    image_files: List[Path] = []
    timestamps: List[datetime] = []
    nm_per_pixel = None

    png_files = sorted(image_folder.glob("*.png"))
    if not png_files:
        return images, image_files, timestamps, nm_per_pixel

    for png_file in png_files:
        parts = png_file.name.split("_")
        if "nm" in png_file.name:
            for part in parts:
                if "nm" in part:
                    nm_value = part.replace("nm", "")
                    try:
                        if "px" in parts[0]:
                            px_value = float(parts[0].replace("px", ""))
                        else:
                            px_value = 1024
                        nm_per_pixel = float(nm_value) / px_value
                    except Exception:
                        pass
                    break
        img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            image_files.append(png_file)

            # Prefer the timestamp from the paired IBW metadata file, if available.
            ibw_path = find_ibw_sidecar(png_file)

            if ibw_path is not None:
                stat = ibw_path.stat()
                timestamp = getattr(stat, "st_mtime_ns", None)
                if timestamp is not None:
                    timestamp /= 1e9
                else:
                    timestamp = stat.st_mtime
            else:
                timestamp = os.path.getmtime(png_file)

            timestamps.append(datetime.fromtimestamp(timestamp))

    if nm_per_pixel is None and images:
        nm_per_pixel = 250.0 / 1024.0
    return images, image_files, timestamps, nm_per_pixel
