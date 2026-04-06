"""
Auto-detect the table region in broadcast table-tennis footage via edge
detection.

Previous approach (HSV colour segmentation) failed because the floor,
backdrop, jerseys, and table are all the same colour in WTT/ITTF broadcasts.

Current approach — **edge-based detection**:

1. Convert to grayscale and apply Canny edge detection.
2. Use Hough Line Transform to find horizontal line segments in the
   centre-bottom region of the frame.
3. Cluster nearby horizontal lines to estimate the table surface extent.
4. If the line cluster passes geometric sanity checks (width, aspect,
   position), return it as the table ROI.
5. Otherwise return ``None`` (camera cut, replay, close-up).

The table tennis table's **white lines** (centre line, edge lines) are its
most distinctive visual feature — far more unique than "blue blob".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableROI:
    """Axis-aligned bounding box of the *play area* (table + airspace above)."""

    x: int
    y: int
    w: int
    h: int

    table_x: int
    table_y: int
    table_w: int
    table_h: int

    def contains(self, px: float, py: float, *, margin: float = 0.0) -> bool:
        return (
            self.x - margin <= px <= self.x + self.w + margin
            and self.y - margin <= py <= self.y + self.h + margin
        )


def _find_table_via_edges(
    frame_bgr: np.ndarray,
    *,
    canny_lo: int = 50,
    canny_hi: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 60,
    max_line_gap: int = 15,
    angle_tolerance_deg: float = 8.0,
) -> tuple[int, int, int, int] | None:
    """
    Detect the table surface rectangle from its white lines using Canny + Hough.

    Returns (x, y, w, h) of the estimated table surface, or None.
    """
    fh, fw = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Focus on the centre-bottom region where the table appears in broadcast.
    search_y_start = int(fh * 0.30)
    search_y_end = int(fh * 0.85)
    search_x_start = int(fw * 0.10)
    search_x_end = int(fw * 0.90)
    roi_gray = gray[search_y_start:search_y_end, search_x_start:search_x_end]

    edges = cv2.Canny(roi_gray, canny_lo, canny_hi)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None or len(lines) == 0:
        return None

    # Keep only near-horizontal lines (angle within tolerance of 0 degrees).
    angle_rad = angle_tolerance_deg * np.pi / 180.0
    horizontal_lines: list[tuple[int, int, int, int]] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < 1:
            continue
        angle = np.arctan2(dy, dx)
        if angle <= angle_rad:
            gy1 = y1 + search_y_start
            gy2 = y2 + search_y_start
            gx1 = x1 + search_x_start
            gx2 = x2 + search_x_start
            horizontal_lines.append((gx1, gy1, gx2, gy2))

    if len(horizontal_lines) < 2:
        return None

    return _cluster_lines_to_table(horizontal_lines, fw, fh)


def _cluster_lines_to_table(
    lines: list[tuple[int, int, int, int]],
    fw: int,
    fh: int,
) -> tuple[int, int, int, int] | None:
    """
    Given a set of near-horizontal lines, find a cluster that forms a
    table-shaped rectangle (wide, moderate height, centred).
    """
    y_coords = np.array([(l[1] + l[3]) / 2.0 for l in lines])
    x_mins = np.array([min(l[0], l[2]) for l in lines])
    x_maxs = np.array([max(l[0], l[2]) for l in lines])

    # Sort by y-coordinate.
    order = np.argsort(y_coords)
    y_sorted = y_coords[order]

    # Find the densest vertical band of horizontal lines (table region).
    best_cluster = _find_densest_y_band(y_sorted, order, max_band_height=fh * 0.25)
    if best_cluster is None or len(best_cluster) < 2:
        return None

    cluster_x_mins = x_mins[best_cluster]
    cluster_x_maxs = x_maxs[best_cluster]
    cluster_ys = y_coords[best_cluster]

    table_x = int(np.min(cluster_x_mins))
    table_y = int(np.min(cluster_ys))
    table_x2 = int(np.max(cluster_x_maxs))
    table_y2 = int(np.max(cluster_ys))
    table_w = table_x2 - table_x
    table_h = max(table_y2 - table_y, 1)

    if not _passes_geometry_check(table_x, table_y, table_w, table_h, fw, fh):
        return None

    return (table_x, table_y, table_w, table_h)


def _find_densest_y_band(
    y_sorted: np.ndarray,
    order: np.ndarray,
    max_band_height: float,
) -> np.ndarray | None:
    """Sliding window to find the y-band containing the most horizontal lines."""
    n = len(y_sorted)
    best_start = 0
    best_count = 0

    lo = 0
    for hi in range(n):
        while y_sorted[hi] - y_sorted[lo] > max_band_height:
            lo += 1
        count = hi - lo + 1
        if count > best_count:
            best_count = count
            best_start = lo

    if best_count < 2:
        return None

    return order[best_start : best_start + best_count]


def _passes_geometry_check(
    tx: int, ty: int, tw: int, th: int, fw: int, fh: int,
) -> bool:
    """Sanity-check that the detected region looks like a table."""
    if tw < fw * 0.15 or tw > fw * 0.85:
        return False
    if th < 3:
        return False
    aspect = tw / max(th, 1)
    if aspect < 2.0 or aspect > 15.0:
        return False
    cx = tx + tw / 2.0
    if abs(cx - fw / 2.0) > fw * 0.30:
        return False
    cy = ty + th / 2.0
    vy = cy / max(fh, 1)
    if vy < 0.30 or vy > 0.85:
        return False
    return True


def _build_roi_from_table(
    tx: int, ty: int, tw: int, th: int,
    fw: int, fh: int,
    *,
    vertical_expand: float = 1.8,
    horizontal_margin: float = 0.12,
) -> TableROI:
    expand_up = int(th * vertical_expand)
    h_margin = int(fw * horizontal_margin)

    roi_x = max(0, tx - h_margin)
    roi_y = max(0, ty - expand_up)
    roi_w = min(fw - roi_x, tw + 2 * h_margin)
    roi_h = min(fh - roi_y, (ty + th) - roi_y)

    return TableROI(
        x=roi_x, y=roi_y, w=roi_w, h=roi_h,
        table_x=tx, table_y=ty, table_w=tw, table_h=th,
    )


def detect_table_roi_frame(
    frame_bgr: np.ndarray,
    **_kwargs,
) -> TableROI | None:
    """
    Detect the table ROI from a single (already-resized) frame.
    Returns ``None`` when the table is not visible (e.g. camera cut / replay).

    Accepts **kwargs for backward compatibility with callers passing
    ``tuning=...`` (edge-based detection does not use TableRoiTuning).
    """
    rect = _find_table_via_edges(frame_bgr)
    if rect is None:
        return None

    fh, fw = frame_bgr.shape[:2]
    tx, ty, tw, th = rect
    return _build_roi_from_table(tx, ty, tw, th, fw, fh)
