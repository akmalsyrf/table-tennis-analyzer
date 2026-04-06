"""
Frame-differencing motion detector for small fast-moving objects.

The table-tennis ball is typically 5-15 px across in broadcast footage but
moves very fast — frame differencing picks up this motion signal even when
YOLO (trained on COCO) misses the ball entirely.

Usage::

    md = MotionDetector()
    blobs = md.detect(prev_gray, curr_gray)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionBlob:
    """A small region of motion detected via frame differencing."""

    cx: float
    cy: float
    area: float
    intensity: float  # mean pixel intensity in the diff image (0..255)


_KERNEL_3 = np.ones((3, 3), np.uint8)


class MotionDetector:
    """
    Detect small moving objects by comparing consecutive grayscale frames.

    Parameters
    ----------
    diff_threshold:
        Pixel-level threshold on the absolute difference image (0-255).
    min_area / max_area:
        Contour area bounds — filters out noise (too small) and large
        camera-motion artefacts (too large).
    aspect_max:
        Maximum width/height ratio.  Rejects elongated blobs that are
        typically edge artefacts rather than ball motion.
    """

    def __init__(
        self,
        *,
        diff_threshold: int = 25,
        min_area: float = 4.0,
        max_area: float = 800.0,
        aspect_max: float = 3.0,
    ) -> None:
        self._thresh = diff_threshold
        self._min_area = min_area
        self._max_area = max_area
        self._aspect_max = aspect_max

    def detect(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> list[MotionBlob]:
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, mask = cv2.threshold(diff, self._thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        return self._filter_contours(contours, diff)

    def _filter_contours(
        self,
        contours: list,
        diff: np.ndarray,
    ) -> list[MotionBlob]:
        blobs: list[MotionBlob] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self._min_area or area > self._max_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue
            aspect = max(w, h) / float(min(w, h))
            if aspect > self._aspect_max:
                continue

            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])

            roi = diff[y : y + h, x : x + w]
            intensity = float(np.mean(roi)) if roi.size > 0 else 0.0

            blobs.append(MotionBlob(cx=cx, cy=cy, area=area, intensity=intensity))

        return blobs
