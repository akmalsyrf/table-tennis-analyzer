from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    conf: float


class BallDetector:
    """
    Very simple detector wrapper.

    Strategy:
    - Try YOLOv8 via ultralytics if available and model loads
    - Otherwise fall back to a simple color-based heuristic (orange/white-ish ball)
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        yolo_conf: float = 0.25,
        prefer_coco_sports_ball: bool = True,
    ) -> None:
        self._yolo = None
        self._yolo_ready = False
        self._yolo_conf = float(yolo_conf)
        self._prefer_coco_sports_ball = bool(prefer_coco_sports_ball)

        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo = YOLO(yolo_model)
            self._yolo_ready = True
            logger.info("YOLO detector ready with model=%s", yolo_model)
        except Exception as e:
            self._yolo = None
            self._yolo_ready = False
            logger.info("YOLO unavailable; using fallback detector (%s)", e)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        """
        Return the most likely ball center as a Detection, or None if not found.
        """
        if self._yolo_ready:
            det = self._detect_yolo(frame_bgr)
            if det is not None:
                return det

        return self._detect_fallback(frame_bgr)

    def _detect_yolo(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        assert self._yolo is not None
        try:
            res = self._yolo.predict(frame_bgr, verbose=False)
            if not res:
                return None
            boxes = res[0].boxes
            if boxes is None or len(boxes) == 0:
                return None

            confs = boxes.conf.detach().cpu().numpy().astype(float)
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
            clses = None
            try:
                clses = boxes.cls.detach().cpu().numpy().astype(int)
            except Exception:
                clses = None

            # COCO class id for "sports ball" is 32 in common YOLO COCO mappings.
            # Filtering massively reduces false positives for the default yolov8n.pt model.
            idxs = np.arange(len(confs))
            idxs = idxs[confs >= self._yolo_conf]
            if idxs.size == 0:
                return None

            if self._prefer_coco_sports_ball and clses is not None and len(clses) == len(confs):
                ball_idxs = idxs[clses[idxs] == 32]
                if ball_idxs.size > 0:
                    idxs = ball_idxs

            best_i = int(idxs[np.argmax(confs[idxs])])
            x1, y1, x2, y2 = xyxy[best_i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return Detection(x=cx, y=cy, conf=float(confs[best_i]))
        except Exception:
            logger.debug("YOLO detect failed; falling back", exc_info=True)
            return None

    def _detect_fallback(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        """
        Very rough heuristic:
        - blur
        - HSV threshold for bright/white/orange-ish spots
        - find small-ish circular-ish contour and return its center
        """
        try:
            h, w = frame_bgr.shape[:2]
            if h == 0 or w == 0:
                return None

            blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            # White-ish (low saturation, high value)
            white_mask = cv2.inRange(hsv, (0, 0, 200), (179, 70, 255))
            # Orange-ish (common table tennis ball color)
            orange_mask = cv2.inRange(hsv, (5, 120, 120), (25, 255, 255))

            mask = cv2.bitwise_or(white_mask, orange_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            best = None
            best_score = -1.0
            for c in contours:
                area = float(cv2.contourArea(c))
                if area < 4.0 or area > 800.0:
                    continue
                _, _, cw, ch = cv2.boundingRect(c)
                if cw <= 0 or ch <= 0:
                    continue
                aspect = cw / float(ch)
                if aspect < 0.5 or aspect > 2.0:
                    continue

                m = cv2.moments(c)
                if m["m00"] == 0:
                    continue
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])

                # Prefer small, compact blobs near the center (weak prior)
                center_dist = ((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) ** 0.5
                score = (1.0 / (1.0 + center_dist)) + (1.0 / (1.0 + abs(area - 40.0)))
                if score > best_score:
                    best_score = score
                    best = (cx, cy)

            if best is None:
                return None
            return Detection(x=best[0], y=best[1], conf=0.25)
        except Exception:
            logger.debug("Fallback detect failed", exc_info=True)
            return None

