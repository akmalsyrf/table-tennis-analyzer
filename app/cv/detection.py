from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.cv.motion import MotionBlob, MotionDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    conf: float
    area: float = 0.0
    source: str = "yolo"  # "yolo" or "yolo+motion"


class BallDetector:
    """
    Combined ball detector: YOLO object detection + frame-differencing motion.

    Strategy:
    1. Run YOLO for "sports ball" candidates
    2. Run frame differencing for motion blobs
    3. Score each YOLO candidate with a motion-confirmation bonus
       (motion alone is NOT used as a detection source — it produces
       too many false positives from crowd, graphics, camera pan)
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        *,
        yolo_conf: float,
        prefer_coco_sports_ball: bool = True,
        max_box_area: float = 6000.0,
        min_box_area: float = 4.0,
        motion_match_radius: float = 40.0,
    ) -> None:
        self._yolo = None
        self._yolo_ready = False
        self._yolo_conf = float(yolo_conf)
        self._prefer_coco_sports_ball = bool(prefer_coco_sports_ball)
        self._max_box_area = float(max_box_area)
        self._min_box_area = float(min_box_area)
        self._motion_match_radius = float(motion_match_radius)
        self._motion = MotionDetector()

        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo = YOLO(yolo_model)
            self._yolo_ready = True
            logger.info("YOLO detector ready with model=%s", yolo_model)
        except Exception as e:
            self._yolo = None
            self._yolo_ready = False
            logger.info("YOLO unavailable; using fallback detector (%s)", e)

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        last_pos: tuple[float, float] | None = None,
        prev_gray: np.ndarray | None = None,
    ) -> Optional[Detection]:
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        motion_blobs: list[MotionBlob] = []
        if prev_gray is not None:
            motion_blobs = self._motion.detect(prev_gray, curr_gray)

        if self._yolo_ready:
            return self._detect_yolo_with_motion(
                frame_bgr,
                motion_blobs=motion_blobs,
                last_pos=last_pos,
            )

        return None

    def _detect_yolo_with_motion(
        self,
        frame_bgr: np.ndarray,
        *,
        motion_blobs: list[MotionBlob],
        last_pos: tuple[float, float] | None = None,
    ) -> Optional[Detection]:
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
            clses = _get_classes(boxes)

            idxs = np.arange(len(confs))
            idxs = idxs[confs >= self._yolo_conf]
            if idxs.size == 0:
                return None

            if self._prefer_coco_sports_ball and clses is not None:
                ball_idxs = idxs[clses[idxs] == 32]
                if ball_idxs.size > 0:
                    idxs = ball_idxs

            areas = (xyxy[idxs, 2] - xyxy[idxs, 0]) * (xyxy[idxs, 3] - xyxy[idxs, 1])
            size_mask = (areas >= self._min_box_area) & (areas <= self._max_box_area)
            idxs = idxs[size_mask]
            if idxs.size == 0:
                return None

            return self._pick_best_candidate(
                xyxy, confs, idxs, frame_bgr.shape[:2],
                motion_blobs=motion_blobs,
                last_pos=last_pos,
            )
        except Exception:
            logger.debug("YOLO detect failed", exc_info=True)
            return None

    def _pick_best_candidate(
        self,
        xyxy: np.ndarray,
        confs: np.ndarray,
        idxs: np.ndarray,
        frame_hw: tuple[int, int],
        *,
        motion_blobs: list[MotionBlob],
        last_pos: tuple[float, float] | None,
    ) -> Detection:
        cxs = (xyxy[idxs, 0] + xyxy[idxs, 2]) / 2.0
        cys = (xyxy[idxs, 1] + xyxy[idxs, 3]) / 2.0
        h, w = frame_hw
        diag = max(1.0, np.sqrt(float(w * w + h * h)))

        yolo_scores = confs[idxs].copy()

        # Proximity to last known position.
        prox_scores = np.zeros_like(yolo_scores)
        if last_pos is not None:
            dists = np.sqrt((cxs - last_pos[0]) ** 2 + (cys - last_pos[1]) ** 2)
            prox_scores = 1.0 - np.clip(dists / diag, 0, 1)

        # Motion confirmation: does a nearby motion blob exist?
        motion_scores = np.zeros_like(yolo_scores)
        if motion_blobs:
            motion_scores = _compute_motion_scores(
                cxs, cys, motion_blobs, self._motion_match_radius,
            )

        # Weighted combination: YOLO conf + proximity + motion confirmation.
        # Motion-confirmed candidates get a significant boost.
        scores = 0.30 * yolo_scores + 0.35 * prox_scores + 0.35 * motion_scores

        best_j = int(np.argmax(scores))
        best_i = int(idxs[best_j])
        x1, y1, x2, y2 = xyxy[best_i]
        source = "yolo+motion" if motion_scores[best_j] > 0.1 else "yolo"
        return Detection(
            x=(x1 + x2) / 2.0,
            y=(y1 + y2) / 2.0,
            conf=float(confs[best_i]),
            area=float((x2 - x1) * (y2 - y1)),
            source=source,
        )


def _get_classes(boxes) -> np.ndarray | None:
    try:
        return boxes.cls.detach().cpu().numpy().astype(int)
    except Exception:
        return None


def _compute_motion_scores(
    cxs: np.ndarray,
    cys: np.ndarray,
    blobs: list[MotionBlob],
    radius: float,
) -> np.ndarray:
    """For each YOLO candidate, find the closest motion blob within *radius*."""
    blob_xy = np.array([[b.cx, b.cy] for b in blobs])
    blob_intensities = np.array([b.intensity for b in blobs])
    max_intensity = max(blob_intensities.max(), 1.0)

    scores = np.zeros(len(cxs))
    for i in range(len(cxs)):
        dists = np.sqrt((blob_xy[:, 0] - cxs[i]) ** 2 + (blob_xy[:, 1] - cys[i]) ** 2)
        within = dists <= radius
        if not np.any(within):
            continue
        closest_idx = int(np.argmin(dists))
        proximity = 1.0 - min(dists[closest_idx] / radius, 1.0)
        intensity_norm = blob_intensities[closest_idx] / max_intensity
        scores[i] = 0.6 * proximity + 0.4 * intensity_norm
    return scores
