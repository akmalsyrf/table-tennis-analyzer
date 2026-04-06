from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import cv2

from app.config import PIPELINE_TUNING
from app.cv.detection import BallDetector, Detection
from app.cv.events import compute_rallies
from app.cv.table_roi import TableROI  # detect_table_roi_frame disabled
from app.cv.tracking import TrackPoint, track_positions
from app.models.schema import AnalyzeResponse, RallyStats
from app.services.overlay_service import generate_rally_hit_overlay_video

logger = logging.getLogger(__name__)


def _resize_frame_if_needed(frame, max_width: int):
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width or w <= 0:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _roi_to_dict(roi: TableROI) -> dict:
    return {
        "x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h,
        "table_x": roi.table_x, "table_y": roi.table_y,
        "table_w": roi.table_w, "table_h": roi.table_h,
    }


def _build_track_entry(
    pt: TrackPoint | None,
    roi: TableROI | None,
) -> dict | None:
    if pt is None:
        return None
    entry = asdict(pt)
    if roi is not None:
        entry["roi"] = _roi_to_dict(roi)
    else:
        entry["roi"] = None
    return entry


def _collect_detections_and_rois(
    cap: cv2.VideoCapture,
    *,
    frame_step: int,
    max_width: int,
    detector: BallDetector,
) -> tuple[list[Detection | None], list[TableROI | None]]:
    """
    For each sampled frame:
    - detect table ROI (metadata only)
    - detect ball via combined YOLO + frame differencing
    - feed last known ball position back for proximity scoring
    """
    detections: list[Detection | None] = []
    rois: list[TableROI | None] = []
    last_pos: tuple[float, float] | None = None
    prev_gray: cv2.typing.MatLike | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        frame = _resize_frame_if_needed(frame, max_width)

        roi = None  # ROI detection disabled: both HSV and edge-based produce noise
        rois.append(roi)

        det = detector.detect(frame, last_pos=last_pos, prev_gray=prev_gray)
        detections.append(det)
        if det is not None:
            last_pos = (det.x, det.y)

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

    return detections, rois


def analyze_video(video_path: Path, outputs_dir: Path) -> AnalyzeResponse:
    """
    End-to-end POC analysis:
    - per-frame table ROI detection
    - detect ball center positions (filtered by per-frame ROI)
    - track positions across frames
    - compute rallies and hits with simple heuristics
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return AnalyzeResponse(total_rallies=0, rallies=[])

    fps_native = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    t = PIPELINE_TUNING
    frame_step = max(1, int(round(max(fps_native, 1.0) / t.target_fps)))
    fps_eff = fps_native / frame_step

    detector = BallDetector(
        yolo_model="yolov8n.pt",
        yolo_conf=t.yolo_conf,
        prefer_coco_sports_ball=True,
        max_box_area=t.max_box_area,
        min_box_area=t.min_box_area,
        motion_match_radius=t.motion_match_radius,
    )

    detections_by_frame, rois_by_frame = _collect_detections_and_rois(
        cap,
        frame_step=frame_step,
        max_width=t.max_width,
        detector=detector,
    )

    cap.release()

    track: list[TrackPoint | None] = track_positions(
        detections_by_frame=detections_by_frame,
        max_jump_px=t.max_jump_px,
        reacquire_cooldown=t.reacquire_cooldown,
    )

    rallies = compute_rallies(
        track=track,
        fps=fps_eff,
        missing_end_frames=t.missing_end_frames(fps_eff),
        direction_change_threshold=t.direction_change_threshold,
        min_frames_per_rally=max(6, int(round(1.0 * fps_eff))),
    )

    # Generate overlay video with per-frame ROI debug lines.
    try:
        overlay_path = outputs_dir / "overlays" / f"{video_path.stem}.mp4"
        generate_rally_hit_overlay_video(
            video_path=video_path,
            overlay_path=overlay_path,
            track=track,
            rallies=rallies,
            fps_eff=fps_eff,
            frame_step=frame_step,
            resize_max_width=t.max_width,
            rois_by_frame=rois_by_frame,
        )
    except Exception:
        logger.exception("Failed to generate overlay video for %s", video_path)

    response = AnalyzeResponse(
        total_rallies=len(rallies),
        rallies=[RallyStats(hits=r.hits, duration=r.duration_s) for r in rallies],
    )

    # Write full JSON output with per-frame track + ROI.
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = outputs_dir / f"{video_path.stem}.json"
        track_entries = [
            _build_track_entry(track[i], rois_by_frame[i] if i < len(rois_by_frame) else None)
            for i in range(len(track))
        ]
        payload = {
            "video": str(video_path),
            "fps_native": fps_native,
            "fps_effective": fps_eff,
            "frame_step": frame_step,
            "resize_max_width": t.max_width,
            "total_rallies": response.total_rallies,
            "rallies": [r.model_dump() for r in response.rallies],
            "debug": {
                "num_frames": len(detections_by_frame),
                "num_detections": sum(1 for d in detections_by_frame if d is not None),
                "num_tracked": sum(1 for p in track if p is not None),
            },
            "track": track_entries,
        }
        out_path.write_text(json.dumps(payload, indent=2))
    except Exception:
        logger.exception("Failed to write outputs JSON")

    return response
