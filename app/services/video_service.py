from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import cv2

from app.cv.detection import BallDetector, Detection
from app.cv.events import compute_rallies
from app.cv.tracking import TrackPoint, track_positions
from app.models.schema import AnalyzeResponse, RallyStats
from app.services.overlay_service import generate_rally_hit_overlay_video

logger = logging.getLogger(__name__)


def analyze_video(video_path: Path, outputs_dir: Path) -> AnalyzeResponse:
    """
    End-to-end POC analysis:
    - read frames
    - detect ball center positions
    - track positions across frames
    - compute rallies and hits with simple heuristics
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return AnalyzeResponse(total_rallies=0, rallies=[])

    fps_native = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # POC speed knobs (kept simple and conservative).
    target_fps = 10.0
    frame_step = max(1, int(round(max(fps_native, 1.0) / target_fps)))
    fps_eff = fps_native / frame_step

    max_width = 640  # resize for faster detection on CPU

    detector = BallDetector(yolo_model="yolov8n.pt", yolo_conf=0.25, prefer_coco_sports_ball=True)

    detections_by_frame: list[Detection | None] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        if max_width > 0:
            h, w = frame.shape[:2]
            if w > max_width and w > 0:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

        det = detector.detect(frame)
        detections_by_frame.append(det)
        frame_idx += 1

    cap.release()

    track: list[TrackPoint | None] = track_positions(
        detections_by_frame=detections_by_frame,
        # With lower FPS and resized frames, allow a bit more jump tolerance.
        max_jump_px=120.0,
    )

    rallies = compute_rallies(
        track=track,
        fps=fps_eff,
        # Missing threshold expressed in "effective frames": ~0.8s of missing ends a rally.
        missing_end_frames=max(4, int(round(0.8 * fps_eff))),
        direction_change_threshold=0.55,
        # Require ~1.0s minimum rally length at effective FPS.
        min_frames_per_rally=max(6, int(round(1.0 * fps_eff))),
    )

    # Generate an offline overlay video (rally + hit markers) for easier review.
    try:
        overlay_path = outputs_dir / "overlays" / f"{video_path.stem}.mp4"
        generate_rally_hit_overlay_video(
            video_path=video_path,
            overlay_path=overlay_path,
            track=track,
            rallies=rallies,
            fps_eff=fps_eff,
            frame_step=frame_step,
            resize_max_width=max_width,
        )
    except Exception:
        logger.exception("Failed to generate overlay video for %s", video_path)

    response = AnalyzeResponse(
        total_rallies=len(rallies),
        rallies=[RallyStats(hits=r.hits, duration=r.duration_s) for r in rallies],
    )

    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = outputs_dir / f"{video_path.stem}.json"
        payload = {
            "video": str(video_path),
            "fps_native": fps_native,
            "fps_effective": fps_eff,
            "frame_step": frame_step,
            "resize_max_width": max_width,
            "total_rallies": response.total_rallies,
            "rallies": [r.model_dump() for r in response.rallies],
            "debug": {
                "num_frames": len(detections_by_frame),
                "num_detections": sum(1 for d in detections_by_frame if d is not None),
                "num_tracked": sum(1 for p in track if p is not None),
            },
            "track": [asdict(p) if p is not None else None for p in track],
        }
        out_path.write_text(json.dumps(payload, indent=2))
    except Exception:
        logger.exception("Failed to write outputs JSON")

    return response

