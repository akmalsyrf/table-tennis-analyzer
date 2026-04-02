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

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    detector = BallDetector()

    detections_by_frame: list[Detection | None] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = detector.detect(frame)
        detections_by_frame.append(det)
        frame_idx += 1

    cap.release()

    track: list[TrackPoint | None] = track_positions(
        detections_by_frame=detections_by_frame,
        max_jump_px=80.0,
    )

    rallies = compute_rallies(
        track=track,
        fps=fps,
        missing_end_frames=12,
        direction_change_threshold=0.65,
        min_frames_per_rally=10,
    )

    response = AnalyzeResponse(
        total_rallies=len(rallies),
        rallies=[RallyStats(hits=r.hits, duration=r.duration_s) for r in rallies],
    )

    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = outputs_dir / f"{video_path.stem}.json"
        payload = {
            "video": str(video_path),
            "fps": fps,
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

