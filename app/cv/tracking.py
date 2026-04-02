from __future__ import annotations

import math
from dataclasses import dataclass

from app.cv.detection import Detection


@dataclass(frozen=True)
class TrackPoint:
    x: float
    y: float
    conf: float


def _dist(a: TrackPoint, b: Detection) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def track_positions(
    detections_by_frame: list[Detection | None],
    max_jump_px: float = 80.0,
) -> list[TrackPoint | None]:
    """
    Very simple single-object tracker:
    - keep last known point
    - if next detection is within max_jump_px, accept it
    - otherwise treat as missing (None)

    Returns a list aligned with frames.
    """
    track: list[TrackPoint | None] = []
    last: TrackPoint | None = None

    for det in detections_by_frame:
        if det is None:
            track.append(None)
            continue

        if last is None:
            last = TrackPoint(x=det.x, y=det.y, conf=det.conf)
            track.append(last)
            continue

        if _dist(last, det) <= max_jump_px:
            last = TrackPoint(x=det.x, y=det.y, conf=det.conf)
            track.append(last)
        else:
            # Too far: likely false positive or lost track; mark missing.
            track.append(None)

    return track

