from __future__ import annotations

import math
from dataclasses import dataclass

from app.cv.tracking import TrackPoint


@dataclass(frozen=True)
class Rally:
    start_frame: int
    end_frame: int
    hits: int
    duration_s: float


def _unit(vx: float, vy: float) -> tuple[float, float] | None:
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return None
    return (vx / n, vy / n)


def _direction_change(prev_v: tuple[float, float] | None, v: tuple[float, float] | None) -> float:
    """
    Return direction change magnitude in [0..2] (based on 1 - dot product).
    """
    if prev_v is None or v is None:
        return 0.0
    dot = max(-1.0, min(1.0, prev_v[0] * v[0] + prev_v[1] * v[1]))
    return 1.0 - dot


def compute_rallies(
    track: list[TrackPoint | None],
    fps: float,
    missing_end_frames: int = 12,
    direction_change_threshold: float = 0.65,
    min_frames_per_rally: int = 10,
) -> list[Rally]:
    """
    POC event logic:
    - Rally starts at first non-None track point.
    - Rally ends when the ball is missing for `missing_end_frames` consecutive frames.
    - Hit increments when direction changes significantly between consecutive velocity vectors.
    """
    rallies: list[Rally] = []

    in_rally = False
    start = 0
    last_seen = -1
    missing_run = 0
    hits = 0

    prev_pt: TrackPoint | None = None
    prev_v: tuple[float, float] | None = None

    for i, pt in enumerate(track):
        if pt is None:
            if in_rally:
                missing_run += 1
                if missing_run >= missing_end_frames:
                    end = last_seen if last_seen >= 0 else i
                    if end - start + 1 >= min_frames_per_rally:
                        duration = max(0.0, (end - start + 1) / max(fps, 1e-6))
                        rallies.append(Rally(start_frame=start, end_frame=end, hits=hits, duration_s=duration))
                    in_rally = False
                    prev_pt = None
                    prev_v = None
                    hits = 0
                    missing_run = 0
            continue

        # pt is present
        last_seen = i
        missing_run = 0

        if not in_rally:
            in_rally = True
            start = i
            prev_pt = pt
            prev_v = None
            hits = 0
            continue

        assert prev_pt is not None
        vx = pt.x - prev_pt.x
        vy = pt.y - prev_pt.y
        v = _unit(vx, vy)

        if _direction_change(prev_v, v) >= direction_change_threshold:
            hits += 1

        prev_v = v if v is not None else prev_v
        prev_pt = pt

    # flush if video ends mid-rally
    if in_rally:
        end = last_seen if last_seen >= 0 else len(track) - 1
        if end - start + 1 >= min_frames_per_rally:
            duration = max(0.0, (end - start + 1) / max(fps, 1e-6))
            rallies.append(Rally(start_frame=start, end_frame=end, hits=hits, duration_s=duration))

    return rallies

