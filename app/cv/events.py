from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.cv.tracking import TrackPoint


@dataclass(frozen=True)
class Rally:
    start_frame: int
    end_frame: int
    hits: int
    duration_s: float
    # Frame indices (on the "effective" sampled timeline) where a "hit" is inferred.
    hit_frames: tuple[int, ...]


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


def _append_rally_if_long_enough(
    rallies: list[Rally],
    *,
    start: int,
    end: int,
    hits: int,
    hit_frames: list[int],
    fps: float,
    min_frames_per_rally: int,
) -> None:
    if end - start + 1 < min_frames_per_rally:
        return
    duration = max(0.0, (end - start + 1) / max(fps, 1e-6))
    rallies.append(
        Rally(
            start_frame=start,
            end_frame=end,
            hits=hits,
            duration_s=duration,
            hit_frames=tuple(hit_frames),
        )
    )


@dataclass
class _RallyScan:
    rallies: list[Rally] = field(default_factory=list)
    in_rally: bool = False
    start: int = 0
    last_seen: int = -1
    missing_run: int = 0
    hits: int = 0
    hit_frames: list[int] = field(default_factory=list)
    prev_pt: TrackPoint | None = None
    prev_v: tuple[float, float] | None = None

    def reset_open_rally(self) -> None:
        self.in_rally = False
        self.prev_pt = None
        self.prev_v = None
        self.hits = 0
        self.hit_frames.clear()
        self.missing_run = 0

    def begin_rally(self, frame_i: int, pt: TrackPoint) -> None:
        self.in_rally = True
        self.start = frame_i
        self.prev_pt = pt
        self.prev_v = None
        self.hits = 0
        self.hit_frames.clear()

    def on_missing(
        self,
        frame_i: int,
        *,
        missing_end_frames: int,
        fps: float,
        min_frames_per_rally: int,
    ) -> None:
        if not self.in_rally:
            return
        self.missing_run += 1
        if self.missing_run < missing_end_frames:
            return
        end = self.last_seen if self.last_seen >= 0 else frame_i
        _append_rally_if_long_enough(
            self.rallies,
            start=self.start,
            end=end,
            hits=self.hits,
            hit_frames=self.hit_frames,
            fps=fps,
            min_frames_per_rally=min_frames_per_rally,
        )
        self.reset_open_rally()

    def on_present(
        self,
        frame_i: int,
        pt: TrackPoint,
        *,
        direction_change_threshold: float,
    ) -> None:
        self.last_seen = frame_i
        self.missing_run = 0
        if not self.in_rally:
            self.begin_rally(frame_i, pt)
            return
        assert self.prev_pt is not None
        vx = pt.x - self.prev_pt.x
        vy = pt.y - self.prev_pt.y
        v = _unit(vx, vy)
        if _direction_change(self.prev_v, v) >= direction_change_threshold:
            self.hits += 1
            self.hit_frames.append(frame_i)
        self.prev_v = v if v is not None else self.prev_v
        self.prev_pt = pt


def compute_rallies(
    track: list[TrackPoint | None],
    fps: float,
    *,
    missing_end_frames: int,
    direction_change_threshold: float,
    min_frames_per_rally: int = 10,
) -> list[Rally]:
    """
    POC event logic:
    - Rally starts at first non-None track point.
    - Rally ends when the ball is missing for `missing_end_frames` consecutive frames.
    - Hit increments when direction changes significantly between consecutive velocity vectors.
    """
    scan = _RallyScan()
    for i, pt in enumerate(track):
        if pt is None:
            scan.on_missing(
                i,
                missing_end_frames=missing_end_frames,
                fps=fps,
                min_frames_per_rally=min_frames_per_rally,
            )
            continue
        scan.on_present(i, pt, direction_change_threshold=direction_change_threshold)

    if scan.in_rally:
        end = scan.last_seen if scan.last_seen >= 0 else len(track) - 1
        _append_rally_if_long_enough(
            scan.rallies,
            start=scan.start,
            end=end,
            hits=scan.hits,
            hit_frames=scan.hit_frames,
            fps=fps,
            min_frames_per_rally=min_frames_per_rally,
        )

    return scan.rallies
