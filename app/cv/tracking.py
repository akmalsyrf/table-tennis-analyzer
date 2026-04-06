from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.cv.detection import Detection


@dataclass(frozen=True)
class TrackPoint:
    x: float
    y: float
    conf: float
    source: str = "yolo"


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


@dataclass
class _TrackerState:
    last: TrackPoint | None = None
    prev: TrackPoint | None = None
    miss_count: int = 0
    track: list[TrackPoint | None] = field(default_factory=list)

    def reset(self) -> None:
        self.last = None
        self.prev = None

    def accept(self, det: Detection) -> None:
        pt = TrackPoint(x=det.x, y=det.y, conf=det.conf, source=det.source)
        self.track.append(pt)
        self.prev = self.last
        self.last = pt
        self.miss_count = 0

    def reject(self, *, reacquire_cooldown: int) -> None:
        self.track.append(None)
        self.miss_count += 1
        if self.miss_count >= reacquire_cooldown:
            self.reset()

    def predicted_position(self) -> tuple[float, float]:
        assert self.last is not None
        px, py = self.last.x, self.last.y
        if self.prev is not None:
            px += self.last.x - self.prev.x
            py += self.last.y - self.prev.y
        return px, py


def track_positions(
    detections_by_frame: list[Detection | None],
    *,
    max_jump_px: float,
    reacquire_cooldown: int = 3,
) -> list[TrackPoint | None]:
    """
    Single-object tracker with velocity prediction and re-acquire cooldown.

    - Predict next position from last known velocity.
    - Accept detection if it is within `max_jump_px` of predicted position.
    - After losing track for `reacquire_cooldown` consecutive frames, reset
      the tracker so the next detection starts a fresh track segment
      (prevents "jumping" to a distant false positive immediately).
    """
    st = _TrackerState()

    for det in detections_by_frame:
        if det is None:
            st.reject(reacquire_cooldown=reacquire_cooldown)
            continue

        if st.last is None:
            st.accept(det)
            continue

        pred_x, pred_y = st.predicted_position()
        if _dist(det.x, det.y, pred_x, pred_y) <= max_jump_px:
            st.accept(det)
        else:
            st.reject(reacquire_cooldown=reacquire_cooldown)

    return st.track
