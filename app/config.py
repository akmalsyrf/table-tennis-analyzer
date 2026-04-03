"""
Central defaults for pipeline tuning knobs (see docs/flow.md § "Parameter yang paling sering di-tuning").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineTuning:
    target_fps: float = 10.0
    max_width: int = 640
    yolo_conf: float = 0.25
    max_jump_px: float = 120.0
    # Tunable analogue to `missing_end_frames` at a given effective FPS (sampled timeline).
    missing_end_seconds: float = 0.8
    missing_end_min_frames: int = 4
    direction_change_threshold: float = 0.55

    def missing_end_frames(self, fps_effective: float) -> int:
        return max(
            self.missing_end_min_frames,
            int(round(self.missing_end_seconds * fps_effective)),
        )


PIPELINE_TUNING = PipelineTuning()
