"""
Central defaults for pipeline tuning knobs (see docs/flow.md § "Parameter yang paling sering di-tuning").

Tuned for broadcast table-tennis footage (e.g. wide-angle match recordings)
where the ball is small and fast-moving.

Table ROI colour/geometry/morphology: edit ``TableRoiTuning`` (``PIPELINE_TUNING.table_roi``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# (H, S, V) lower / upper for cv2.inRange — multiple ranges = blue + green ITTF tables.
HsvRangePair = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass(frozen=True)
class TableRoiTuning:
    """
    Knobs for per-frame table segmentation (HSV mask + contour scoring + ROI expansion).
    Tune here rather than editing app/cv/table_roi.py.
    """

    # --- play-area ROI built from detected table rect ---
    vertical_expand: float = 1.8
    horizontal_margin: float = 0.12

    # --- colour mask (ITTF dark blue + dark green) ---
    hsv_ranges: Tuple[HsvRangePair, ...] = (
        ((90, 40, 40), (130, 255, 255)),
        ((35, 40, 40), (85, 255, 255)),
    )

    # --- morphology on combined mask (odd sizes typical; min 1) ---
    morph_close_kernel: int = 3
    morph_open_kernel: int = 3

    # --- bbox area as fraction of frame ---
    area_frac_min: float = 0.01
    area_frac_max: float = 0.25
    size_target_frac: float = 0.06
    size_score_scale: float = 0.06

    # --- width / height of table bbox in perspective ---
    aspect_min: float = 2.0
    aspect_max: float = 12.0
    ideal_aspect: float = 4.5

    # --- vertical position of bbox centre (normalised 0..1 from top) ---
    vy_min: float = 0.30
    vy_max: float = 0.85
    ideal_vy: float = 0.58
    vy_score_scale: float = 0.27

    # --- contour solidity (area / bbox area); rejects irregular blobs ---
    solidity_min: float = 0.35

    # --- minimum sum of sub-scores (size+aspect+h+vy+solidity) to accept a candidate ---
    min_total_score: float = 2.0


@dataclass(frozen=True)
class PipelineTuning:
    # --- sampling ---
    target_fps: float = 15.0
    max_width: int = 960

    # --- ball detection (YOLO + motion) ---
    yolo_conf: float = 0.30
    max_box_area: float = 5000.0
    min_box_area: float = 4.0
    motion_match_radius: float = 40.0

    # --- tracking ---
    max_jump_px: float = 100.0
    reacquire_cooldown: int = 4

    # --- table ROI (per-frame colour + geometry; metadata only, not used for detection filtering) ---
    table_roi: TableRoiTuning = field(default_factory=TableRoiTuning)

    # --- rally event detection ---
    missing_end_seconds: float = 0.8
    missing_end_min_frames: int = 4
    direction_change_threshold: float = 0.55

    def missing_end_frames(self, fps_effective: float) -> int:
        return max(
            self.missing_end_min_frames,
            int(round(self.missing_end_seconds * fps_effective)),
        )


PIPELINE_TUNING = PipelineTuning()
