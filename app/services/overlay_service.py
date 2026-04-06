from __future__ import annotations

import logging
from pathlib import Path

import cv2

from app.cv.events import Rally
from app.cv.table_roi import TableROI
from app.cv.tracking import TrackPoint

logger = logging.getLogger(__name__)


def _build_rally_by_frame(track_len: int, rallies: list[Rally]) -> list[int]:
    rally_by_frame = [-1] * track_len
    for idx, r in enumerate(rallies):
        if r.end_frame < 0 or r.start_frame >= track_len:
            continue
        s = max(0, r.start_frame)
        e = min(track_len - 1, r.end_frame)
        for fi in range(s, e + 1):
            rally_by_frame[fi] = idx
    return rally_by_frame


def _resize_if_needed(frame, *, resize_max_width: int):
    if resize_max_width <= 0:
        return frame
    w = frame.shape[1]
    if w <= 0 or w <= resize_max_width:
        return frame
    h = frame.shape[0]
    scale = resize_max_width / float(w)
    return cv2.resize(frame, (resize_max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _timeline_x(*, x0: int, x1: int, frame_idx: int, total: int) -> int:
    denom = max(total - 1, 1)
    return int(x0 + (frame_idx / denom) * (x1 - x0))


def _draw_timeline_bar(*, frame, track_len: int, rally: Rally, track_i: int, seg_color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    y_bar = h - 18
    x0, x1 = 10, w - 10
    if x1 <= x0 + 10:
        return

    # background
    cv2.rectangle(frame, (x0, y_bar - 8), (x1, y_bar + 4), (40, 40, 40), -1)
    # active interval
    start_x = _timeline_x(x0=x0, x1=x1, frame_idx=rally.start_frame, total=track_len)
    end_x = _timeline_x(x0=x0, x1=x1, frame_idx=rally.end_frame, total=track_len)
    cv2.rectangle(frame, (start_x, y_bar - 8), (end_x, y_bar + 4), seg_color, -1)
    # current position marker
    cur_x = _timeline_x(x0=x0, x1=x1, frame_idx=track_i, total=track_len)
    cv2.line(frame, (cur_x, y_bar - 8), (cur_x, y_bar + 4), (255, 255, 255), 2)


def _draw_roi_debug(frame, roi: TableROI | None) -> None:
    """Draw table ROI boundary lines for debugging (to be removed later)."""
    if roi is None:
        return

    roi_color = (255, 0, 255)  # magenta — play area boundary
    table_color = (255, 255, 0)  # cyan — detected table surface

    # Play area ROI (outer boundary for detection filtering).
    cv2.rectangle(
        frame,
        (roi.x, roi.y),
        (roi.x + roi.w, roi.y + roi.h),
        roi_color,
        1,
    )
    cv2.putText(
        frame, "ROI", (roi.x + 4, roi.y + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, roi_color, 1, cv2.LINE_AA,
    )

    # Table surface rectangle.
    cv2.rectangle(
        frame,
        (roi.table_x, roi.table_y),
        (roi.table_x + roi.table_w, roi.table_y + roi.table_h),
        table_color,
        1,
    )
    cv2.putText(
        frame, "TABLE", (roi.table_x + 4, roi.table_y + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, table_color, 1, cv2.LINE_AA,
    )


def _draw_frame_overlay(
    *,
    frame,
    track_i: int,
    track_len: int,
    pt: TrackPoint | None,
    rally_idx: int,
    rallies: list[Rally],
    hits_by_rally: list[set[int]],
    table_roi: TableROI | None = None,
) -> None:
    _draw_roi_debug(frame, table_roi)

    ball_color = (0, 255, 255)  # yellow-ish
    seg_color = (0, 180, 255)  # orange-ish

    cx = cy = None
    if pt is not None:
        cx = int(round(pt.x))
        cy = int(round(pt.y))
        cv2.circle(frame, (cx, cy), 6, ball_color, -1)

    if rally_idx < 0:
        return

    rally = rallies[rally_idx]

    # Text label for current rally.
    cv2.putText(
        frame,
        f"Rally {rally_idx + 1}/{len(rallies)} | Hits {rally.hits}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        seg_color,
        2,
        cv2.LINE_AA,
    )

    _draw_timeline_bar(frame=frame, track_len=track_len, rally=rally, track_i=track_i, seg_color=seg_color)

    # Hit marker on frames where direction-change is inferred.
    if track_i in hits_by_rally[rally_idx]:
        cv2.putText(
            frame,
            "HIT",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        if cx is not None and cy is not None:
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)


def _open_video_writer(
    *,
    overlay_path: Path,
    fps_eff: float,
    out_w: int,
    out_h: int,
    codec_candidates: list[str],
) -> cv2.VideoWriter | None:
    """
    Try multiple codecs to maximize browser compatibility.

    Some OpenCV builds can write H.264 only when the proper codec is available.
    """
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(overlay_path), fourcc, float(fps_eff), (out_w, out_h))
        if writer.isOpened():
            logger.info("Overlay writer opened with codec=%s", codec)
            return writer
    return None


def _get_frame_roi(rois_by_frame: list[TableROI | None] | None, track_i: int) -> TableROI | None:
    if not rois_by_frame or track_i >= len(rois_by_frame):
        return None
    return rois_by_frame[track_i]


def _write_overlay_frames(
    *,
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter | None,
    track: list[TrackPoint | None],
    track_len: int,
    rally_by_frame: list[int],
    hits_by_rally: list[set[int]],
    rallies: list[Rally],
    rois_by_frame: list[TableROI | None] | None,
    frame_step: int,
    resize_max_width: int,
    overlay_path: Path,
    fps_eff: float,
    codec_candidates: list[str],
) -> tuple[cv2.VideoWriter | None, int, int]:
    frame_idx = 0
    track_i = 0

    while track_i < track_len:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        frame = _resize_if_needed(frame, resize_max_width=resize_max_width)

        if writer is None:
            out_h, out_w = frame.shape[:2]
            writer = _open_video_writer(
                overlay_path=overlay_path, fps_eff=fps_eff,
                out_w=out_w, out_h=out_h, codec_candidates=codec_candidates,
            )
            if writer is None:
                logger.warning("Could not open VideoWriter for overlay: %s", overlay_path)
                break

        _draw_frame_overlay(
            frame=frame,
            track_i=track_i,
            track_len=track_len,
            pt=track[track_i],
            rally_idx=rally_by_frame[track_i],
            rallies=rallies,
            hits_by_rally=hits_by_rally,
            table_roi=_get_frame_roi(rois_by_frame, track_i),
        )
        writer.write(frame)
        track_i += 1
        frame_idx += 1

    return writer, frame_idx, track_i


def generate_rally_hit_overlay_video(
    *,
    video_path: Path,
    overlay_path: Path,
    track: list[TrackPoint | None],
    rallies: list[Rally],
    fps_eff: float,
    frame_step: int,
    resize_max_width: int,
    rois_by_frame: list[TableROI | None] | None = None,
) -> None:
    """
    Generate an overlay MP4 where we draw:
    - current rally interval (start/end as a timeline bar)
    - current rally id and hit count
    - individual inferred hit frames (based on direction-change events)
    - tracked ball center

    Note: This overlay is generated on the same sampled timeline as `track`.
    """
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video for overlay: %s", video_path)
        return

    track_len = len(track)
    if track_len == 0:
        cap.release()
        return

    rally_by_frame = _build_rally_by_frame(track_len=track_len, rallies=rallies)
    hits_by_rally = [set(r.hit_frames) for r in rallies]

    writer: cv2.VideoWriter | None = None
    codec_candidates = ["avc1", "H264", "X264", "mp4v"]
    writer, _, _ = _write_overlay_frames(
        cap=cap,
        writer=writer,
        track=track,
        track_len=track_len,
        rally_by_frame=rally_by_frame,
        hits_by_rally=hits_by_rally,
        rallies=rallies,
        rois_by_frame=rois_by_frame,
        frame_step=frame_step,
        resize_max_width=resize_max_width,
        overlay_path=overlay_path,
        fps_eff=fps_eff,
        codec_candidates=codec_candidates,
    )

    cap.release()
    if writer is not None:
        writer.release()

