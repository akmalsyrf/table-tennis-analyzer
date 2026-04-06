#!/usr/bin/env python3
"""
Visual evaluation tool for the ball-detection pipeline.

Extracts evenly-spaced sample frames from a video and saves a side-by-side
diagnostic image for each frame showing:

  [Original + YOLO boxes] | [Frame-diff mask] | [Final detection]

Usage::

    python scripts/eval_detection.py uploads/my_video.mp4 --samples 30
    # Opens eval_out/<video_stem>/ with numbered PNG files.

Inspect the images to quickly spot false positives / false negatives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PIPELINE_TUNING  # noqa: E402
from app.cv.motion import MotionDetector  # noqa: E402
from app.cv.table_roi import detect_table_roi_frame  # noqa: E402


def _resize(frame: np.ndarray, max_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / float(w)
    return cv2.resize(frame, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _run_yolo_raw(yolo, frame_bgr: np.ndarray, conf_thresh: float):
    """Return all YOLO detections (not just winner) for visualisation."""
    results = []
    try:
        res = yolo.predict(frame_bgr, verbose=False)
        if not res:
            return results
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return results
        confs = boxes.conf.detach().cpu().numpy().astype(float)
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
        try:
            clses = boxes.cls.detach().cpu().numpy().astype(int)
        except Exception:
            clses = None
        for i in range(len(confs)):
            if confs[i] < conf_thresh * 0.5:
                continue
            cls_id = int(clses[i]) if clses is not None else -1
            results.append({
                "xyxy": xyxy[i],
                "conf": float(confs[i]),
                "cls": cls_id,
                "is_ball": cls_id == 32,
            })
    except Exception:
        pass
    return results


def _draw_yolo_panel(frame: np.ndarray, detections: list[dict], conf_thresh: float) -> np.ndarray:
    panel = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"].astype(int)
        is_ball = d["is_ball"]
        above_thresh = d["conf"] >= conf_thresh
        if is_ball and above_thresh:
            color = (0, 255, 0)  # green — sports ball above threshold
        elif is_ball:
            color = (0, 255, 255)  # yellow — sports ball below threshold
        elif above_thresh:
            color = (0, 0, 255)  # red — non-ball above threshold
        else:
            color = (128, 128, 128)  # gray — non-ball below threshold
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 1)
        label = f"c{d['cls']}:{d['conf']:.2f}"
        cv2.putText(panel, label, (x1, max(y1 - 4, 12)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return panel


def _draw_motion_panel(diff_mask: np.ndarray, blobs) -> np.ndarray:
    panel = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)
    for b in blobs:
        cv2.circle(panel, (int(b.cx), int(b.cy)), 6, (0, 255, 255), 1)
        cv2.putText(panel, f"a={b.area:.0f}", (int(b.cx) + 8, int(b.cy)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def _draw_result_panel(frame: np.ndarray, detection, roi) -> np.ndarray:
    panel = frame.copy()
    if roi is not None:
        cv2.rectangle(panel, (roi.table_x, roi.table_y),
                      (roi.table_x + roi.table_w, roi.table_y + roi.table_h),
                      (255, 255, 0), 1)
        cv2.putText(panel, "TABLE", (roi.table_x + 2, roi.table_y + 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    if detection is not None:
        cx, cy = int(detection.x), int(detection.y)
        cv2.circle(panel, (cx, cy), 8, (0, 255, 0), 2)
        cv2.putText(panel, f"{detection.source} {detection.conf:.2f}",
                     (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                     (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "NO DETECTION", (10, 24),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual ball-detection evaluator")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--samples", type=int, default=30, help="Number of frames to sample")
    parser.add_argument("--out", type=str, default="eval_out", help="Output directory")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Could not read frame count")
        sys.exit(1)

    t = PIPELINE_TUNING
    sample_indices = np.linspace(0, total_frames - 1, args.samples, dtype=int)

    out_dir = Path(args.out) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = None
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        print("YOLO loaded")
    except Exception as e:
        print(f"YOLO unavailable: {e}")

    motion = MotionDetector()
    prev_gray: np.ndarray | None = None

    from app.cv.detection import BallDetector
    detector = BallDetector(
        yolo_model="yolov8n.pt",
        yolo_conf=t.yolo_conf,
        max_box_area=t.max_box_area,
        min_box_area=t.min_box_area,
        motion_match_radius=t.motion_match_radius,
    )

    last_pos: tuple[float, float] | None = None

    for sample_idx, frame_no in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _resize(frame, t.max_width)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        yolo_dets = _run_yolo_raw(yolo, frame, t.yolo_conf) if yolo else []

        blobs = motion.detect(prev_gray, curr_gray) if prev_gray is not None else []
        diff_mask = cv2.absdiff(prev_gray, curr_gray) if prev_gray is not None else np.zeros_like(curr_gray)

        det = detector.detect(frame, last_pos=last_pos, prev_gray=prev_gray)
        if det is not None:
            last_pos = (det.x, det.y)

        roi = detect_table_roi_frame(frame, tuning=t.table_roi)

        panel_yolo = _draw_yolo_panel(frame, yolo_dets, t.yolo_conf)
        panel_motion = _draw_motion_panel(diff_mask, blobs)
        panel_result = _draw_result_panel(frame, det, roi)

        h = frame.shape[0]
        cv2.putText(panel_yolo, "YOLO candidates", (4, h - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel_motion, "Motion (frame diff)", (4, h - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel_result, "Final result", (4, h - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.hstack([panel_yolo, panel_motion, panel_result])

        label = f"Frame {frame_no}/{total_frames}"
        cv2.putText(combined, label, (10, 18),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = out_dir / f"{sample_idx:04d}_f{frame_no:06d}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  [{sample_idx + 1}/{args.samples}] frame {frame_no} → {out_path.name}")

        prev_gray = curr_gray

    cap.release()
    print(f"\nDone. {args.samples} diagnostic images saved to: {out_dir}/")


if __name__ == "__main__":
    main()
