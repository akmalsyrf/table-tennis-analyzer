from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryItem:
    """Summary row for the history list (no full track payload)."""

    id: str
    video_filename: str
    total_rallies: int
    fps_native: float | None
    fps_effective: float | None
    num_frames: int | None
    num_detections: int | None
    modified_ts: float


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def list_history(outputs_dir: Path, limit: int = 200) -> list[HistoryItem]:
    """
    Scan ``outputs/*.json`` (newest first) and return lightweight metadata.
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        return []

    items: list[HistoryItem] = []
    paths = sorted(outputs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    for p in paths[:limit]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable output: %s", p)
            continue

        video_path = str(data.get("video") or "")
        video_filename = Path(video_path).name if video_path else p.stem + ".mp4"
        debug = data.get("debug") or {}
        st = p.stat()
        items.append(
            HistoryItem(
                id=p.stem,
                video_filename=video_filename,
                total_rallies=_safe_int(data.get("total_rallies")) or 0,
                fps_native=_safe_float(data.get("fps_native")),
                fps_effective=_safe_float(data.get("fps_effective")),
                num_frames=_safe_int(debug.get("num_frames")),
                num_detections=_safe_int(debug.get("num_detections")),
                modified_ts=st.st_mtime,
            )
        )

    return items


def load_result_for_display(outputs_dir: Path, analysis_id: str) -> dict[str, Any] | None:
    """
    Load one output JSON and strip heavy fields (e.g. per-frame track) for templates.
    """
    if not re.fullmatch(r"[0-9a-f]{32}", analysis_id):
        return None
    path = Path(outputs_dir) / f"{analysis_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    data = dict(data)
    data.pop("track", None)
    data["id"] = analysis_id
    data["video_filename"] = Path(str(data.get("video") or "")).name or f"{analysis_id}.mp4"
    return data


def find_uploaded_video(uploads_dir: Path, analysis_id: str) -> Path | None:
    """
    Locate ``uploads/<analysis_id>.<ext>`` for a valid hex id (one file expected).
    """
    if not re.fullmatch(r"[0-9a-f]{32}", analysis_id):
        return None
    uploads_dir = Path(uploads_dir)
    if not uploads_dir.is_dir():
        return None
    matches = list(uploads_dir.glob(f"{analysis_id}.*"))
    files = [p for p in matches if p.is_file()]
    if not files:
        return None
    return files[0]
