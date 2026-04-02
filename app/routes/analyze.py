from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.models.schema import AnalyzeResponse
from app.services.video_service import analyze_video

router = APIRouter()

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Upload a video, run a simple CV pipeline, and return basic rally/hit stats.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "").suffix or ".mp4"
    video_id = uuid.uuid4().hex
    dst_path = UPLOADS_DIR / f"{video_id}{suffix}"

    with dst_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    result = analyze_video(video_path=dst_path, outputs_dir=OUTPUTS_DIR)
    return result

