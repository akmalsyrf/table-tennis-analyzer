from __future__ import annotations

import mimetypes
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.services.history_service import find_uploaded_video, list_history, load_result_for_display
from app.services.video_service import analyze_video

router = APIRouter(tags=["web"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


templates.env.filters["dt"] = _format_ts

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")


@router.get("/")
async def home(request: Request):
    """Upload form and short intro."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Upload"},
    )


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept multipart upload, run analysis, redirect to result page.
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

    analyze_video(video_path=dst_path, outputs_dir=OUTPUTS_DIR)
    return RedirectResponse(url=f"/results/{video_id}", status_code=303)


@router.get("/history")
async def history_page(request: Request):
    """List past analyses from ``outputs/*.json``."""
    items = list_history(OUTPUTS_DIR)
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"title": "History", "items": items},
    )


@router.get("/videos/{analysis_id}")
async def serve_analysis_video(analysis_id: str):
    """Stream the uploaded video file for playback (same id as analysis JSON)."""
    path = find_uploaded_video(UPLOADS_DIR, analysis_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Video not found")
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path,
        media_type=media_type or "video/mp4",
        filename=path.name,
    )


@router.get("/results/{analysis_id}")
async def result_page(request: Request, analysis_id: str):
    """Detail view for one analysis."""
    data = load_result_for_display(OUTPUTS_DIR, analysis_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    rallies = data.get("rallies") or []
    video_path = find_uploaded_video(UPLOADS_DIR, analysis_id)
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "title": f"Result · {analysis_id[:8]}…",
            "data": data,
            "rallies": rallies,
            "video_available": video_path is not None,
        },
    )
