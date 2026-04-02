from __future__ import annotations

import logging

from fastapi import FastAPI

from app.routes.analyze import router as analyze_router


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    app = FastAPI(title="Table Tennis Analyzer (POC)", version="0.1.0")
    app.include_router(analyze_router)
    return app


app = create_app()

