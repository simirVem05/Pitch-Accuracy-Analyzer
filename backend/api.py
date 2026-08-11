from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import run_backend


MAX_UPLOAD_BYTES = 60 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif"}


class AnalyzeResponse(BaseModel):
    metrics: Dict[str, Any]
    graph_points: List[Tuple[float, Optional[float]]]
    report: str


app = FastAPI(title="Pitch Accuracy Analyzer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# Deliberately sync: separation and pitch tracking are minutes of blocking CPU
# work, so this must run on the threadpool rather than the event loop.
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp, length=1024 * 1024)

        size = os.path.getsize(tmp_path)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is {size / 1e6:.0f} MB; limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )

        graph_points, metrics, report = run_backend(tmp_path)
        return AnalyzeResponse(metrics=metrics, graph_points=graph_points, report=report)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis failed. See server logs for details.")
    finally:
        file.file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
