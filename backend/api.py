from __future__ import annotations

import os
import tempfile
from typing import Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import analyze_vocals
from .config import AnalyzerConfig, Genre


app = FastAPI(title="Pitch Accuracy Analyzer API", version="0.1.0")

# --- CORS (for React/Vite dev server) ---
# Update origins as needed (Vite default is http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    summary: dict
    graph: dict
    gemini_prompt: str


def _build_config(
    strictness: float,
    confidence_threshold: float,
    step_size_ms: int,
    model_capacity: str,
    viterbi: bool,
    a4_hz: float,
) -> AnalyzerConfig:
    # Keep your defaults but allow overrides
    return AnalyzerConfig(
        strictness=strictness,
        confidence_threshold=confidence_threshold,
        step_size_ms=step_size_ms,
        model_capacity=model_capacity,
        viterbi=viterbi,
        a4_hz=a4_hz,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    key: str = Form(...),
    genre: Genre = Form(...),

    # Optional overrides (match your AnalyzerConfig knobs)
    strictness: float = Form(0.6),
    confidence_threshold: float = Form(0.25),  # recommended for tiny
    step_size_ms: int = Form(20),
    model_capacity: str = Form("tiny"),
    viterbi: bool = Form(True),
    a4_hz: float = Form(440.0),
):
    # Basic validation
    if strictness < 0 or strictness > 1:
        raise HTTPException(status_code=422, detail="strictness must be between 0 and 1")

    # Only allow safe model capacities
    if model_capacity not in {"tiny", "small", "medium", "large", "full"}:
        raise HTTPException(status_code=422, detail="model_capacity must be one of tiny|small|medium|large|full")

    # Save upload to a temp file (librosa/crepe expects a path or array; your pipeline uses path)
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        # librosa can handle more formats depending on backend, but these are common
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        cfg = _build_config(
            strictness=float(strictness),
            confidence_threshold=float(confidence_threshold),
            step_size_ms=int(step_size_ms),
            model_capacity=str(model_capacity),
            viterbi=bool(viterbi),
            a4_hz=float(a4_hz),
        )

        result = analyze_vocals(
            wav_path=tmp_path,
            declared_key=key,
            genre=genre,
            config=cfg,
        )

        return AnalyzeResponse(
            summary=result.summary,
            graph=result.graph,
            gemini_prompt=result.gemini_prompt,
        )

    except Exception as e:
        # You can log e here
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
