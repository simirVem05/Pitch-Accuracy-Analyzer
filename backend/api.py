from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import run_backend

class AnalyzeResponse(BaseModel):
    metrics: Dict[str, Any]
    graph_tuples: List[Tuple[float, float]]
    report: str

app = FastAPI(title="Pitch Accuracy Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Pitch Accuracy Analyzer API. Visit /docs"}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status" : "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    key: str = Form(...),
    genre: str = Form(...),
) -> AnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    
    if not key or len(key.strip()) < 3:
        raise HTTPException(status_code=400, detail='Invalid key. Ex: B minor')
    if not genre or len(genre.strip()) < 2:
        raise HTTPException(status_code=400, detail='Invalid genre. Ex: rnb')
    
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(content)
        graph_tuples, metrics, report = run_backend(tmp_path, key, genre)

        return AnalyzeResponse(
            metrics=metrics,
            graph_tuples=graph_tuples,
            report=report,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass