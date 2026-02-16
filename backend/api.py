from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, List, Tuple, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import run_backend

class AnalyzeResponse(BaseModel):
    metrics: Dict[str, Any]
    graph_tuples: List[Tuple[float, Optional[float]]] # Allows null for graph breaks
    report: str

app = FastAPI(title="Pitch Accuracy Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), key: str = Form(...), genre: str = Form(...)) -> AnalyzeResponse:
    if not file.filename: raise HTTPException(status_code=400, detail="Missing filename.")
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
        
        graph_tuples, metrics, report = run_backend(tmp_path, key, genre)
        return AnalyzeResponse(metrics=metrics, graph_tuples=graph_tuples, report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path): os.remove(tmp_path)