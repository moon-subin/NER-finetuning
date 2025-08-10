# src/serving/api.py
# (선택) FastAPI 엔드포인트
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json, os
from ..inference.predict import predict_main  # or implement inline for API
# This is a placeholder to show structure. Implement actual predict function for API if needed.

app = FastAPI(title="NER Serving (KoELECTRA+CRF)")

class PredictIn(BaseModel):
    tokens: List[str]

@app.get("/health")
def health():
    return {"ok": True}
