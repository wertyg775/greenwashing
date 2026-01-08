# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import greenwashing_analyzer
import torch
from greenwashing_analyzer import GreenwashingAnalyzer
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = GreenwashingAnalyzer(
    specificity_model_path="./models/specificity"
)

class AnalysisRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    return analyzer.analyze(request.text)


@app.get("/health")
async def health():
    return {"status": "ok"}