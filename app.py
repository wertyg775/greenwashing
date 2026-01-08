# backend/app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from greenwashing_analyzer import GreenwashingAnalyzer
from fastapi.staticfiles import StaticFiles


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

@app.post("/api/analyze_additive")
async def analyze_additive(request: AnalysisRequest):
    return analyzer.analyze_additive(request.text)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
