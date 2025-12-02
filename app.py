# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = AutoModelForSequenceClassification.from_pretrained("./models/specificity")
tokenizer = AutoTokenizer.from_pretrained("./models/specificity")

# Add after loading specificity model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

class AnalysisRequest(BaseModel):
    text: str

# Modify analyze endpoint
@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    # Module 1: Specificity
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    
    prediction = "SPECIFIC" if torch.argmax(probs).item() == 1 else "VAGUE"
    confidence = probs.max().item()
    specificity_points = confidence * 40 if prediction == "VAGUE" else 0
    
    # Module 2: Sentiment
    sentiment_result = sentiment_analyzer(request.text[:512])[0]
    stars = int(sentiment_result['label'][0])  # "1 star" -> 1
    sentiment_score = (stars - 1) / 4  # normalize to 0-1
    
    # High sentiment + vague = manipulation
    if prediction == "VAGUE" and sentiment_score > 0.6:
        sentiment_points = sentiment_score * 30
    else:
        sentiment_points = 0
    
    total_score = specificity_points + sentiment_points
    
    if total_score >= 50:
        risk_level = "HIGH"
    elif total_score >= 25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        "text": request.text,
        "total_score": round(total_score, 1),
        "risk_level": risk_level,
        "specificity": {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "points": round(specificity_points, 1)
        },
        "sentiment": {
            "score": round(sentiment_score, 3),
            "points": round(sentiment_points, 1)
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}