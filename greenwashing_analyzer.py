# greenwashing_analyzer.py

import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

class GreenwashingAnalyzer:
    def __init__(self, specificity_model_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load specificity model
        self.tokenizer = BertTokenizer.from_pretrained(specificity_model_path)
        self.model = BertForSequenceClassification.from_pretrained(specificity_model_path)
        self.model.to(self.device)
        self.model.eval()

        # Sentiment model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if self.device.type != "cpu" else -1
        )

        # Buzzwords + weights
        self.buzzwords = {
            "eco-friendly": 1.2,
            "sustainable": 1.2,
            "green": 1.1,
            "natural": 1.1,
            "planet-friendly": 1.3
        }

    # ---------- Sentence Split ----------
    def split_sentences(self, text):
        text = re.sub(r'[\r\n]+', '. ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= 15]

    # ---------- Specificity ----------
    def predict_specificity(self, sentences):
        if not sentences:
            return 0.5  # neutral fallback

        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        specific_probs = probs[:, 1]  # label 0 = VAGUE
        return specific_probs.mean().item()

    # ---------- Sentiment ----------
    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text[:512])[0]
        stars = int(result["label"][0])
        return (stars - 1) / 4  # normalize to 0–1

    # ---------- Buzzword Weight ----------
    def compute_buzzword_weight(self, text):
        text_lower = text.lower()

        vague_terms = [
            'eco-friendly', 'green', 'sustainable', 'natural',
            'earth-conscious', 'mesra alam', 'hijau'
        ]

        certifications = [
            'gots', 'fsc', 'fair trade', 'usda', 'grs'
        ]

        vague_count = sum(term in text_lower for term in vague_terms)
        has_cert = any(cert in text_lower for cert in certifications)
        has_metrics = bool(re.search(r'\d+\s*%|\d+\s*(kg|g|grams)', text))

        weight = 1.0

        if vague_count > 0:
            weight += 0.15 * vague_count

        if not has_cert:
            weight += 0.2

        if not has_metrics:
            weight += 0.1

        return min(weight, 2.0)

    # ---------- Main composite analysis ----------
    def analyze(self, text):
        sentences = self.split_sentences(text)

        specificity = self.predict_specificity(sentences)
        sentiment = self.analyze_sentiment(text)
        buzz_weight = self.compute_buzzword_weight(text)

        risk_score = (1-specificity) * sentiment * buzz_weight

        label = "greenwashing" if risk_score >= 0.4 else "not_greenwashing"

        return {
            "risk_score": round(risk_score, 3),
            "label": label,
            "specificity": round(specificity, 3),
            "sentiment": round(sentiment, 3),
            "buzzword_weight": round(buzz_weight, 2)
        }
        # ---------- Alternative composite analysis (Additive risk) ----------
    def analyze_additive(self, text, alpha=0.5, beta=0.3, gamma=0.2):
        """
        Alternative risk formulation using additive weighted components.
        """

        sentences = self.split_sentences(text)

        specificity = self.predict_specificity(sentences)
        sentiment = self.analyze_sentiment(text)
        buzz_weight = self.compute_buzzword_weight(text)

        # Normalize buzzword weight to 0–1
        buzz_norm = min(buzz_weight / 2.0, 1.0)

        risk_score = (
            alpha * (1 - specificity) +
            beta * sentiment +
            gamma * buzz_norm
        )

        risk_score = min(max(risk_score, 0.0), 1.0)

        label = "greenwashing" if risk_score >= 0.4 else "not_greenwashing"

        return {
            "risk_score": round(risk_score, 3),
            "label": label,
            "specificity": round(specificity, 3),
            "sentiment": round(sentiment, 3),
            "buzzword_weight": round(buzz_weight, 2),
            "formula": "additive"
        }

    # ---------- Specificity-only scoring ----------
    def analyze_specificity_only(self, text):
        sentences = self.split_sentences(text)
        specificity = self.predict_specificity(sentences)
        specificity_score = 1 - specificity

        label = "greenwashing" if specificity_score >= 0.4 else "not_greenwashing"

        return {
            "specificity_score": round(specificity_score, 3),
            "label": label
        }