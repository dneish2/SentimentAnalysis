"""FastAPI application wiring for housing and sentiment endpoints."""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from analytics.price_stats import PriceStats
from data.listings_repository import ListingsRepository
from services.house_avg_service import HouseAvgService
from services.sentiment.infer import infer_sentiment

CSV_PATH = os.getenv("LISTINGS_CSV", "data/listings.csv")
SENTI_CKPT = os.getenv("SENTI_CKPT", "checkpoints/sentiment")

repository = ListingsRepository(CSV_PATH)
stats = PriceStats()
house_avg = HouseAvgService(repository, stats)

app = FastAPI(title="Neighborhood Pricing & Sentiment", version="0.2.0")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "version": "0.2.0"}


class HouseAvgIn(BaseModel):
    neighborhood: str
    bedrooms: int = Field(ge=0, le=10)
    bathrooms: float = Field(ge=0, le=10)
    sqft: int = Field(ge=100, le=20000)
    bootstrap_samples: int = Field(1000, ge=100, le=10000)


@app.post("/house_avg")
def house_avg_ep(req: HouseAvgIn) -> Dict[str, Any]:
    response = house_avg.estimate(
        neighborhood=req.neighborhood,
        bedrooms=req.bedrooms,
        bathrooms=req.bathrooms,
        sqft=req.sqft,
        bootstrap_samples=req.bootstrap_samples,
    )
    if "error" in response:
        raise HTTPException(status_code=404, detail=response["error"])
    return response


class SentimentIn(BaseModel):
    input: str
    k: int = Field(3, ge=1, le=10)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


@app.post("/sentiment")
def sentiment_ep(req: SentimentIn) -> Dict[str, Any]:
    articles = infer_sentiment(req.input, k=req.k, threshold=req.threshold, ckpt_dir=SENTI_CKPT)
    pos = sum(1 for art in articles if art["label"] == "positive")
    neg = sum(1 for art in articles if art["label"] == "negative")
    mixed = sum(1 for art in articles if art["label"] == "mixed")
    dominant = max(("positive", pos), ("negative", neg), ("mixed", mixed), key=lambda item: item[1])[0]

    return {
        "summary": {
            "dominant": dominant,
            "rationale": f"{pos} positive; {neg} negative; {mixed} mixed",
        },
        "articles": [
            {
                "url": article.get("url"),
                "label": article["label"],
                "prob_pos": round(float(article.get("pos", 0.0)), 3),
                "prob_neg": round(float(article.get("neg", 0.0)), 3),
                "preview": article.get("preview", ""),
                "why": article.get("why", [])[:3],
            }
            for article in articles
        ],
        "context": {
            "k": req.k,
            "threshold": req.threshold,
            "model_version": "finbert@1.0",
            "retrieval": "GNews RSS + URL fetch",
        },
    }
