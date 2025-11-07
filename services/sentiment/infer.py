"""Sentiment inference that prefers FinBERT with a keyword fallback."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List

from .retriever import get_texts


def _softmax(logits: List[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _finbert_infer(texts: List[str], ckpt_dir: str) -> List[Dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    checkpoint = os.path.join(ckpt_dir, "sentiment.pt") if ckpt_dir else ""
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()

    outputs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for text in texts:
            if not text:
                outputs.append({"scores": [0.0, 0.0, 0.0]})
                continue
            encoded = tokenizer(
                text[:1500],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**encoded).logits[0].tolist()
            scores = _softmax(logits)
            outputs.append({"scores": scores})
    return outputs


def infer_sentiment(text_or_query: str, k: int, threshold: float, ckpt_dir: str) -> List[Dict[str, Any]]:
    """Return sentiment labels for URLs or queries."""

    items = get_texts(text_or_query, k=k)
    texts = [text for _, text in items]

    try:
        results = _finbert_infer(list(texts), ckpt_dir=ckpt_dir)
    except Exception:
        results = []

    output: List[Dict[str, Any]] = []
    if results:
        for (url, text), result in zip(items, results):
            scores = result.get("scores", [0.0, 0.0, 0.0])
            neg, neu, pos = scores
            label = "positive"
            rationale: List[str] = []
            if pos >= threshold and neg >= threshold:
                label = "mixed"
                rationale.append("positive and negative probabilities exceed threshold")
            elif max(scores) == neu:
                label = "mixed"
                rationale.append("FinBERT predicted neutral tone")
            elif neg >= pos:
                label = "negative"
                rationale.append("negative probability highest")
            else:
                rationale.append("positive probability highest")
            output.append(
                {
                    "url": url,
                    "label": label,
                    "pos": float(pos),
                    "neg": float(neg),
                    "preview": text[:140],
                    "why": rationale,
                }
            )
        return output[:k]

    # Fallback heuristic when transformers/torch are not installed.
    positive_markers = ["gain", "beat", "surge", "growth", "record", "strong", "bull"]
    negative_markers = ["loss", "miss", "fall", "decline", "weak", "bear", "layoff", "risk"]

    for url, text in items[:k]:
        lower_text = text.lower()
        pos_hits = sum(lower_text.count(marker) for marker in positive_markers)
        neg_hits = sum(lower_text.count(marker) for marker in negative_markers)
        pos_score = min(0.99, 0.5 + 0.1 * pos_hits)
        neg_score = min(0.99, 0.5 + 0.1 * neg_hits)
        if pos_score >= threshold and neg_score >= threshold:
            label = "mixed"
        elif pos_score >= neg_score:
            label = "positive"
        else:
            label = "negative"
        why: List[str] = []
        if pos_hits:
            why.append(f"found {pos_hits} positive keywords")
        if neg_hits:
            why.append(f"found {neg_hits} negative keywords")
        output.append(
            {
                "url": url,
                "label": label,
                "pos": pos_score,
                "neg": neg_score,
                "preview": text[:140],
                "why": why,
            }
        )
    return output
