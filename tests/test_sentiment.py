import services.sentiment.infer as infer_module


def test_sentiment_fallback(monkeypatch):
    calls = {}

    def fake_get_texts(text_or_query, k):
        calls["called_with"] = (text_or_query, k)
        return [("http://example.com", "The company reported record growth and strong gains.")]

    def failing_finbert(texts, ckpt_dir):
        raise RuntimeError("transformers unavailable")

    monkeypatch.setattr(infer_module, "get_texts", fake_get_texts)
    monkeypatch.setattr(infer_module, "_finbert_infer", failing_finbert)

    results = infer_module.infer_sentiment("Example", k=1, threshold=0.6, ckpt_dir="")

    assert calls["called_with"] == ("Example", 1)
    assert results[0]["label"] == "positive"
    assert results[0]["pos"] > results[0]["neg"]
