# Neighborhood Pricing & Sentiment Service

This repository contains a FastAPI application that provides two capabilities:

1. **Housing price estimation** using transparent comparable filtering and a bootstrap confidence interval.
2. **Financial sentiment analysis** using the pre-trained [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) model with an automatic keyword-based fallback when model weights are unavailable.

The project emphasises production-friendly structure, easy-to-follow code, and a lightweight inference container. Training occurs elsewhere and model checkpoints can be mounted at runtime.

## Project layout

```
.
├── api/
│   └── app.py              # FastAPI wiring and endpoints
├── analytics/
│   └── price_stats.py      # Comparable filtering and bootstrap CI logic
├── data/
│   ├── listings.csv        # Sample housing data
│   └── listings_repository.py
├── domain/
│   └── listings.py         # Listing entity definition
├── services/
│   ├── house_avg_service.py
│   └── sentiment/
│       ├── infer.py        # FinBERT inference with fallback heuristic
│       └── retriever.py    # Text retrieval from URLs or Google News RSS
├── checkpoints/
│   └── sentiment/          # Place optional sentiment.pt checkpoint here
├── tests/                  # Automated checks
├── Dockerfile              # CPU-only inference image
├── pyproject.toml          # Application dependencies
└── README.md
```

## Quickstart

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[sentiment]
```

### Run the API

```bash
uvicorn api.app:app --reload --port 8000
```

### Docker workflow

```bash
docker build -t neighborhood-sentiment:cpu .
docker run -p 8000:8000 \
  -e LISTINGS_CSV=data/listings.csv \
  -e SENTI_CKPT=checkpoints/sentiment \
  neighborhood-sentiment:cpu
```

Mount the `checkpoints/sentiment` directory with `sentiment.pt` to enable LoRA-merged weights (optional). Without the checkpoint or transformer dependencies, the service falls back to a keyword heuristic so the API stays responsive for demos.

## Tests

```bash
pytest
```

## Configuration

Environment variables:

- `LISTINGS_CSV`: Path to the CSV file with housing listings (default: `data/listings.csv`).
- `SENTI_CKPT`: Directory that contains the optional `sentiment.pt` file for FinBERT fine-tuning.

## Notes

- The housing estimate intentionally avoids machine learning for maximum transparency.
- Sentiment inference prefers FinBERT. If transformers/torch are not installed or the checkpoint is missing, it uses a deterministic keyword signal as a graceful fallback.
- Training scripts for LoRA adapters can live outside this repo; copy `sentiment.pt` into `checkpoints/sentiment/` when deploying.
