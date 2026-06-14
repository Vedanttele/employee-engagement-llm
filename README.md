# Employee Engagement Intelligence

> Production-ready AI engineering platform for employee survey analysis — PII anonymization, multilingual sentiment analysis, burnout risk detection, RAG-grounded insights, and a FastAPI + Streamlit interface deployed on Azure.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Claude AI](https://img.shields.io/badge/Claude-AI-orange)](https://www.anthropic.com/)
[![Docker](https://img.shields.io/badge/Docker-compose-blue?logo=docker)](https://www.docker.com/)
[![Azure](https://img.shields.io/badge/Azure-Container%20Apps-blue?logo=microsoftazure)](https://azure.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Pipeline

```
Survey Data
     ↓
Data Validation          (Pydantic schema enforcement)
     ↓
PII Detection & Anonymization  (Microsoft Presidio + regex fallback)
     ↓
Sentiment Service        (Claude — batch, multilingual)
     ↓
Topic Extraction Service (BERTopic + multilingual-MiniLM embeddings)
     ↓
Burnout Risk Detection   (Rule-based scoring on sentiment + themes + emotions)
     ↓
RAG Knowledge Base       (ChromaDB — HR best-practice documents)
     ↓
LLM Insights Generator   (Claude — grounded executive report)
     ↓
FastAPI                  (REST API, Prometheus metrics, health endpoints)
     ↓
Streamlit                (Interactive dashboard — upload, analyze, export)
     ↓
Docker                   (Multi-stage builds, docker-compose for local stack)
     ↓
Azure Container Apps     (CI/CD via GitHub Actions)
     ↓
Monitoring & Evaluation  (Prometheus + Grafana)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude claude-sonnet-4-5 (Anthropic) |
| PII Detection | Microsoft Presidio + spaCy |
| Topic Modeling | BERTopic + `paraphrase-multilingual-MiniLM-L12-v2` |
| RAG | ChromaDB (persistent, local) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker (multi-stage) + Docker Compose |
| Cloud | Azure Container Apps |
| CI/CD | GitHub Actions |
| Config | Pydantic Settings |
| Logging | structlog (JSON) |

---

## Project Structure

```
employee-engagement-llm/
├── employee_engagement/          # Installable Python package
│   ├── config.py                 # Pydantic Settings — all config from env
│   ├── data/
│   │   ├── schemas.py            # Pydantic models (SurveyResponse, AnalyzedResponse, ...)
│   │   ├── validator.py          # Batch validation
│   │   ├── pii.py               # PII anonymization
│   │   └── generator.py          # Synthetic data generation via Claude
│   ├── services/
│   │   ├── sentiment.py          # Claude sentiment + theme analysis
│   │   ├── topics.py             # BERTopic multilingual topic extraction
│   │   ├── burnout.py            # Burnout risk scoring
│   │   └── rag.py               # ChromaDB RAG knowledge base
│   ├── insights/
│   │   └── generator.py          # LLM insights + RAG grounding
│   ├── api/
│   │   ├── main.py              # FastAPI app (CORS, metrics, lifespan)
│   │   ├── deps.py              # Singleton dependency injection
│   │   └── routes/
│   │       ├── health.py         # GET /health, GET /ready
│   │       ├── survey.py         # POST /api/v1/survey/analyze, /validate
│   │       └── insights.py       # POST /api/v1/insights/generate
│   ├── monitoring/
│   │   └── metrics.py            # Prometheus counters/histograms
│   └── utils/
│       └── logging.py            # structlog JSON configuration
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
├── tests/
│   ├── conftest.py
│   ├── test_data/                # Validator + PII tests
│   ├── test_services/            # Burnout detector tests (no API calls)
│   └── test_api/                 # FastAPI route tests
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.streamlit
├── monitoring/
│   └── prometheus.yml
├── .github/workflows/
│   └── azure-deploy.yml          # Build → Test → Push ACR → Deploy
├── pyproject.toml
├── docker-compose.yml
└── Makefile
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- `ANTHROPIC_API_KEY` (from [console.anthropic.com](https://console.anthropic.com))
- Docker (for full stack)

### Local development

```bash
# 1. Clone
git clone https://github.com/Vedanttele/employee-engagement-llm.git
cd employee-engagement-llm

# 2. Install
cp .env.example .env          # add your ANTHROPIC_API_KEY
make install-dev              # installs package + dev deps + spaCy model

# 3. Run API
make run-api                  # http://localhost:8000/docs

# 4. Run Streamlit (separate terminal)
make run-app                  # http://localhost:8501
```

### Docker (full stack including Prometheus + Grafana)

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
make docker-up

# Services:
#   API         → http://localhost:8000/docs
#   Streamlit   → http://localhost:8501
#   Prometheus  → http://localhost:9090
#   Grafana     → http://localhost:3000  (admin/admin)
```

### Generate synthetic data

```bash
python -m employee_engagement.data.generator --samples 100
```

---

## API Reference

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + version |
| `GET` | `/ready` | Kubernetes readiness probe |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/api/v1/survey/validate` | Validate records without analyzing |
| `POST` | `/api/v1/survey/analyze` | Full pipeline (PII → sentiment → burnout → topics) |
| `POST` | `/api/v1/insights/generate` | Generate LLM executive report + RAG context |

### Example: Analyze surveys

```bash
curl -X POST http://localhost:8000/api/v1/survey/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "responses": [
      {
        "response_id": "r1",
        "text": "The workload is unsustainable. I am completely burned out.",
        "department": "Engineering",
        "language": "en"
      }
    ],
    "run_topic_modeling": false
  }'
```

---

## Configuration

All settings are loaded from environment variables (via `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Claude model ID |
| `MAX_TOKENS` | `1024` | Max tokens per LLM call |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI port |
| `ANALYSIS_BATCH_SIZE` | `10` | Responses per Claude call |
| `N_TOPICS` | `15` | BERTopic topic count |
| `RAG_TOP_K` | `3` | RAG documents retrieved per query |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Tests

```bash
make test
# or
pytest tests/ -v --cov=employee_engagement
```

Tests requiring the Claude API are integration tests and require `ANTHROPIC_API_KEY`. Unit tests (validator, PII, burnout) run without any API key.

---

## Azure Deployment

Required GitHub Secrets:

| Secret | Description |
|--------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `AZURE_CREDENTIALS` | Azure service principal JSON |
| `AZURE_REGISTRY` | ACR login server (e.g. `myregistry.azurecr.io`) |
| `AZURE_ACR_USERNAME` | ACR username |
| `AZURE_ACR_PASSWORD` | ACR password |

The workflow (`.github/workflows/azure-deploy.yml`) runs on every push to `main`:
`lint → test → build & push images → deploy to Azure Container Apps`

---

## Supported Languages

English, German, French, Spanish, Hindi, Chinese (Simplified)

## License

MIT
