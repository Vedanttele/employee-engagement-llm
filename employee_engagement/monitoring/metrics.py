"""Prometheus metrics — request counters, latency histograms, business KPIs."""

import time
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# ── HTTP metrics ───────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── Business metrics ───────────────────────────────────────────────────────────
SURVEYS_ANALYZED = Counter(
    "surveys_analyzed_total",
    "Total survey responses analyzed",
)
BURNOUT_HIGH_RISK = Counter(
    "burnout_high_risk_detected_total",
    "Survey responses flagged as high/critical burnout risk",
)
URGENT_ACTIONS = Counter(
    "urgent_actions_flagged_total",
    "Responses flagged for urgent HR action",
)
PII_DETECTED = Counter(
    "pii_detected_total",
    "Responses where PII was detected and anonymized",
)
INSIGHTS_GENERATED = Counter(
    "insights_generated_total",
    "Insight reports generated",
)
LLM_CALL_LATENCY = Histogram(
    "llm_call_duration_seconds",
    "LLM (Claude API) call latency",
    ["operation"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)


def instrument_app(app: FastAPI) -> None:
    """Add Prometheus middleware and /metrics endpoint to a FastAPI app."""

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(duration)

        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
