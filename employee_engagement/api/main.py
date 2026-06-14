"""FastAPI application entry point."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from employee_engagement import __version__
from employee_engagement.config import get_settings
from employee_engagement.api.routes import health, survey, insights
from employee_engagement.monitoring.metrics import instrument_app

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    log.info("app_startup", version=__version__, port=settings.api_port)
    yield
    log.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Employee Engagement Intelligence API",
        description=(
            "AI-powered survey analysis: PII anonymization, sentiment analysis, "
            "burnout detection, topic modeling, and LLM-generated insights."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.enable_metrics:
        instrument_app(app)

    app.include_router(health.router)
    app.include_router(survey.router, prefix="/api/v1")
    app.include_router(insights.router, prefix="/api/v1")

    return app


app = create_app()


def run() -> None:
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "employee_engagement.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
    )


if __name__ == "__main__":
    run()
