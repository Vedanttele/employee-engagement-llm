"""Central configuration via Pydantic Settings — all values from environment."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Paths ──────────────────────────────────────────────────────────────────
    root_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Path(__file__).parent.parent / "data"
    synthetic_dir: Path = Path(__file__).parent.parent / "data" / "synthetic"
    processed_dir: Path = Path(__file__).parent.parent / "data" / "processed"
    rag_dir: Path = Path(__file__).parent.parent / "data" / "rag"

    # ── Anthropic ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-5"
    max_tokens: int = 1024

    # ── FastAPI ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    cors_origins: list[str] = ["http://localhost:8501"]

    # ── NLP ────────────────────────────────────────────────────────────────────
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    n_topics: int = 15
    min_topic_size: int = 10
    analysis_batch_size: int = 10
    generation_batch_size: int = 5
    retry_attempts: int = 3
    retry_wait_seconds: int = 2

    # ── Data generation ────────────────────────────────────────────────────────
    samples_per_language: int = 500
    random_seed: int = 42

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_collection_name: str = "hr_knowledge_base"
    rag_top_k: int = 3

    # ── Monitoring ────────────────────────────────────────────────────────────
    enable_metrics: bool = True
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        for d in [self.data_dir, self.synthetic_dir, self.processed_dir, self.rag_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── Domain constants (not env-driven) ─────────────────────────────────────────

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "zh": "Chinese (Simplified)",
}

ENGAGEMENT_THEMES: dict[str, str] = {
    "workload_balance": "Workload & Work-Life Balance",
    "management_leadership": "Management & Leadership Quality",
    "recognition_rewards": "Recognition & Rewards",
    "career_growth": "Career Development & Growth",
    "team_collaboration": "Team Collaboration & Culture",
    "company_strategy": "Company Strategy & Direction",
    "tools_resources": "Tools, Resources & Work Environment",
    "inclusion_belonging": "Diversity, Inclusion & Belonging",
    "wellbeing_safety": "Employee Wellbeing & Psychological Safety",
    "communication": "Internal Communication & Transparency",
}

SENTIMENT_LABELS: list[str] = [
    "very_positive", "positive", "neutral", "negative", "very_negative"
]

SENTIMENT_COLORS: dict[str, str] = {
    "very_positive": "#2ECC71",
    "positive": "#82E0AA",
    "neutral": "#AEB6BF",
    "negative": "#E59866",
    "very_negative": "#E74C3C",
}

SENTIMENT_DISTRIBUTION: dict[str, float] = {
    "very_positive": 0.15,
    "positive": 0.35,
    "neutral": 0.25,
    "negative": 0.18,
    "very_negative": 0.07,
}

DEPARTMENTS: list[str] = [
    "Engineering", "Product", "Finance", "HR", "Operations",
    "Sales", "Marketing", "Legal", "Customer Success", "Data & Analytics",
]

BURNOUT_RISK_THRESHOLDS: dict[str, float] = {
    "low": 0.3,
    "medium": 0.55,
    "high": 0.75,
    # above 0.75 → critical
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
