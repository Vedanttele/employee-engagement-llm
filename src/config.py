"""
config.py — Central configuration for the project.
All constants live here. Never hardcode values in other modules.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

for d in [RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API ────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# ── Languages ─────────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "zh": "Chinese (Simplified)",
}

LANG_CODE_TO_NAME = SUPPORTED_LANGUAGES

# ── Survey Themes ──────────────────────────────────────────────────────────────
# These mirror real Allianz-style engagement survey dimensions
ENGAGEMENT_THEMES = {
    "workload_balance":     "Workload & Work-Life Balance",
    "management_leadership":"Management & Leadership Quality",
    "recognition_rewards":  "Recognition & Rewards",
    "career_growth":        "Career Development & Growth",
    "team_collaboration":   "Team Collaboration & Culture",
    "company_strategy":     "Company Strategy & Direction",
    "tools_resources":      "Tools, Resources & Work Environment",
    "inclusion_belonging":  "Diversity, Inclusion & Belonging",
    "wellbeing_safety":     "Employee Wellbeing & Psychological Safety",
    "communication":        "Internal Communication & Transparency",
}

THEME_LIST = list(ENGAGEMENT_THEMES.keys())
THEME_DISPLAY = list(ENGAGEMENT_THEMES.values())

# ── Sentiment Labels ───────────────────────────────────────────────────────────
SENTIMENT_LABELS = ["very_positive", "positive", "neutral", "negative", "very_negative"]

SENTIMENT_COLORS = {
    "very_positive": "#2ECC71",
    "positive":      "#82E0AA",
    "neutral":       "#AEB6BF",
    "negative":      "#E59866",
    "very_negative": "#E74C3C",
}

# ── Synthetic Data ─────────────────────────────────────────────────────────────
SAMPLES_PER_LANGUAGE = int(os.getenv("SYNTHETIC_SAMPLES_PER_LANGUAGE", "500"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Distribution of sentiments (must sum to 1.0)
# Intentionally skewed like real surveys: more positive/neutral than negative
SENTIMENT_DISTRIBUTION = {
    "very_positive": 0.15,
    "positive":      0.35,
    "neutral":       0.25,
    "negative":      0.18,
    "very_negative": 0.07,
}

# ── LLM Batch settings ────────────────────────────────────────────────────────
ANALYSIS_BATCH_SIZE = 10      # responses per Claude call (cost control)
GENERATION_BATCH_SIZE = 5     # survey responses to generate per Claude call
RETRY_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 2
