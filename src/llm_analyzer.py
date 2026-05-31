"""
llm_analyzer.py — LLM-powered sentiment & theme analysis using Claude.

Key design decisions:
- Batch 10 responses per API call (cost efficiency)
- Structured JSON output via system prompt engineering
- Returns confidence scores, detected themes, key phrases
- Handles multilingual input natively (Claude handles all 6 languages)
- Saves results incrementally

Usage:
    python -m src.llm_analyzer \
        --input data/processed/cleaned_responses.csv \
        --output data/processed/analyzed_responses.csv
"""

import anthropic
import pandas as pd
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from rich.console import Console

from src.config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS,
    ENGAGEMENT_THEMES, SENTIMENT_LABELS,
    ANALYSIS_BATCH_SIZE, PROCESSED_DIR,
    RETRY_ATTEMPTS, RETRY_WAIT_SECONDS
)

console = Console()

# ── System prompt for structured analysis ──────────────────────────────────────
ANALYSIS_SYSTEM_PROMPT = """You are an expert HR analytics AI that analyzes employee engagement survey responses.

You receive batches of survey responses in multiple languages (English, German, French, Spanish, Hindi, Chinese).
Analyze each response and return structured JSON analysis.

For EACH response return:
{
  "id": <original index>,
  "sentiment": one of ["very_positive", "positive", "neutral", "negative", "very_negative"],
  "sentiment_confidence": float 0.0-1.0,
  "primary_theme": one of the theme keys provided,
  "secondary_themes": list of up to 2 additional theme keys (can be empty),
  "key_phrases": list of 2-4 key phrases extracted (in original language),
  "summary_en": 1-sentence English summary of what the employee is saying,
  "emotion_tags": list of 1-3 emotions (e.g. ["frustrated", "hopeful"]),
  "action_signal": one of ["urgent_action", "monitor", "positive_share", "no_action"],
  "language_detected": ISO code of the detected language
}

Action signal rules:
- "urgent_action": very_negative + themes around wellbeing/safety, or mentions of leaving
- "monitor": negative or mixed signals worth tracking
- "positive_share": very_positive, worth sharing as best practice
- "no_action": neutral or routine

Return ONLY a valid JSON array. No markdown, no extra text."""


def _build_analysis_prompt(batch: list[dict], theme_list: list[str]) -> str:
    """Build user prompt for a batch of responses."""
    theme_options = "\n".join(f"  - {k}: {v}" for k, v in ENGAGEMENT_THEMES.items())

    responses_json = json.dumps(
        [{"id": item["id"], "text": item["text"], "declared_language": item["lang"]}
         for item in batch],
        ensure_ascii=False, indent=2
    )

    return f"""Analyze these {len(batch)} employee survey responses.

Available theme keys:
{theme_options}

Responses to analyze:
{responses_json}

Return a JSON array with one analysis object per response."""


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(RETRY_WAIT_SECONDS))
def _call_claude_analysis(client: anthropic.Anthropic, prompt: str) -> list[dict]:
    """Call Claude and parse structured analysis response."""
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS * ANALYSIS_BATCH_SIZE,  # More tokens for batch
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    return json.loads(raw)


def _make_fallback_result(row_id: int) -> dict:
    """Return a null/fallback result when Claude call fails."""
    return {
        "id": row_id,
        "sentiment": "neutral",
        "sentiment_confidence": 0.0,
        "primary_theme": "unknown",
        "secondary_themes": [],
        "key_phrases": [],
        "summary_en": "[analysis failed]",
        "emotion_tags": [],
        "action_signal": "no_action",
        "language_detected": "unknown",
        "analysis_failed": True,
    }


def analyze_dataset(
    input_path: Path,
    output_path: Path = PROCESSED_DIR / "analyzed_responses.csv",
    batch_size: int = ANALYSIS_BATCH_SIZE,
    resume: bool = True,
    limit: int = None,
) -> pd.DataFrame:
    """
    Run LLM analysis on the cleaned dataset.

    Args:
        input_path:  Cleaned CSV from preprocessing.py
        output_path: Where to save results
        batch_size:  Responses per API call
        resume:      Skip already-analyzed rows
        limit:       Analyze only first N rows (for testing)
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    df = pd.read_csv(input_path)
    if limit:
        df = df.head(limit)

    console.print(f"Loaded {len(df)} responses for analysis")

    # Resume logic
    analysis_col = "llm_sentiment"
    if resume and output_path.exists():
        df_existing = pd.read_csv(output_path)
        analyzed_ids = set(df_existing[df_existing[analysis_col].notna()]["row_id"].tolist())
        df_todo = df[~df["row_id"].isin(analyzed_ids)].copy()
        console.print(f"[yellow]Resume: {len(analyzed_ids)} already analyzed, {len(df_todo)} remaining[/yellow]")
    else:
        df_todo = df.copy()
        df_existing = pd.DataFrame()

    # Prepare results accumulator
    results_list = []

    # Build batches
    rows = df_todo.to_dict("records")
    batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

    for batch_rows in tqdm(batches, desc="Analyzing with Claude"):
        batch_input = [
            {
                "id": r["row_id"],
                "text": r["response_text"],
                "lang": r.get("language_code", "unknown"),
            }
            for r in batch_rows
        ]

        theme_list = list(ENGAGEMENT_THEMES.keys())
        prompt = _build_analysis_prompt(batch_input, theme_list)

        try:
            analysis_results = _call_claude_analysis(client, prompt)
            # Index by id for safe mapping
            analysis_map = {r["id"]: r for r in analysis_results}
        except Exception as e:
            console.print(f"[red]Batch failed: {e}[/red]")
            analysis_map = {r["id"]: _make_fallback_result(r["id"]) for r in batch_input}

        for row in batch_rows:
            rid = row["row_id"]
            analysis = analysis_map.get(rid, _make_fallback_result(rid))
            results_list.append({
                **row,
                "llm_sentiment":          analysis.get("sentiment", "neutral"),
                "llm_sentiment_confidence": analysis.get("sentiment_confidence", 0.0),
                "llm_primary_theme":      analysis.get("primary_theme", "unknown"),
                "llm_secondary_themes":   json.dumps(analysis.get("secondary_themes", [])),
                "llm_key_phrases":        json.dumps(analysis.get("key_phrases", [])),
                "llm_summary_en":         analysis.get("summary_en", ""),
                "llm_emotion_tags":       json.dumps(analysis.get("emotion_tags", [])),
                "llm_action_signal":      analysis.get("action_signal", "no_action"),
                "llm_language_detected":  analysis.get("language_detected", "unknown"),
                "analysis_failed":        analysis.get("analysis_failed", False),
            })

    # Merge with existing
    df_new = pd.DataFrame(results_list)
    if not df_existing.empty:
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(output_path, index=False)
    console.print(f"[green]✓ Analysis saved: {output_path} ({len(df_final)} rows)[/green]")

    # Quick stats
    if "llm_sentiment" in df_final.columns:
        success = df_final[~df_final.get("analysis_failed", False)].shape[0]
        console.print(f"Success rate: {success}/{len(df_final)} ({success/len(df_final)*100:.1f}%)")
        console.print("\nSentiment distribution:")
        console.print(df_final["llm_sentiment"].value_counts().to_string())

    return df_final


def compute_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a numeric engagement score (0-100) from sentiment labels.
    Used for aggregated dashboards and trend analysis.
    """
    score_map = {
        "very_positive": 100,
        "positive":       75,
        "neutral":        50,
        "negative":       25,
        "very_negative":   0,
    }
    df = df.copy()
    df["engagement_score"] = df["llm_sentiment"].map(score_map).fillna(50)

    # Weight by confidence
    if "llm_sentiment_confidence" in df.columns:
        # Pull toward neutral (50) when confidence is low
        df["engagement_score"] = (
            df["engagement_score"] * df["llm_sentiment_confidence"]
            + 50 * (1 - df["llm_sentiment_confidence"])
        ).round(1)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM analysis on cleaned survey responses")
    parser.add_argument("--input", type=str,
                        default=str(PROCESSED_DIR / "cleaned_responses.csv"))
    parser.add_argument("--output", type=str,
                        default=str(PROCESSED_DIR / "analyzed_responses.csv"))
    parser.add_argument("--batch-size", type=int, default=ANALYSIS_BATCH_SIZE)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only analyze first N rows (for testing)")
    args = parser.parse_args()

    analyze_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        batch_size=args.batch_size,
        resume=not args.no_resume,
        limit=args.limit,
    )
