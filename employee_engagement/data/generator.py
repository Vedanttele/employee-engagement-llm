"""Synthetic multilingual survey data generation via Claude API."""

import json
import random
import argparse
from pathlib import Path

import anthropic
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from rich.console import Console

from employee_engagement.config import (
    get_settings,
    SUPPORTED_LANGUAGES,
    ENGAGEMENT_THEMES,
    SENTIMENT_LABELS,
    SENTIMENT_DISTRIBUTION,
    DEPARTMENTS,
)

console = Console()

EXPERIENCE_LEVELS = ["0-1 year", "1-3 years", "3-5 years", "5-10 years", "10+ years"]

_SENTIMENT_GUIDANCE = {
    "very_positive": "extremely enthusiastic, grateful, highly satisfied",
    "positive": "generally satisfied, appreciative, with minor nuances",
    "neutral": "factual, balanced, neither praising nor criticizing",
    "negative": "frustrated or disappointed, specific complaints, professional tone",
    "very_negative": "strongly dissatisfied, urgent concerns, possibly mentioning intent to leave",
}


def _build_prompt(
    language: str,
    language_name: str,
    theme_key: str,
    theme_display: str,
    sentiment: str,
    department: str,
    n: int,
) -> str:
    return f"""Generate exactly {n} realistic employee engagement survey responses in **{language_name}** for:
- Theme: {theme_display}
- Sentiment: {sentiment} ({_SENTIMENT_GUIDANCE[sentiment]})
- Department: {department}

Requirements:
1. Write ENTIRELY in {language_name}
2. Length: 1-4 sentences, realistic survey length
3. Reflect real employee voice with specific examples
4. Vary phrasing — no repetition

Return ONLY a valid JSON array with exactly {n} strings:
["response 1", "response 2", ...]"""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _generate_batch(
    client: anthropic.Anthropic,
    model: str,
    language: str,
    language_name: str,
    theme_key: str,
    theme_display: str,
    sentiment: str,
    department: str,
    n: int,
) -> list[str]:
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": _build_prompt(
                language, language_name, theme_key, theme_display, sentiment, department, n
            )}
        ],
    )
    return json.loads(message.content[0].text)


def generate_synthetic_dataset(
    samples_per_language: int | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic multilingual survey responses using Claude.
    Returns a DataFrame and optionally saves to CSV.
    """
    settings = get_settings()
    settings.ensure_dirs()

    n = samples_per_language or settings.samples_per_language
    out = output_path or (settings.synthetic_dir / "survey_responses.csv")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    random.seed(settings.random_seed)

    records: list[dict] = []
    total = len(SUPPORTED_LANGUAGES) * n

    console.print(f"[cyan]Generating {total} survey responses ({n} per language)...[/cyan]")

    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        for _ in range(n // settings.generation_batch_size):
            theme_key = random.choice(list(ENGAGEMENT_THEMES.keys()))
            sentiment = random.choices(
                list(SENTIMENT_DISTRIBUTION.keys()),
                weights=list(SENTIMENT_DISTRIBUTION.values()),
            )[0]
            dept = random.choice(DEPARTMENTS)

            try:
                responses = _generate_batch(
                    client,
                    settings.claude_model,
                    lang_code,
                    lang_name,
                    theme_key,
                    ENGAGEMENT_THEMES[theme_key],
                    sentiment,
                    dept,
                    settings.generation_batch_size,
                )
                for text in responses:
                    records.append({
                        "response_id": f"{lang_code}_{len(records):05d}",
                        "text": text,
                        "language": lang_code,
                        "declared_sentiment": sentiment,
                        "survey_theme": theme_key,
                        "department": dept,
                        "experience": random.choice(EXPERIENCE_LEVELS),
                    })
            except Exception as exc:
                console.print(f"[red]Batch failed ({lang_code}/{theme_key}): {exc}[/red]")

    df = pd.DataFrame(records)
    df.to_csv(out, index=False, encoding="utf-8")
    console.print(f"[green]Saved {len(df)} responses → {out}[/green]")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic survey data")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    generate_synthetic_dataset(samples_per_language=args.samples, output_path=args.output)


if __name__ == "__main__":
    main()
