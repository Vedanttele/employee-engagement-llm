"""
data_generation.py — Generate synthetic multilingual employee survey responses.

Strategy:
- Prompt Claude to generate realistic free-text survey responses
- Each response tagged with: language, theme, sentiment, department
- Batch generation (5 responses per API call) to stay cost-efficient
- Saves incrementally to CSV so progress is not lost on failure

Usage:
    python -m src.data_generation --samples 500 --output data/synthetic/survey_responses.csv
"""

import anthropic
import pandas as pd
import json
import random
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from rich.console import Console

from src.config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS,
    SUPPORTED_LANGUAGES, ENGAGEMENT_THEMES, SENTIMENT_LABELS,
    SENTIMENT_DISTRIBUTION, GENERATION_BATCH_SIZE, SAMPLES_PER_LANGUAGE,
    SYNTHETIC_DIR, RANDOM_SEED, RETRY_ATTEMPTS, RETRY_WAIT_SECONDS
)

console = Console()
random.seed(RANDOM_SEED)

# ── Departments (realistic corporate) ─────────────────────────────────────────
DEPARTMENTS = [
    "Engineering", "Product", "Finance", "HR", "Operations",
    "Sales", "Marketing", "Legal", "Customer Success", "Data & Analytics"
]

# ── Experience levels ──────────────────────────────────────────────────────────
EXPERIENCE_LEVELS = ["0-1 year", "1-3 years", "3-5 years", "5-10 years", "10+ years"]


def _build_generation_prompt(
    language: str,
    language_name: str,
    theme_key: str,
    theme_display: str,
    sentiment: str,
    department: str,
    n: int,
) -> str:
    """Build the prompt for generating n survey responses."""
    sentiment_guidance = {
        "very_positive": "extremely enthusiastic, grateful, highly satisfied — use strong positive language",
        "positive":      "generally satisfied, appreciative, with minor nuances",
        "neutral":       "factual, balanced, neither praising nor criticizing significantly",
        "negative":      "frustrated or disappointed, specific complaints, but professional tone",
        "very_negative": "strongly dissatisfied, urgent concerns, possibly mentioning intent to leave",
    }

    return f"""You are generating realistic employee engagement survey open-text responses for an internal HR analytics project.

Generate exactly {n} survey responses in **{language_name}** ({language}) for:
- Survey theme: {theme_display}
- Employee sentiment: {sentiment} ({sentiment_guidance[sentiment]})
- Department: {department}

Requirements:
1. Write ENTIRELY in {language_name} — no mixing of languages
2. Length: 1-4 sentences each, realistic survey length (not too polished)
3. Reflect real employee voice: colloquial where appropriate, specific examples
4. Vary phrasing and structure across responses — avoid repetition
5. For German: use formal "Sie" form occasionally, mix with informal
6. For Hindi/Chinese: use natural script (Devanagari / Simplified Han)

Return ONLY a valid JSON array with exactly {n} strings, no other text:
["response 1", "response 2", ...]"""


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(RETRY_WAIT_SECONDS))
def _call_claude(client: anthropic.Anthropic, prompt: str) -> list[str]:
    """Call Claude API and parse the JSON list response."""
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    return json.loads(raw)


def generate_synthetic_dataset(
    samples_per_language: int = SAMPLES_PER_LANGUAGE,
    output_path: Path = SYNTHETIC_DIR / "survey_responses.csv",
    resume: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic multilingual survey dataset.

    Args:
        samples_per_language: Number of responses to generate per language
        output_path: Where to save the CSV incrementally
        resume: If True, skip rows already generated (reads existing CSV)
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment/.env")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build the full job list
    weighted_sentiments = []
    for s, prob in SENTIMENT_DISTRIBUTION.items():
        weighted_sentiments.extend([s] * int(prob * samples_per_language))

    records = []
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        random.shuffle(weighted_sentiments)
        for i, (theme_key, theme_display) in enumerate(ENGAGEMENT_THEMES.items()):
            # Roughly distribute samples across themes
            theme_count = samples_per_language // len(ENGAGEMENT_THEMES)
            for j in range(theme_count):
                sentiment = weighted_sentiments[(i * theme_count + j) % len(weighted_sentiments)]
                dept = random.choice(DEPARTMENTS)
                exp = random.choice(EXPERIENCE_LEVELS)
                records.append({
                    "language_code": lang_code,
                    "language_name": lang_name,
                    "theme_key": theme_key,
                    "theme_display": theme_display,
                    "sentiment_label": sentiment,
                    "department": dept,
                    "experience_level": exp,
                    "response_text": None,  # To be filled
                })

    df_plan = pd.DataFrame(records)

    # Resume: load existing and skip already-generated rows
    if resume and output_path.exists():
        df_existing = pd.read_csv(output_path)
        existing_count = df_existing["response_text"].notna().sum()
        console.print(f"[yellow]Resuming: {existing_count} responses already generated[/yellow]")
        # Merge existing responses back
        df_plan = df_plan.iloc[existing_count:].reset_index(drop=True)
        existing_rows = df_existing[df_existing["response_text"].notna()]
    else:
        existing_rows = pd.DataFrame()

    all_generated = [existing_rows] if not existing_rows.empty else []

    # Batch generation
    batch = []
    pbar = tqdm(df_plan.iterrows(), total=len(df_plan), desc="Generating responses")

    for idx, row in pbar:
        batch.append(row)

        if len(batch) >= GENERATION_BATCH_SIZE or idx == len(df_plan) - 1:
            # Group by (lang, theme, sentiment, dept) for efficient prompting
            # For simplicity: generate each group's batch in one call
            first = batch[0]
            prompt = _build_generation_prompt(
                language=first["language_code"],
                language_name=first["language_name"],
                theme_key=first["theme_key"],
                theme_display=first["theme_display"],
                sentiment=first["sentiment_label"],
                department=first["department"],
                n=len(batch),
            )

            try:
                texts = _call_claude(client, prompt)
                if len(texts) < len(batch):
                    texts += ["[generation failed]"] * (len(batch) - len(texts))
            except Exception as e:
                console.print(f"[red]Generation failed for batch: {e}[/red]")
                texts = ["[generation failed]"] * len(batch)

            for row_data, text in zip(batch, texts):
                row_dict = row_data.to_dict()
                row_dict["response_text"] = text
                all_generated.append(pd.DataFrame([row_dict]))

            batch = []

            # Incremental save every 50 rows
            if len(all_generated) % 10 == 0:
                df_partial = pd.concat(all_generated, ignore_index=True)
                df_partial.to_csv(output_path, index=False)
                pbar.set_postfix({"saved": len(df_partial)})

    # Final save
    df_final = pd.concat(all_generated, ignore_index=True)
    df_final.to_csv(output_path, index=False)
    console.print(f"[green]✓ Dataset saved: {output_path} ({len(df_final)} rows)[/green]")
    return df_final


def load_huggingface_base_dataset() -> pd.DataFrame:
    """
    Load the real multilingual sentiment dataset from HuggingFace
    (tyqiangz/multilingual-sentiments) as supplementary data.

    Maps it to our schema — only keeps EN, DE, FR, ES, ZH rows.
    Note: This is product-review domain, flagged as 'auxiliary' source.
    """
    try:
        from datasets import load_dataset
        console.print("[cyan]Loading HuggingFace multilingual-sentiments dataset...[/cyan]")
        ds = load_dataset("tyqiangz/multilingual-sentiments", split="train")
        df = ds.to_pandas()

        # Keep only our target languages
        lang_map = {"english": "en", "german": "de", "french": "fr",
                    "spanish": "es", "chinese_simplified": "zh"}
        df = df[df["language"].isin(lang_map.keys())].copy()
        df["language_code"] = df["language"].map(lang_map)
        df["language_name"] = df["language_code"].map(SUPPORTED_LANGUAGES)

        # Map 3-class sentiment to our 5-class (coarse mapping)
        sentiment_map = {"positive": "positive", "neutral": "neutral", "negative": "negative"}
        df["sentiment_label"] = df["label"].map(sentiment_map)
        df["theme_key"] = "external_reviews"
        df["theme_display"] = "External Review (Auxiliary)"
        df["department"] = "Unknown"
        df["experience_level"] = "Unknown"
        df["response_text"] = df["text"]
        df["data_source"] = "huggingface_multilingual_sentiments"

        out_cols = ["response_text", "sentiment_label", "language_code",
                    "language_name", "theme_key", "theme_display",
                    "department", "experience_level", "data_source"]
        df = df[out_cols].dropna(subset=["response_text"])

        save_path = RAW_DIR / "hf_multilingual_sentiments.csv"
        df.to_csv(save_path, index=False)
        console.print(f"[green]✓ HF dataset saved: {save_path} ({len(df)} rows)[/green]")
        return df

    except Exception as e:
        console.print(f"[yellow]HF dataset load failed (not critical): {e}[/yellow]")
        return pd.DataFrame()


if __name__ == "__main__":
    from src.config import RAW_DIR  # noqa: F811

    parser = argparse.ArgumentParser(description="Generate synthetic multilingual survey dataset")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_LANGUAGE,
                        help="Samples per language (default: 500)")
    parser.add_argument("--output", type=str,
                        default=str(SYNTHETIC_DIR / "survey_responses.csv"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--hf-only", action="store_true",
                        help="Only download HuggingFace base dataset")
    args = parser.parse_args()

    if args.hf_only:
        load_huggingface_base_dataset()
    else:
        generate_synthetic_dataset(
            samples_per_language=args.samples,
            output_path=Path(args.output),
            resume=not args.no_resume,
        )
