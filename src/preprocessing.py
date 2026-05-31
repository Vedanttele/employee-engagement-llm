"""
preprocessing.py — Clean, detect language, and normalize survey responses.

Pipeline:
1. Remove nulls / failed generations
2. Detect actual language (validate against declared language)
3. Clean text (whitespace, special chars, encoding issues)
4. Flag mismatches (declared vs detected language)
5. Compute basic text features (length, word count)

Usage:
    python -m src.preprocessing \
        --input data/synthetic/survey_responses.csv \
        --output data/processed/cleaned_responses.csv
"""

import pandas as pd
import numpy as np
import re
import argparse
import unicodedata
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.config import SUPPORTED_LANGUAGES, PROCESSED_DIR, SYNTHETIC_DIR

console = Console()

# ── Language detection setup ───────────────────────────────────────────────────
# Use langdetect as primary, lingua as fallback (more accurate for short text)
def _build_detector():
    """Build lingua detector for the 6 supported languages."""
    try:
        from lingua import Language, LanguageDetectorBuilder
        lang_map = {
            "en": Language.ENGLISH,
            "de": Language.GERMAN,
            "fr": Language.FRENCH,
            "es": Language.SPANISH,
            "hi": Language.HINDI,
            "zh": Language.CHINESE,
        }
        langs = list(lang_map.values())
        detector = LanguageDetectorBuilder.from_languages(*langs).build()
        return detector, lang_map
    except ImportError:
        console.print("[yellow]lingua not installed — using langdetect only[/yellow]")
        return None, {}

_lingua_detector, _lingua_lang_map = _build_detector()
_lingua_reverse = {v: k for k, v in _lingua_lang_map.items()} if _lingua_lang_map else {}


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect language of text.
    Returns (iso_code, confidence). Returns ('unknown', 0.0) on failure.
    """
    if not text or len(text.strip()) < 5:
        return "unknown", 0.0

    # Try lingua first (better for short text and non-Latin scripts)
    if _lingua_detector:
        try:
            result = _lingua_detector.detect_language_of(text)
            if result:
                code = _lingua_reverse.get(result, "unknown")
                return code, 0.95  # lingua doesn't return confidence, use high fixed
        except Exception:
            pass

    # Fallback: langdetect
    try:
        from langdetect import detect, detect_langs
        langs = detect_langs(text)
        if langs:
            top = langs[0]
            # Map full codes (zh-cn → zh)
            code = top.lang.split("-")[0]
            return code, float(top.prob)
    except Exception:
        pass

    return "unknown", 0.0


def clean_text(text: str) -> str:
    """
    Clean a single response text.
    - Normalize unicode
    - Remove control characters
    - Normalize whitespace
    - Preserve CJK, Devanagari, Latin scripts
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode (NFC: compose characters)
    text = unicodedata.normalize("NFC", text)

    # Remove control characters (except newline/tab)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C"
                   or ch in "\n\t")

    # Collapse multiple whitespace/newlines → single space
    text = re.sub(r"\s+", " ", text).strip()

    # Remove generation artifacts
    text = re.sub(r"\[generation failed\]", "", text, flags=re.IGNORECASE).strip()

    return text


def compute_text_features(text: str) -> dict:
    """Compute basic features useful for filtering and EDA."""
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0,
        "has_question": "?" in text,
        "sentence_count": max(1, len(re.findall(r"[.!?。！？]", text))),
    }


def validate_language(declared: str, detected: str) -> str:
    """
    Classify language validation result.
    Returns: 'match' | 'mismatch' | 'uncertain'
    """
    if detected == "unknown":
        return "uncertain"
    if detected == declared:
        return "match"
    # Chinese variants (zh-cn, zh-tw both map to zh)
    if declared == "zh" and detected in ("zh", "zh-cn", "zh-tw"):
        return "match"
    return "mismatch"


def preprocess_dataset(
    input_path: Path,
    output_path: Path = PROCESSED_DIR / "cleaned_responses.csv",
    drop_mismatches: bool = False,
    min_word_count: int = 3,
    max_word_count: int = 150,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Args:
        input_path:       Raw CSV from data_generation.py
        output_path:      Cleaned output path
        drop_mismatches:  If True, drop rows where detected language ≠ declared
        min_word_count:   Drop responses shorter than this
        max_word_count:   Drop responses longer than this (likely generation errors)
    """
    console.print(f"[cyan]Loading: {input_path}[/cyan]")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    console.print(f"Loaded {initial_count} rows")

    # ── Step 1: Drop nulls and failed generations ──────────────────────────────
    df = df[df["response_text"].notna()].copy()
    df = df[df["response_text"].str.strip().str.len() > 0].copy()
    df = df[~df["response_text"].str.contains(r"\[generation failed\]", na=False)].copy()
    console.print(f"After null/failure drop: {len(df)} rows")

    # ── Step 2: Clean text ─────────────────────────────────────────────────────
    console.print("[cyan]Cleaning text...[/cyan]")
    df["response_text"] = df["response_text"].apply(clean_text)

    # ── Step 3: Text features ──────────────────────────────────────────────────
    console.print("[cyan]Computing text features...[/cyan]")
    features = df["response_text"].apply(compute_text_features).apply(pd.Series)
    df = pd.concat([df, features], axis=1)

    # ── Step 4: Length filter ──────────────────────────────────────────────────
    df = df[
        (df["word_count"] >= min_word_count) &
        (df["word_count"] <= max_word_count)
    ].copy()
    console.print(f"After length filter ({min_word_count}–{max_word_count} words): {len(df)} rows")

    # ── Step 5: Language detection ─────────────────────────────────────────────
    console.print("[cyan]Detecting languages...[/cyan]")
    tqdm.pandas(desc="Detecting language")
    lang_results = df["response_text"].progress_apply(
        lambda t: pd.Series(detect_language(t), index=["detected_lang", "lang_confidence"])
    )
    df = pd.concat([df, lang_results], axis=1)

    # Validate
    df["lang_validation"] = df.apply(
        lambda r: validate_language(r["language_code"], r["detected_lang"]),
        axis=1
    )

    # ── Step 6: Optionally drop mismatches ────────────────────────────────────
    mismatch_count = (df["lang_validation"] == "mismatch").sum()
    console.print(f"Language mismatches: {mismatch_count} / {len(df)}")

    if drop_mismatches:
        df = df[df["lang_validation"] != "mismatch"].copy()
        console.print(f"After mismatch drop: {len(df)} rows")

    # ── Step 7: Add metadata ───────────────────────────────────────────────────
    if "data_source" not in df.columns:
        df["data_source"] = "synthetic_claude"

    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    # ── Save ───────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"\n[green]✓ Cleaned dataset saved: {output_path}[/green]")

    # ── Summary report ─────────────────────────────────────────────────────────
    _print_summary(df, initial_count)
    return df


def _print_summary(df: pd.DataFrame, initial_count: int):
    """Print a rich summary table."""
    table = Table(title="Dataset Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Initial rows", str(initial_count))
    table.add_row("Final rows", str(len(df)))
    table.add_row("Retention rate", f"{len(df)/initial_count*100:.1f}%")
    table.add_row("Languages", ", ".join(df["language_code"].unique().tolist()))
    table.add_row("Themes", str(df["theme_key"].nunique()))
    table.add_row("Avg word count", f"{df['word_count'].mean():.1f}")
    table.add_row("Lang match rate", f"{(df['lang_validation']=='match').mean()*100:.1f}%")

    console.print(table)

    # Per-language counts
    lang_table = Table(title="Responses by Language")
    lang_table.add_column("Language")
    lang_table.add_column("Count")
    lang_table.add_column("Match %")
    for lang in sorted(df["language_code"].unique()):
        sub = df[df["language_code"] == lang]
        match_pct = (sub["lang_validation"] == "match").mean() * 100
        lang_table.add_row(
            SUPPORTED_LANGUAGES.get(lang, lang),
            str(len(sub)),
            f"{match_pct:.0f}%"
        )
    console.print(lang_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess survey response dataset")
    parser.add_argument("--input", type=str,
                        default=str(SYNTHETIC_DIR / "survey_responses.csv"))
    parser.add_argument("--output", type=str,
                        default=str(PROCESSED_DIR / "cleaned_responses.csv"))
    parser.add_argument("--drop-mismatches", action="store_true")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=150)
    args = parser.parse_args()

    preprocess_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        drop_mismatches=args.drop_mismatches,
        min_word_count=args.min_words,
        max_word_count=args.max_words,
    )
