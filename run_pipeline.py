"""
run_pipeline.py — Single entry point to run the full pipeline end-to-end.

Steps:
  1. generate   → Generate synthetic multilingual survey dataset via Claude API
  2. fetch-hf   → (Optional) Fetch HuggingFace base dataset as auxiliary data
  3. preprocess  → Clean, detect languages, compute text features
  4. analyze    → LLM-based sentiment + theme analysis via Claude API
  5. topics     → BERTopic multilingual topic modeling
  6. all        → Run steps 1 → 2 → 3 → 4 → 5

Usage examples:
    python run_pipeline.py all --samples 200          # Quick test run
    python run_pipeline.py generate --samples 500
    python run_pipeline.py preprocess
    python run_pipeline.py analyze --limit 100        # Test first 100 rows
    python run_pipeline.py topics
    python run_pipeline.py all --samples 500          # Full pipeline
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    SYNTHETIC_DIR, PROCESSED_DIR, SAMPLES_PER_LANGUAGE
)

console = Console()

STEPS = {
    "generate":   "1. Generate synthetic multilingual dataset",
    "fetch-hf":   "2. Fetch HuggingFace auxiliary dataset",
    "preprocess": "3. Clean and preprocess responses",
    "analyze":    "4. LLM sentiment & theme analysis (Claude API)",
    "topics":     "5. Multilingual topic modeling (BERTopic)",
}


def step_generate(samples: int):
    console.print(Panel(f"[cyan]Generating {samples} synthetic responses per language...[/cyan]"))
    from src.data_generation import generate_synthetic_dataset
    df = generate_synthetic_dataset(
        samples_per_language=samples,
        output_path=SYNTHETIC_DIR / "survey_responses.csv",
    )
    console.print(f"[green]✓ Generated {len(df)} total responses[/green]")
    return df


def step_fetch_hf():
    console.print(Panel("[cyan]Fetching HuggingFace auxiliary dataset...[/cyan]"))
    from src.data_generation import load_huggingface_base_dataset
    df = load_huggingface_base_dataset()
    if len(df):
        console.print(f"[green]✓ Fetched {len(df)} HF rows[/green]")
    return df


def step_preprocess(drop_mismatches: bool = False):
    console.print(Panel("[cyan]Preprocessing dataset...[/cyan]"))
    from src.preprocessing import preprocess_dataset
    input_path = SYNTHETIC_DIR / "survey_responses.csv"
    output_path = PROCESSED_DIR / "cleaned_responses.csv"

    if not input_path.exists():
        console.print(f"[red]Input not found: {input_path}[/red]")
        console.print("[yellow]Run 'generate' step first[/yellow]")
        return None

    df = preprocess_dataset(
        input_path=input_path,
        output_path=output_path,
        drop_mismatches=drop_mismatches,
    )
    console.print(f"[green]✓ Preprocessed {len(df)} rows[/green]")
    return df


def step_analyze(limit: int = None, batch_size: int = 10):
    console.print(Panel("[cyan]Running LLM analysis (Claude API)...[/cyan]"))
    from src.llm_analyzer import analyze_dataset, compute_engagement_score
    input_path = PROCESSED_DIR / "cleaned_responses.csv"
    output_path = PROCESSED_DIR / "analyzed_responses.csv"

    if not input_path.exists():
        console.print(f"[red]Input not found: {input_path}[/red]")
        console.print("[yellow]Run 'preprocess' step first[/yellow]")
        return None

    df = analyze_dataset(
        input_path=input_path,
        output_path=output_path,
        batch_size=batch_size,
        limit=limit,
    )
    df = compute_engagement_score(df)
    df.to_csv(output_path, index=False)
    console.print(f"[green]✓ Analyzed {len(df)} rows[/green]")
    return df


def step_topics(n_topics: int = 15):
    console.print(Panel("[cyan]Running topic modeling (BERTopic)...[/cyan]"))
    from src.topic_modeling import run_full_pipeline
    input_path = PROCESSED_DIR / "analyzed_responses.csv"
    output_path = PROCESSED_DIR / "topics.csv"

    if not input_path.exists():
        console.print(f"[red]Input not found: {input_path}[/red]")
        console.print("[yellow]Run 'analyze' step first[/yellow]")
        return None

    df = run_full_pipeline(
        input_path=input_path,
        output_path=output_path,
        n_topics=n_topics,
    )
    console.print(f"[green]✓ Topics discovered and saved[/green]")
    return df


def run_all(samples: int, limit: int = None, skip_hf: bool = False):
    console.print(Panel(
        f"[bold cyan]Running Full Pipeline[/bold cyan]\n"
        f"Samples per language: {samples} | Languages: 6 | Total: ~{samples*6}",
        title="Employee Engagement NLP Pipeline"
    ))

    start = time.time()

    step_generate(samples)
    if not skip_hf:
        step_fetch_hf()
    step_preprocess()
    step_analyze(limit=limit)
    step_topics()

    elapsed = time.time() - start
    console.print(Panel(
        f"[bold green]✓ Pipeline complete in {elapsed:.0f}s[/bold green]\n"
        f"Dashboard: [cyan]streamlit run app/streamlit_app.py[/cyan]",
        title="Done"
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Employee Engagement NLP Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("step", choices=list(STEPS.keys()) + ["all"],
                        help="Pipeline step to run")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_LANGUAGE,
                        help=f"Samples per language (default: {SAMPLES_PER_LANGUAGE})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit rows for analyze step (testing)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for LLM API calls")
    parser.add_argument("--n-topics", type=int, default=15,
                        help="Number of topics for BERTopic")
    parser.add_argument("--drop-mismatches", action="store_true",
                        help="Drop language-mismatched rows in preprocessing")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace dataset fetch in 'all' step")
    args = parser.parse_args()

    if args.step == "generate":
        step_generate(args.samples)
    elif args.step == "fetch-hf":
        step_fetch_hf()
    elif args.step == "preprocess":
        step_preprocess(args.drop_mismatches)
    elif args.step == "analyze":
        step_analyze(limit=args.limit, batch_size=args.batch_size)
    elif args.step == "topics":
        step_topics(n_topics=args.n_topics)
    elif args.step == "all":
        run_all(samples=args.samples, limit=args.limit, skip_hf=args.skip_hf)


if __name__ == "__main__":
    main()
