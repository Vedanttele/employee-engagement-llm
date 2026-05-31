"""
topic_modeling.py — Multilingual topic discovery using BERTopic.

Uses paraphrase-multilingual-MiniLM-L12-v2 for embeddings:
- Handles all 6 languages natively
- Fast, lightweight (420MB), no GPU required
- Sentence-level semantic similarity

Output:
- Topic clusters with top keywords
- Per-response topic assignment
- Topic-sentiment cross-analysis

Usage:
    python -m src.topic_modeling \
        --input data/processed/analyzed_responses.csv \
        --output data/processed/topics.csv
"""

import pandas as pd
import numpy as np
import json
import argparse
import pickle
from pathlib import Path
from rich.console import Console
from rich.table import Table

from src.config import PROCESSED_DIR, ENGAGEMENT_THEMES, SUPPORTED_LANGUAGES

console = Console()

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_or_compute_embeddings(
    texts: list[str],
    cache_path: Path,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Compute or load cached embeddings.
    Caching is critical — embeddings take ~2 min for 3k texts.
    """
    if cache_path.exists() and not force_recompute:
        console.print(f"[yellow]Loading cached embeddings: {cache_path}[/yellow]")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    console.print(f"[cyan]Computing embeddings with {EMBEDDING_MODEL}...[/cyan]")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)
    console.print(f"[green]Embeddings cached: {cache_path}[/green]")
    return embeddings


def run_topic_modeling(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_topics: int = 15,
    min_topic_size: int = 10,
) -> tuple[object, pd.DataFrame]:
    """
    Run BERTopic with tuned parameters for HR survey data.

    Key choices:
    - min_topic_size=10: avoids tiny noisy clusters
    - nr_topics=n_topics: reduce to interpretable number
    - Using HDBSCAN (default) for density-based clustering
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    # Tuned UMAP for short texts
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN tuned for HR survey granularity
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    console.print("[cyan]Fitting BERTopic...[/cyan]")
    topics, probs = topic_model.fit_transform(df["response_text"].tolist(), embeddings)

    df = df.copy()
    df["topic_id"] = topics
    df["topic_probability"] = [p.max() if hasattr(p, "max") else p for p in probs]

    # Map topic IDs to human-readable labels
    topic_info = topic_model.get_topic_info()
    topic_labels = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_labels[tid] = "Outliers / Mixed"
        else:
            # Take top 3 words as label
            top_words = [w for w, _ in topic_model.get_topic(tid)[:3]]
            topic_labels[tid] = " | ".join(top_words)

    df["topic_label"] = df["topic_id"].map(topic_labels)
    return topic_model, df


def map_topics_to_themes(
    topic_model: object,
    engagement_themes: dict = ENGAGEMENT_THEMES,
) -> dict[int, str]:
    """
    Map discovered BERTopic topics to our pre-defined engagement themes.
    Simple keyword matching — not perfect, but interpretable.
    """
    theme_keywords = {
        "workload_balance":      ["workload", "balance", "overtime", "stress", "hours", "burnout",
                                  "Überlastung", "équilibre", "carga"],
        "management_leadership": ["manager", "management", "leader", "boss", "supervisor",
                                  "Chef", "direction", "gestión"],
        "recognition_rewards":   ["recognition", "reward", "pay", "salary", "bonus", "appreciate",
                                  "Anerkennung", "salaire", "reconocimiento"],
        "career_growth":         ["career", "growth", "promotion", "learning", "development", "training",
                                  "Karriere", "Entwicklung", "carrière"],
        "team_collaboration":    ["team", "collaboration", "colleagues", "culture", "together",
                                  "Team", "collègues", "equipo"],
        "company_strategy":      ["strategy", "vision", "direction", "future", "goals",
                                  "Strategie", "stratégie", "estrategia"],
        "tools_resources":       ["tools", "resources", "equipment", "software", "technology",
                                  "Werkzeuge", "outils", "herramientas"],
        "inclusion_belonging":   ["diversity", "inclusion", "belonging", "equity", "respect",
                                  "Vielfalt", "diversité", "diversidad"],
        "wellbeing_safety":      ["wellbeing", "health", "safety", "mental", "support",
                                  "Gesundheit", "santé", "bienestar"],
        "communication":         ["communication", "transparency", "information", "feedback",
                                  "Kommunikation", "transparence", "comunicación"],
    }

    topic_to_theme = {}
    for topic_id in set(topic_model.get_topics().keys()):
        if topic_id == -1:
            topic_to_theme[topic_id] = "mixed"
            continue

        topic_words = [w.lower() for w, _ in topic_model.get_topic(topic_id)]
        best_theme = None
        best_score = 0

        for theme, keywords in theme_keywords.items():
            score = sum(1 for kw in keywords if any(kw.lower() in w for w in topic_words))
            if score > best_score:
                best_score = score
                best_theme = theme

        topic_to_theme[topic_id] = best_theme or "other"

    return topic_to_theme


def run_full_pipeline(
    input_path: Path,
    output_path: Path = PROCESSED_DIR / "topics.csv",
    model_path: Path = PROCESSED_DIR / "topic_model",
    n_topics: int = 15,
    force_recompute_embeddings: bool = False,
) -> pd.DataFrame:
    """Full topic modeling pipeline."""
    df = pd.read_csv(input_path)
    console.print(f"Loaded {len(df)} rows")

    # Filter out failed analyses
    if "analysis_failed" in df.columns:
        df = df[~df["analysis_failed"].fillna(False)].copy()
        console.print(f"After filtering failures: {len(df)} rows")

    # Embeddings
    embed_cache = PROCESSED_DIR / "embeddings_cache.pkl"
    embeddings = load_or_compute_embeddings(
        df["response_text"].tolist(),
        cache_path=embed_cache,
        force_recompute=force_recompute_embeddings,
    )

    # Topic modeling
    topic_model, df_topics = run_topic_modeling(
        df, embeddings, n_topics=n_topics
    )

    # Map to themes
    topic_theme_map = map_topics_to_themes(topic_model)
    df_topics["discovered_theme"] = df_topics["topic_id"].map(topic_theme_map)

    # Save model
    model_path.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
    console.print(f"[green]✓ Topic model saved: {model_path}[/green]")

    # Save enriched dataset
    df_topics.to_csv(output_path, index=False)
    console.print(f"[green]✓ Topics saved: {output_path}[/green]")

    # Summary
    _print_topic_summary(df_topics, topic_model)
    return df_topics


def _print_topic_summary(df: pd.DataFrame, topic_model: object):
    """Print topic distribution summary."""
    table = Table(title=f"Discovered Topics (n={df['topic_id'].nunique() - 1})")
    table.add_column("Topic ID")
    table.add_column("Label")
    table.add_column("Count")
    table.add_column("Top Sentiment")

    topic_counts = df[df["topic_id"] != -1]["topic_id"].value_counts()

    for tid, count in topic_counts.head(15).items():
        sub = df[df["topic_id"] == tid]
        top_sentiment = sub["llm_sentiment"].mode()[0] if "llm_sentiment" in sub.columns else "N/A"
        words = [w for w, _ in topic_model.get_topic(tid)[:4]]
        table.add_row(str(tid), " | ".join(words), str(count), top_sentiment)

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multilingual topic modeling")
    parser.add_argument("--input", type=str,
                        default=str(PROCESSED_DIR / "analyzed_responses.csv"))
    parser.add_argument("--output", type=str,
                        default=str(PROCESSED_DIR / "topics.csv"))
    parser.add_argument("--n-topics", type=int, default=15)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    run_full_pipeline(
        input_path=Path(args.input),
        output_path=Path(args.output),
        n_topics=args.n_topics,
        force_recompute_embeddings=args.force_recompute,
    )
