"""BERTopic-based multilingual topic extraction service."""

import pickle
import structlog
import numpy as np
from pathlib import Path

from employee_engagement.config import get_settings
from employee_engagement.data.schemas import TopicAssignment

log = structlog.get_logger(__name__)


class TopicService:
    """
    Multilingual topic modeling using BERTopic + paraphrase-multilingual-MiniLM-L12-v2.
    Models are fit on first call to fit_transform(); subsequent calls use predict().
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self._topic_labels: dict[int, str] = {}
        self._embedding_model = None

    def _load_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._settings.embedding_model)
            log.info("embedding_model_loaded", model=self._settings.embedding_model)
        return self._embedding_model

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        cache_path = self._settings.processed_dir / "embeddings.pkl"
        if cache_path.exists():
            log.info("embeddings_cache_hit", path=str(cache_path))
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        log.info("computing_embeddings", n_texts=len(texts))
        embeddings = self._load_embedding_model().encode(
            texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

    def fit_transform(self, texts: list[str]) -> list[TopicAssignment]:
        """Fit BERTopic on texts and return topic assignments."""
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired

        embeddings = self._compute_embeddings(texts)
        representation_model = KeyBERTInspired()

        self._model = BERTopic(
            embedding_model=self._load_embedding_model(),
            representation_model=representation_model,
            nr_topics=self._settings.n_topics,
            min_topic_size=self._settings.min_topic_size,
            language="multilingual",
            calculate_probabilities=True,
            verbose=False,
        )
        topics, probs = self._model.fit_transform(texts, embeddings)
        self._build_labels()
        self._save_model()

        return [
            self._make_assignment(i, topic_id, probs[i] if probs is not None else np.array([]))
            for i, topic_id in enumerate(topics)
        ]

    def predict(self, texts: list[str]) -> list[TopicAssignment]:
        """Predict topics for new texts using a fitted model."""
        if self._model is None:
            self._load_model()
        embeddings = self._load_embedding_model().encode(texts, normalize_embeddings=True)
        topics, probs = self._model.transform(texts, embeddings)
        return [
            self._make_assignment(i, topic_id, probs[i] if probs is not None else np.array([]))
            for i, topic_id in enumerate(topics)
        ]

    def _make_assignment(self, idx: int, topic_id: int, probs) -> TopicAssignment:
        label = self._topic_labels.get(topic_id, f"Topic {topic_id}")
        keywords = []
        if self._model and topic_id != -1:
            topic_info = self._model.get_topic(topic_id)
            keywords = [word for word, _ in topic_info[:5]] if topic_info else []

        confidence = float(probs[topic_id]) if topic_id >= 0 and len(probs) > topic_id else 0.0
        return TopicAssignment(
            response_id=str(idx),
            topic_id=topic_id,
            topic_label=label,
            topic_keywords=keywords,
            topic_confidence=min(confidence, 1.0),
        )

    def _build_labels(self) -> None:
        if self._model:
            for topic_id, words in self._model.get_topics().items():
                top_words = [w for w, _ in words[:3]]
                self._topic_labels[topic_id] = " / ".join(top_words)

    def _save_model(self) -> None:
        path = self._settings.processed_dir / "bertopic_model"
        if self._model:
            self._model.save(str(path), serialization="safetensors", save_ctfidf=True)
            log.info("topic_model_saved", path=str(path))

    def _load_model(self) -> None:
        from bertopic import BERTopic
        path = self._settings.processed_dir / "bertopic_model"
        if path.exists():
            self._model = BERTopic.load(str(path))
            self._build_labels()
            log.info("topic_model_loaded", path=str(path))
        else:
            raise RuntimeError("No fitted BERTopic model found. Run fit_transform() first.")
