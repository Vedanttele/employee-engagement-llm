"""
RAG Knowledge Base using ChromaDB.

Stores HR best-practice documents; retrieved context is passed to the
LLM Insights Generator to produce grounded, actionable recommendations.
"""

import structlog
from pathlib import Path

from employee_engagement.config import get_settings

log = structlog.get_logger(__name__)

# Built-in HR knowledge seed documents
_SEED_DOCUMENTS: list[dict] = [
    {
        "id": "wb_001",
        "topic": "workload_balance",
        "text": (
            "Workload imbalance is the top predictor of voluntary attrition. "
            "Effective interventions: workload audits every quarter, explicit capacity planning "
            "in sprint/project cycles, and manager training on sustainable pacing."
        ),
    },
    {
        "id": "wb_002",
        "topic": "workload_balance",
        "text": (
            "Flexible work arrangements reduce reported burnout by 23% (Gallup 2023). "
            "Async-first policies and protected focus time blocks are the highest-impact levers."
        ),
    },
    {
        "id": "ml_001",
        "topic": "management_leadership",
        "text": (
            "Manager quality explains 70% of variance in team engagement scores. "
            "Best-in-class programs: bi-weekly 1:1s with structured templates, "
            "skip-level meetings quarterly, and 360 feedback cycles every 6 months."
        ),
    },
    {
        "id": "rr_001",
        "topic": "recognition_rewards",
        "text": (
            "Peer recognition programs increase engagement by up to 36%. "
            "Timely recognition (within 24h) is more impactful than monetary rewards. "
            "Recommended: lightweight recognition tooling (Kudos, Lattice) embedded in daily workflow."
        ),
    },
    {
        "id": "cg_001",
        "topic": "career_growth",
        "text": (
            "Lack of career clarity is the #2 driver of disengagement for tenured employees. "
            "Individual Development Plans (IDPs) reviewed quarterly and tied to project assignments "
            "reduce churn by 18% (LinkedIn Workforce Report)."
        ),
    },
    {
        "id": "ws_001",
        "topic": "wellbeing_safety",
        "text": (
            "Psychological safety — the belief that one can speak up without fear of punishment — "
            "is the strongest predictor of high-performing teams (Google Project Aristotle). "
            "Establish: anonymous feedback channels, blameless post-mortems, "
            "and explicit norms around respectful disagreement."
        ),
    },
    {
        "id": "ws_002",
        "topic": "wellbeing_safety",
        "text": (
            "Burnout crisis intervention: immediate workload relief (reassign tasks), "
            "access to EAP (Employee Assistance Program), confidential 1:1 with HR, "
            "and 2-4 week follow-up check-in. Document and escalate to senior leadership."
        ),
    },
    {
        "id": "tc_001",
        "topic": "team_collaboration",
        "text": (
            "Distributed team cohesion is built through structured rituals: "
            "weekly async standups, monthly virtual socials, and cross-functional project pairing. "
            "Avoid over-reliance on synchronous meetings — meeting load is a top complaint."
        ),
    },
    {
        "id": "comm_001",
        "topic": "communication",
        "text": (
            "Communication opacity (lack of strategic transparency) erodes trust faster than "
            "any other factor. Best practices: monthly all-hands with live Q&A, "
            "decision logs in a searchable wiki, and executive AMAs."
        ),
    },
    {
        "id": "ib_001",
        "topic": "inclusion_belonging",
        "text": (
            "Belonging scores predict retention 2x better than satisfaction scores. "
            "Actionable levers: inclusive hiring panels, ERG funding, "
            "pay equity audits, and bias-free performance review calibration."
        ),
    },
]


class RAGKnowledgeBase:
    """
    ChromaDB-backed HR knowledge base for retrieval-augmented generation.
    Lazy-initialized: ChromaDB client is created on first use.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = None
        self._collection = None

    def _init(self) -> None:
        if self._collection is not None:
            return
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_path = str(self._settings.rag_dir)
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._settings.rag_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Seed if empty
        if self._collection.count() == 0:
            self._seed()

    def _seed(self) -> None:
        log.info("rag_seeding", n_docs=len(_SEED_DOCUMENTS))
        self._collection.add(
            documents=[d["text"] for d in _SEED_DOCUMENTS],
            ids=[d["id"] for d in _SEED_DOCUMENTS],
            metadatas=[{"topic": d["topic"]} for d in _SEED_DOCUMENTS],
        )

    def retrieve(self, query: str, n_results: int | None = None) -> list[str]:
        """Retrieve top-k relevant HR knowledge snippets for a query."""
        self._init()
        k = n_results or self._settings.rag_top_k
        results = self._collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        log.debug("rag_retrieved", query=query[:60], n=len(docs))
        return docs

    def add_documents(self, documents: list[dict]) -> None:
        """Add custom HR documents. Each dict must have: id, text, topic."""
        self._init()
        self._collection.add(
            documents=[d["text"] for d in documents],
            ids=[d["id"] for d in documents],
            metadatas=[{"topic": d.get("topic", "general")} for d in documents],
        )
        log.info("rag_documents_added", n=len(documents))
