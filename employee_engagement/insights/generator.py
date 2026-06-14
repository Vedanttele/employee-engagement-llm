"""LLM Insights Generator — produces executive reports from analyzed survey data."""

import json
import structlog
from collections import Counter
from tenacity import retry, stop_after_attempt, wait_fixed

import anthropic

from employee_engagement.config import get_settings, ENGAGEMENT_THEMES
from employee_engagement.data.schemas import AnalyzedResponse, InsightReport
from employee_engagement.services.rag import RAGKnowledgeBase

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are a senior HR analytics consultant generating an executive insights report
from employee engagement survey analysis data.

Your report must be:
- Evidence-based: cite the numbers from the data provided
- Actionable: every finding must pair with a concrete recommended action
- Concise: executive audience, C-suite level

Return ONLY valid JSON with exactly this structure:
{
  "executive_summary": "2-3 sentence strategic summary",
  "key_findings": ["finding 1", "finding 2", "finding 3", "finding 4", "finding 5"],
  "recommended_actions": [
    {"priority": "high|medium|low", "theme": "<theme_key>", "action": "<specific action>", "owner": "<HR|Management|Leadership>"},
    ...
  ]
}"""


class InsightsGenerator:
    """Generates LLM-powered insight reports using analysis results + RAG context."""

    def __init__(self, rag: RAGKnowledgeBase | None = None) -> None:
        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model
        self._rag = rag or RAGKnowledgeBase()

    def generate(self, responses: list[AnalyzedResponse]) -> InsightReport:
        """Generate a full insight report from a cohort of analyzed responses."""
        if not responses:
            raise ValueError("Cannot generate insights from empty response list")

        stats = self._compute_stats(responses)
        context_queries = self._build_rag_queries(stats)
        rag_docs = self._retrieve_rag_context(context_queries)

        llm_output = self._call_claude(stats, rag_docs)

        return InsightReport(
            total_responses=stats["total"],
            sentiment_distribution=stats["sentiment_dist"],
            top_themes=stats["top_themes"],
            burnout_risk_summary=stats["burnout_summary"],
            executive_summary=llm_output.get("executive_summary", ""),
            key_findings=llm_output.get("key_findings", []),
            recommended_actions=llm_output.get("recommended_actions", []),
            rag_references=rag_docs[:3],
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call_claude(self, stats: dict, rag_docs: list[str]) -> dict:
        rag_context = "\n\n".join(f"- {doc}" for doc in rag_docs) if rag_docs else "N/A"
        user_prompt = f"""Survey Analysis Statistics:
{json.dumps(stats, indent=2)}

HR Knowledge Base Context (use to ground recommendations):
{rag_context}

Generate the executive insights report as JSON."""

        message = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return json.loads(message.content[0].text)

    @staticmethod
    def _compute_stats(responses: list[AnalyzedResponse]) -> dict:
        sentiment_dist = Counter(r.sentiment.value for r in responses)
        theme_dist = Counter(r.primary_theme for r in responses)
        burnout_levels = Counter(r.burnout_risk_level.value for r in responses)
        dept_dist = Counter(r.department for r in responses if r.department)
        action_dist = Counter(r.action_signal.value for r in responses)

        avg_burnout = sum(r.burnout_risk_score for r in responses) / len(responses)
        urgent = sum(1 for r in responses if r.action_signal.value == "urgent_action")

        return {
            "total": len(responses),
            "sentiment_dist": dict(sentiment_dist),
            "top_themes": [t for t, _ in theme_dist.most_common(5)],
            "theme_distribution": dict(theme_dist),
            "burnout_summary": dict(burnout_levels),
            "avg_burnout_score": round(avg_burnout, 3),
            "urgent_action_count": urgent,
            "action_distribution": dict(action_dist),
            "department_distribution": dict(dept_dist.most_common(5)),
        }

    def _build_rag_queries(self, stats: dict) -> list[str]:
        queries = []
        for theme in stats.get("top_themes", [])[:3]:
            display = ENGAGEMENT_THEMES.get(theme, theme)
            queries.append(f"best practices for {display} employee engagement")
        if stats.get("avg_burnout_score", 0) > 0.55:
            queries.append("burnout prevention and crisis intervention")
        return queries

    def _retrieve_rag_context(self, queries: list[str]) -> list[str]:
        docs: list[str] = []
        seen: set[str] = set()
        for q in queries:
            try:
                for doc in self._rag.retrieve(q):
                    if doc not in seen:
                        seen.add(doc)
                        docs.append(doc)
            except Exception as exc:
                log.warning("rag_retrieve_failed", query=q, error=str(exc))
        return docs
