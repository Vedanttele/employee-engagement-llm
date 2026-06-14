"""FastAPI dependency injection — services are singletons per process."""

from functools import lru_cache
from employee_engagement.data.pii import PIIAnonymizer
from employee_engagement.services.sentiment import SentimentService
from employee_engagement.services.topics import TopicService
from employee_engagement.services.burnout import BurnoutDetector
from employee_engagement.services.rag import RAGKnowledgeBase
from employee_engagement.insights.generator import InsightsGenerator


@lru_cache(maxsize=1)
def get_pii_anonymizer() -> PIIAnonymizer:
    return PIIAnonymizer()


@lru_cache(maxsize=1)
def get_sentiment_service() -> SentimentService:
    return SentimentService()


@lru_cache(maxsize=1)
def get_topic_service() -> TopicService:
    return TopicService()


@lru_cache(maxsize=1)
def get_burnout_detector() -> BurnoutDetector:
    return BurnoutDetector()


@lru_cache(maxsize=1)
def get_rag() -> RAGKnowledgeBase:
    return RAGKnowledgeBase()


@lru_cache(maxsize=1)
def get_insights_generator() -> InsightsGenerator:
    return InsightsGenerator(rag=get_rag())
