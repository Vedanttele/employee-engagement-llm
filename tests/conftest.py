"""Shared test fixtures."""

import pytest
from fastapi.testclient import TestClient
from employee_engagement.api.main import create_app


@pytest.fixture(scope="session")
def app():
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    return TestClient(app)


@pytest.fixture
def sample_responses() -> list[dict]:
    return [
        {
            "response_id": "t1",
            "text": "Great team culture and excellent leadership support.",
            "department": "Engineering",
            "language": "en",
        },
        {
            "response_id": "t2",
            "text": "Workload is consistently overwhelming. No work-life balance.",
            "department": "Operations",
            "language": "en",
        },
        {
            "response_id": "t3",
            "text": "Career growth is unclear. Need better development plans.",
            "department": "Sales",
            "language": "en",
        },
    ]
