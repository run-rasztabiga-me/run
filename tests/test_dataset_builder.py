"""Unit tests for the repository dataset builder helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from src.evaluator.experiments.dataset_builder import (
    DatasetMetadata,
    RepositoryDataset,
    RepositoryFilters,
    RepositoryRecord,
)


def make_record(idx: int) -> RepositoryRecord:
    """Helper to create deterministic repository records for tests."""
    return RepositoryRecord(
        id=idx,
        name=f"repo-{idx}",
        full_name=f"owner/repo-{idx}",
        html_url=f"https://github.com/owner/repo-{idx}",
        clone_url=f"https://github.com/owner/repo-{idx}.git",
        stargazers_count=idx * 10,
    )


def test_build_search_query() -> None:
    filters = RepositoryFilters(
        query="web framework",
        languages=["Python", "JavaScript"],
        topics=["webapp", "fastapi"],
        min_stars=100,
        max_stars=5000,
    )

    query = filters.build_search_query()

    assert "web framework" in query
    assert "language:Python" in query
    assert "language:JavaScript" in query
    assert "topic:webapp" in query
    assert "topic:fastapi" in query
    assert "stars:100..5000" in query
    assert "fork:false" in query
    assert "archived:false" in query


def test_select_repositories_top_and_sample() -> None:
    metadata = DatasetMetadata(
        name="sample",
        generated_at=datetime.now(timezone.utc),
        total_count=4,
        limit=4,
        sort="stars",
        order="desc",
        search_query="language:Python",
        filters=RepositoryFilters(),
        github_api_url="https://api.github.com",
    )
    dataset = RepositoryDataset(
        metadata=metadata,
        repositories=[make_record(idx) for idx in range(4)],
        selection_seed=42,
    )

    top_two = dataset.select_repositories(top_n=2)
    assert [record.full_name for record in top_two] == ["owner/repo-0", "owner/repo-1"]

    sampled = dataset.select_repositories(top_n=4, sample_size=2)
    assert len(sampled) == 2
    # selection_seed ensures reproducible order: indices [0,1,2,3] sampled with seed=42 -> [0,3]
    assert [record.full_name for record in sampled] == ["owner/repo-0", "owner/repo-3"]


def test_to_experiment_suite_round_trip() -> None:
    metadata = DatasetMetadata(
        name="integration",
        generated_at=datetime.now(timezone.utc),
        total_count=2,
        limit=2,
        sort="stars",
        order="desc",
        search_query="language:Python",
        filters=RepositoryFilters(),
        github_api_url="https://api.github.com",
    )
    dataset = RepositoryDataset(
        metadata=metadata,
        repositories=[make_record(1), make_record(2)],
    )

    suite = dataset.to_experiment_suite(
        experiment_name="python-webapps",
        models=[
            {"provider": "openai", "name": "gpt-4o-mini"},
        ],
        top_n=1,
    )

    assert len(suite.experiments) == 1
    experiment = suite.experiments[0]
    assert experiment.repos == ["https://github.com/owner/repo-1.git"]
    assert experiment.models[0].provider == "openai"
    assert experiment.models[0].name == "gpt-4o-mini"
