"""Utilities for discovering repository datasets and emitting experiment configs."""

from __future__ import annotations

import json
import logging
import os
import random
from fnmatch import fnmatch
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .config import (
    ExperimentDefinition,
    ExperimentSuite,
    ModelSpec,
    PromptVariant,
)

logger = logging.getLogger(__name__)


class RepositoryFilters(BaseModel):
    """Search filters applied when querying GitHub repositories."""

    query: Optional[str] = Field(
        default=None,
        description="Freeform search terms passed directly to GitHub search.",
    )
    languages: List[str] = Field(
        default_factory=list,
        description="Programming language filters (all applied).",
    )
    topics: List[str] = Field(
        default_factory=list,
        description="GitHub topics to require (all applied).",
    )
    min_stars: Optional[int] = Field(
        default=None,
        ge=0,
        description="Lower bound for stargazers.",
    )
    max_stars: Optional[int] = Field(
        default=None,
        ge=0,
        description="Upper bound for stargazers.",
    )
    include_forks: bool = Field(
        default=False,
        description="Include forked repositories when set to True.",
    )
    include_archived: bool = Field(
        default=False,
        description="Include archived repositories when set to True.",
    )
    required_files: List[str] = Field(
        default_factory=list,
        description="File glob patterns that must exist somewhere in the repository tree.",
    )

    @field_validator("languages", "topics", mode="before")
    @classmethod
    def _normalize_collection(cls, value: Iterable[str] | None) -> List[str]:
        if value is None:
            return []
        return [item.strip() for item in value if item and item.strip()]
    
    @field_validator("required_files", mode="before")
    @classmethod
    def _normalize_required_files(cls, value: Iterable[str] | None) -> List[str]:
        if value is None:
            return []
        return [item.strip() for item in value if item and item.strip()]

    @model_validator(mode="after")
    def _validate_stars(self) -> "RepositoryFilters":
        if self.min_stars is not None and self.max_stars is not None:
            if self.min_stars > self.max_stars:
                raise ValueError("min_stars cannot be greater than max_stars.")
        return self

    def build_search_query(self) -> str:
        """Assemble the GitHub search query string."""
        clauses: List[str] = []

        if self.query:
            clauses.append(self.query.strip())

        clauses.extend(f"language:{language}" for language in self.languages)
        clauses.extend(f"topic:{topic}" for topic in self.topics)

        if self.min_stars is not None and self.max_stars is not None:
            clauses.append(f"stars:{self.min_stars}..{self.max_stars}")
        elif self.min_stars is not None:
            clauses.append(f"stars:>={self.min_stars}")
        elif self.max_stars is not None:
            clauses.append(f"stars:<={self.max_stars}")

        if not self.include_forks:
            clauses.append("fork:false")

        if not self.include_archived:
            clauses.append("archived:false")

        return " ".join(clauses).strip()


class RepositoryRecord(BaseModel):
    """Normalized repository metadata persisted in datasets."""

    repository_id: int = Field(alias="id")
    name: str
    full_name: str
    html_url: str
    clone_url: str
    description: Optional[str] = None
    language: Optional[str] = None
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    topics: List[str] = Field(default_factory=list)
    default_branch: Optional[str] = None
    pushed_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    archived: bool = False
    fork: bool = False
    owner: Optional[str] = None
    license: Optional[Dict[str, Any]] = None
    size: Optional[int] = None
    visibility: Optional[str] = None
    homepage: Optional[str] = None

    @field_validator("topics", mode="before")
    @classmethod
    def _ensure_topics(cls, value: Iterable[str] | None) -> List[str]:
        if value is None:
            return []
        return [topic for topic in value if topic]

    @field_validator("owner", mode="before")
    @classmethod
    def _extract_owner(cls, value: Any) -> Optional[str]:
        if isinstance(value, dict):
            return value.get("login") or value.get("name")
        return value


class DatasetMetadata(BaseModel):
    """Metadata describing how the dataset was produced."""

    name: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_count: int
    limit: int
    sort: str
    order: str
    search_query: str
    filters: RepositoryFilters
    github_api_url: str
    topics_included: bool = True
    note: Optional[str] = None

    @field_validator("generated_at", mode="before")
    @classmethod
    def _ensure_datetime(cls, value: datetime | str) -> datetime:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        return datetime.fromisoformat(value).astimezone(timezone.utc)


class RepositoryDataset(BaseModel):
    """Container that holds repository metadata and provenance details."""

    metadata: DatasetMetadata
    repositories: List[RepositoryRecord]
    selection_seed: Optional[int] = Field(
        default=None,
        description="Optional seed stored with the dataset for reproducible sampling.",
    )

    def save_json(self, path: str | Path) -> Path:
        """Persist the dataset as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(
                indent=2,
                by_alias=True,
                exclude_none=True,
            ),
            encoding="utf-8",
        )
        logger.info("Repository dataset saved to %s", path)
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "RepositoryDataset":
        """Load a dataset produced by :meth:`save_json`."""
        path = Path(path)
        payload = path.read_text(encoding="utf-8")
        try:
            return cls.model_validate_json(payload)
        except ValidationError as exc:
            raise ValueError(f"Failed to parse dataset at {path}: {exc}") from exc

    def select_repositories(
        self,
        *,
        top_n: Optional[int] = None,
        sample_size: Optional[int] = None,
        sample_seed: Optional[int] = None,
    ) -> List[RepositoryRecord]:
        """Return a deterministic subset of repositories."""
        candidates: List[RepositoryRecord] = list(self.repositories)

        if top_n is not None:
            if top_n < 1:
                raise ValueError("top_n must be >= 1 when provided.")
            candidates = candidates[:top_n]

        if sample_size is None:
            return candidates

        if sample_size < 1:
            raise ValueError("sample_size must be >= 1 when provided.")

        if sample_size > len(candidates):
            raise ValueError(
                f"Requested sample_size {sample_size} exceeds available repositories ({len(candidates)})."
            )

        seed = sample_seed
        if seed is None and self.selection_seed is not None:
            seed = self.selection_seed

        rng = random.Random(seed)
        indices = rng.sample(range(len(candidates)), sample_size)
        indices.sort()
        return [candidates[idx] for idx in indices]

    def to_experiment_suite(
        self,
        *,
        experiment_name: str,
        models: Sequence[ModelSpec | Dict[str, Any]],
        description: Optional[str] = None,
        top_n: Optional[int] = None,
        sample_size: Optional[int] = None,
        sample_seed: Optional[int] = None,
        repetitions: int = 1,
        generator_overrides: Optional[Dict[str, Any]] = None,
        run_overrides: Optional[Dict[str, Any]] = None,
        prompts: Optional[Sequence[PromptVariant | Dict[str, Any]]] = None,
        cleanup: Optional[str] = None,
    ) -> ExperimentSuite:
        """Build an :class:`ExperimentSuite` referencing this dataset."""
        repositories = self.select_repositories(
            top_n=top_n,
            sample_size=sample_size,
            sample_seed=sample_seed,
        )
        repo_urls = [record.clone_url for record in repositories if record.clone_url]
        if not repo_urls:
            raise ValueError("No repositories available after applying selection filters.")

        model_specs: List[ModelSpec] = []
        for model in models:
            if isinstance(model, ModelSpec):
                model_specs.append(model)
            else:
                model_specs.append(ModelSpec(**model))

        prompt_specs: List[PromptVariant] = []
        if prompts:
            for prompt in prompts:
                if isinstance(prompt, PromptVariant):
                    prompt_specs.append(prompt)
                else:
                    prompt_specs.append(PromptVariant(**prompt))

        overrides = dict(run_overrides or {})
        # Extract cleanup from run_overrides if present there (for backwards compatibility)
        if cleanup is None and "cleanup" in overrides:
            cleanup = overrides.pop("cleanup")

        experiment_kwargs = {
            "name": experiment_name,
            "description": description,
            "repos": repo_urls,
            "models": model_specs,
            "repetitions": repetitions,
            "prompts": prompt_specs,
        }

        # Only include generator_overrides if not empty
        if generator_overrides:
            experiment_kwargs["generator_overrides"] = generator_overrides

        # Only include run_overrides if not empty
        if overrides:
            experiment_kwargs["run_overrides"] = overrides

        if cleanup is not None:
            experiment_kwargs["cleanup"] = cleanup

        experiment = ExperimentDefinition(**experiment_kwargs)
        return ExperimentSuite(experiments=[experiment])

    def write_experiment_yaml(
        self,
        output_path: str | Path,
        *,
        experiment_name: str,
        models: Sequence[ModelSpec | Dict[str, Any]],
        description: Optional[str] = None,
        top_n: Optional[int] = None,
        sample_size: Optional[int] = None,
        sample_seed: Optional[int] = None,
        repetitions: int = 1,
        generator_overrides: Optional[Dict[str, Any]] = None,
        run_overrides: Optional[Dict[str, Any]] = None,
        prompts: Optional[Sequence[PromptVariant | Dict[str, Any]]] = None,
        cleanup: Optional[str] = None,
        indent: int = 2,
    ) -> Path:
        """Materialise an experiment YAML file derived from this dataset."""
        suite = self.to_experiment_suite(
            experiment_name=experiment_name,
            models=models,
            description=description,
            top_n=top_n,
            sample_size=sample_size,
            sample_seed=sample_seed,
            repetitions=repetitions,
            generator_overrides=generator_overrides,
            run_overrides=run_overrides,
            prompts=prompts,
            cleanup=cleanup,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = suite.model_dump(mode='json', exclude_none=True)

        # Remove empty defaults from experiment definitions
        for exp in payload.get('experiments', []):
            # Remove empty dicts and lists
            if exp.get('generator_overrides') == {}:
                exp.pop('generator_overrides', None)
            if exp.get('run_overrides') == {}:
                exp.pop('run_overrides', None)
            if exp.get('prompts') == []:
                exp.pop('prompts', None)

        output_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, indent=indent),
            encoding="utf-8",
        )
        logger.info("Experiment configuration written to %s", output_path)
        return output_path


class GitHubSearchError(RuntimeError):
    """Raised when GitHub search fails."""


@dataclass
class GitHubSearchResult:
    """Result wrapper including payload and pagination metadata."""

    items: List[RepositoryRecord]
    total_count: int


class RepositoryDatasetBuilder:
    """High-level builder that orchestrates GitHub discovery and persistence."""

    SEARCH_ENDPOINT = "search/repositories"
    TOPICS_ENDPOINT = "repos/{full_name}/topics"

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        base_url: str = "https://api.github.com",
        per_page: int = 100,
        request_timeout: int = 30,
        include_topics: bool = True,
        default_dataset_dir: str | Path = "evaluation_reports/datasets",
    ):
        self.base_url = base_url.rstrip("/")
        self.per_page = max(1, min(per_page, 100))
        self.request_timeout = request_timeout
        self.include_topics = include_topics
        self.dataset_dir = Path(default_dataset_dir)
        self.session = requests.Session()
        self._tree_cache: Dict[str, List[str]] = {}
        resolved_token = token or self._resolve_token()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "repository-dataset-builder",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token}"
        self.session.headers.update(headers)

    @staticmethod
    def _resolve_token() -> Optional[str]:
        """Read a GitHub token from common environment variables."""
        for variable in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_API_TOKEN"):
            value = os.getenv(variable)
            if value:
                return value.strip()
        return None

    def discover(
        self,
        *,
        name: str,
        filters: RepositoryFilters,
        limit: int = 50,
        sort: str = "stars",
        order: str = "desc",
        selection_seed: Optional[int] = None,
        note: Optional[str] = None,
        shuffle: bool = False,
        fetch_multiple: int = 1,
    ) -> RepositoryDataset:
        """Discover repositories via the GitHub Search API.

        Args:
            name: Dataset name
            filters: Repository filters to apply
            limit: Number of repositories to return
            sort: GitHub sort field
            order: Sort order (asc/desc)
            selection_seed: Seed for reproducible sampling/shuffling
            note: Optional metadata note
            shuffle: Whether to shuffle results (using selection_seed)
            fetch_multiple: Fetch this many times the limit, then randomly sample
                          (useful for getting diverse results). Must be >= 1.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1.")
        if fetch_multiple < 1:
            raise ValueError("fetch_multiple must be >= 1.")

        fetch_limit = limit * fetch_multiple
        search_query = filters.build_search_query()
        logger.info(
            "Searching GitHub repositories with query=%r sort=%s order=%s limit=%d (fetching %d)",
            search_query,
            sort,
            order,
            limit,
            fetch_limit,
        )
        result = self._search_repositories(
            query=search_query,
            sort=sort,
            order=order,
            limit=fetch_limit,
        )

        repositories: List[RepositoryRecord] = result.items
        if self.include_topics:
            repositories = self._enrich_topics(repositories)
        if filters.required_files:
            repositories = self._filter_by_required_files(repositories, filters.required_files)

        # Apply shuffling or random sampling if requested
        if shuffle or fetch_multiple > 1:
            seed = selection_seed if selection_seed is not None else int(datetime.now(timezone.utc).timestamp())
            rng = random.Random(seed)

            if fetch_multiple > 1 and len(repositories) > limit:
                # Randomly sample from the larger set
                repositories = rng.sample(repositories, min(limit, len(repositories)))
                logger.info("Randomly sampled %d repositories from %d candidates (seed=%d)",
                           len(repositories), len(result.items), seed)
            elif shuffle:
                # Just shuffle the results
                rng.shuffle(repositories)
                logger.info("Shuffled %d repositories (seed=%d)", len(repositories), seed)

        metadata = DatasetMetadata(
            name=name,
            total_count=result.total_count,
            limit=limit,
            sort=sort,
            order=order,
            search_query=search_query,
            filters=filters,
            github_api_url=self.base_url,
            topics_included=self.include_topics,
            note=note,
        )
        if selection_seed is None:
            selection_seed = int(metadata.generated_at.timestamp())
        dataset = RepositoryDataset(
            metadata=metadata,
            repositories=repositories,
            selection_seed=selection_seed,
        )
        return dataset

    def discover_and_save(
        self,
        *,
        name: str,
        filters: RepositoryFilters,
        limit: int = 50,
        sort: str = "stars",
        order: str = "desc",
        output_path: Optional[str | Path] = None,
        selection_seed: Optional[int] = None,
        note: Optional[str] = None,
        shuffle: bool = False,
        fetch_multiple: int = 1,
    ) -> RepositoryDataset:
        """Discover repositories and persist the dataset to disk."""
        dataset = self.discover(
            name=name,
            filters=filters,
            limit=limit,
            sort=sort,
            order=order,
            selection_seed=selection_seed,
            note=note,
            shuffle=shuffle,
            fetch_multiple=fetch_multiple,
        )
        target = Path(output_path) if output_path else self.dataset_dir / f"{name}.json"
        dataset.save_json(target)
        return dataset

    def _search_repositories(
        self,
        *,
        query: str,
        sort: str,
        order: str,
        limit: int,
    ) -> GitHubSearchResult:
        items: List[RepositoryRecord] = []
        page = 1
        total_count = 0
        remaining = limit

        while remaining > 0:
            per_page = min(self.per_page, remaining)
            url = f"{self.base_url}/{self.SEARCH_ENDPOINT}"
            params = {
                "q": query or "",
                "sort": sort,
                "order": order,
                "per_page": per_page,
                "page": page,
            }
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            if response.status_code == 403 and "rate limit" in response.text.lower():
                reset = response.headers.get("X-RateLimit-Reset")
                raise GitHubSearchError(
                    "GitHub API rate limit exceeded. "
                    "Set GITHUB_TOKEN to increase the quota."
                    + (f" Rate limit resets at {reset}." if reset else "")
                )
            if response.status_code >= 400:
                raise GitHubSearchError(
                    f"GitHub search failed with status {response.status_code}: {response.text}"
                )

            payload = response.json()
            if total_count == 0:
                total_count = payload.get("total_count", 0)

            raw_items = payload.get("items", []) or []
            if not raw_items:
                break

            for raw in raw_items:
                try:
                    record = RepositoryRecord.model_validate(raw)
                except ValidationError as exc:
                    logger.warning("Skipping repository due to validation error: %s", exc)
                    continue
                if record.owner is None:
                    owner = raw.get("owner") or {}
                    record.owner = owner.get("login")
                items.append(record)
                remaining -= 1
                if remaining <= 0:
                    break

            if remaining <= 0:
                break

            page += 1
            if page > 10:
                logger.warning(
                    "GitHub Search API pagination capped at 10 pages; results may be truncated."
                )
                break

        return GitHubSearchResult(items=items, total_count=total_count)

    def _filter_by_required_files(
        self,
        repositories: List[RepositoryRecord],
        patterns: Sequence[str],
    ) -> List[RepositoryRecord]:
        """Filter repositories by verifying required files exist in their tree."""
        normalized_patterns = [pattern.lower() for pattern in patterns if pattern]
        if not normalized_patterns:
            return repositories

        matched: List[RepositoryRecord] = []
        for record in repositories:
            paths = self._fetch_repository_tree_paths(record)
            if not paths:
                logger.debug("Skipping %s; unable to fetch repository tree.", record.full_name)
                continue
            if self._paths_match_required_files(paths, normalized_patterns):
                matched.append(record)
        logger.info(
            "Repository file filtering (%s) reduced set from %d to %d.",
            ", ".join(patterns),
            len(repositories),
            len(matched),
        )
        return matched

    def _fetch_repository_tree_paths(self, record: RepositoryRecord) -> Optional[List[str]]:
        """Fetch the recursive tree listing for a repository's default branch."""
        if not record.full_name:
            return None
        branch = record.default_branch or "main"
        cache_key = f"{record.full_name}@{branch}"
        if cache_key in self._tree_cache:
            return self._tree_cache[cache_key]

        url = f"{self.base_url}/repos/{record.full_name}/git/trees/{branch}"
        params = {"recursive": "1"}
        response = self.session.get(url, params=params, timeout=self.request_timeout)
        if response.status_code != 200:
            logger.warning(
                "Failed to fetch tree for %s (branch %s): %s",
                record.full_name,
                branch,
                response.text,
            )
            return None
        payload = response.json()
        tree = payload.get("tree", [])
        paths = [entry.get("path") for entry in tree if entry.get("path")]
        self._tree_cache[cache_key] = paths
        return paths

    @staticmethod
    def _paths_match_required_files(paths: Sequence[str], patterns: Sequence[str]) -> bool:
        """Return True if all required file patterns match at least one path."""
        if not patterns:
            return True
        lowered_paths = [path.lower() for path in paths]
        for pattern in patterns:
            target = pattern.lower()
            if not any(fnmatch(path, target) for path in lowered_paths):
                return False
        return True

    def _enrich_topics(self, repositories: List[RepositoryRecord]) -> List[RepositoryRecord]:
        """Fetch topics for repositories missing that information."""
        enriched: List[RepositoryRecord] = []
        for record in repositories:
            if record.topics:
                enriched.append(record)
                continue

            endpoint = f"{self.base_url}/{self.TOPICS_ENDPOINT.format(full_name=record.full_name)}"
            response = self.session.get(
                endpoint,
                timeout=self.request_timeout,
                headers={"Accept": "application/vnd.github.mercy-preview+json"},
            )
            if response.status_code == 200:
                payload = response.json()
                topics = payload.get("names", [])
                record.topics = [topic for topic in topics if topic]
            else:
                logger.debug(
                    "Skipping topics enrichment for %s (status=%s)",
                    record.full_name,
                    response.status_code,
                )
            enriched.append(record)
        return enriched


def load_model_specs(path: str | Path) -> List[ModelSpec]:
    """Load model specifications from a YAML or JSON document."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    try:
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(raw_text) or {}
        elif suffix == ".json":
            payload = json.loads(raw_text or "[]")
        else:
            payload = yaml.safe_load(raw_text) or {}
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse model spec file {path}: {exc}") from exc

    models_payload = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(models_payload, list):
        raise ValueError(f"Expected 'models' list in {path}.")

    return [ModelSpec(**model) for model in models_payload]


def load_prompts(path: str | Path) -> List[PromptVariant]:
    """Load prompt variants from a YAML or JSON document."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    try:
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(raw_text) or {}
        elif suffix == ".json":
            payload = json.loads(raw_text or "[]")
        else:
            payload = yaml.safe_load(raw_text) or {}
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse prompt spec file {path}: {exc}") from exc

    prompts_payload = payload.get("prompts") if isinstance(payload, dict) else payload
    if not prompts_payload:
        return []
    if not isinstance(prompts_payload, list):
        raise ValueError(f"Expected 'prompts' list in {path}.")
    return [PromptVariant(**prompt) for prompt in prompts_payload]


def default_dataset_path(name: str) -> Path:
    """Return the default storage path for a dataset name."""
    return Path("evaluation_reports") / "datasets" / f"{name}.json"
