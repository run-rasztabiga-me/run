"""Configuration models and loader for experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelSpec(BaseModel):
    """Definition of a single model configuration used during experiments."""

    provider: str = Field(description="Model provider identifier (e.g., 'openai', 'anthropic').")
    name: str = Field(description="Model name as accepted by the provider SDK.")
    temperature: Optional[float] = Field(
        default=None,
        description="Optional temperature override for the generator."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional deterministic seed override."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional GeneratorConfig keyword overrides specific to this model."
    )
    label: Optional[str] = Field(
        default=None,
        description="Human friendly label to use in reports. Falls back to provider+name."
    )

    @field_validator("provider", "name")
    @classmethod
    def _strip_whitespace(cls, value: str) -> str:
        """Normalize whitespace for string fields."""
        return value.strip()

    @property
    def identifier(self) -> str:
        """Return a filesystem-friendly identifier for directory creation."""
        raw = self.label or f"{self.provider}_{self.name}"
        return _sanitize_token(raw)


class ExperimentDefinition(BaseModel):
    """Single experiment definition covering multiple repos and models."""

    name: str = Field(description="Unique experiment name used for directory structure.")
    description: Optional[str] = Field(default=None, description="Optional notes for the experiment.")
    repos: List[str] = Field(description="List of repository URLs to evaluate.")
    models: List[ModelSpec] = Field(description="Model configurations to evaluate against each repo.")
    repetitions: int = Field(default=1, description="Number of repeated runs per repo/model pair.")
    generator_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="GeneratorConfig keyword overrides applied to every run in this experiment."
    )
    run_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata/flags stored with each run (reserved for future use)."
    )
    prompts: List["PromptVariant"] = Field(
        default_factory=list,
        description="Optional prompt variants to evaluate within this experiment."
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Experiment name cannot be empty.")
        return value.strip()

    @field_validator("repos")
    @classmethod
    def _validate_repos(cls, repos: List[str]) -> List[str]:
        if not repos:
            raise ValueError("Experiment must include at least one repository URL.")
        return [repo.strip() for repo in repos]

    @field_validator("models")
    @classmethod
    def _validate_models(cls, models: List[ModelSpec]) -> List[ModelSpec]:
        if not models:
            raise ValueError("Experiment must include at least one model definition.")
        return models

    @field_validator("repetitions")
    @classmethod
    def _validate_repetitions(cls, repetitions: int) -> int:
        if repetitions < 1:
            raise ValueError("Repetitions must be >= 1.")
        return repetitions

    @property
    def safe_name(self) -> str:
        """Return sanitized experiment name for directory usage."""
        return _sanitize_token(self.name)


class ExperimentSuite(BaseModel):
    """Container model holding multiple experiments."""

    experiments: List[ExperimentDefinition] = Field(default_factory=list)
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional base output directory override for all experiments."
    )

    @model_validator(mode="after")
    def _ensure_unique_experiment_names(self) -> "ExperimentSuite":
        seen = set()
        for experiment in self.experiments:
            key = experiment.safe_name
            if key in seen:
                raise ValueError(f"Duplicate experiment name detected: {experiment.name}")
            seen.add(key)
        return self


def load_experiment_suite(path: str | Path) -> ExperimentSuite:
    """Load experiment definitions from a YAML or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported experiment config format: {suffix}")

    if data is None:
        raise ValueError(f"Experiment config file {path} is empty.")

    return ExperimentSuite.model_validate(data)


def _sanitize_token(value: str) -> str:
    """Create a filesystem-friendly token from arbitrary text."""
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.strip())
    sanitized = "-".join(filter(None, sanitized.split("-")))
    return sanitized or "value"


class PromptVariant(BaseModel):
    """Defines a system prompt variant for experimentation."""

    id: str = Field(description="Unique identifier for this prompt variant.")
    description: Optional[str] = Field(default=None, description="Optional description for the prompt intent.")
    system_prompt: Optional[str] = Field(default=None, description="Inline system prompt override.")
    system_prompt_path: Optional[str] = Field(default=None, description="Path to a file containing the system prompt.")

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt variant id cannot be empty.")
        return value.strip()

    @model_validator(mode="after")
    def _ensure_prompt_source(self) -> "PromptVariant":
        if not any([self.system_prompt, self.system_prompt_path]):
            raise ValueError(
                "PromptVariant requires either 'system_prompt' or 'system_prompt_path'."
            )
        return self

    def resolve(self, base_path: Path) -> tuple[str, Optional[str]]:
        """Return tuple of (system_prompt_text, source_path)."""
        if self.system_prompt:
            return self.system_prompt, None

        prompt_path = Path(self.system_prompt_path)
        if not prompt_path.is_absolute():
            prompt_path = (base_path / prompt_path).resolve()
        return prompt_path.read_text(encoding="utf-8"), str(prompt_path)

    @property
    def identifier(self) -> str:
        return self.id


__all__ = ["ModelSpec", "PromptVariant", "ExperimentDefinition", "ExperimentSuite", "load_experiment_suite"]

ExperimentDefinition.model_rebuild()
ExperimentSuite.model_rebuild()
