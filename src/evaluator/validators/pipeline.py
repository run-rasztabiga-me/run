from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from ...common.command_runner import CommandRunner
from ...common.models import DockerBuildMetrics
from ..core.models import ValidationIssue
from ...generator.core.config import GeneratorConfig
from ...generator.core.workspace import RepositoryWorkspace
from ...generator.core.workspace_models import RunContext


@dataclass(slots=True)
class ValidationContext:
    """Static runtime context that is shared across validation steps."""

    workspace: RepositoryWorkspace
    run_context: RunContext
    config: GeneratorConfig
    command_runner: CommandRunner
    logger: logging.Logger


@dataclass(slots=True)
class ValidationState:
    """Mutable state that flows through the validation pipeline."""

    repo_name: str
    dockerfiles: Sequence[str] = field(default_factory=tuple)
    manifests: Sequence[str] = field(default_factory=tuple)
    docker_images: Sequence["DockerImageInfo"] = field(default_factory=tuple)
    test_endpoint: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    build_metrics: List[DockerBuildMetrics] = field(default_factory=list)
    runtime_success: Optional[bool] = None
    step_metadata: Dict[str, Dict[str, object]] = field(default_factory=dict)
    applied_resources: List["AppliedResource"] = field(default_factory=list)


@dataclass(slots=True)
class ValidationStepResult:
    """Outcome of executing a single validation step."""

    issues: List[ValidationIssue] = field(default_factory=list)
    build_metrics: List[DockerBuildMetrics] = field(default_factory=list)
    runtime_success: Optional[bool] = None
    continue_pipeline: bool = True
    metadata: Optional[Dict[str, object]] = None


class ValidationStep(Protocol):
    """Protocol implemented by all validation steps."""

    name: str

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        ...


@dataclass(slots=True)
class ValidationPipelineResult:
    """Aggregate result returned by the validation pipeline."""

    issues: List[ValidationIssue]
    build_metrics: List[DockerBuildMetrics]
    runtime_success: Optional[bool]
    step_metadata: Dict[str, Dict[str, object]]
    step_issues: Dict[str, List[ValidationIssue]]
    step_build_metrics: Dict[str, List[DockerBuildMetrics]]


class ValidationPipeline:
    """Coordinate ordered execution of validation steps."""

    def __init__(self, steps: Sequence[ValidationStep]) -> None:
        self.steps = list(steps)

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationPipelineResult:
        step_issues: Dict[str, List[ValidationIssue]] = {}
        step_build_metrics: Dict[str, List[DockerBuildMetrics]] = {}

        for step in self.steps:
            context.logger.debug("Running validation step: %s", step.name)
            result = step.run(state, context)
            state.issues.extend(result.issues)
            state.build_metrics.extend(result.build_metrics)
            step_issues[step.name] = list(result.issues)
            if result.build_metrics:
                step_build_metrics[step.name] = list(result.build_metrics)
            if result.runtime_success is not None:
                state.runtime_success = result.runtime_success
            if result.metadata is not None:
                state.step_metadata[step.name] = result.metadata

            if not result.continue_pipeline:
                context.logger.debug("Stopping pipeline after step %s", step.name)
                break

        return ValidationPipelineResult(
            issues=list(state.issues),
            build_metrics=list(state.build_metrics),
            runtime_success=state.runtime_success,
            step_metadata=state.step_metadata,
            step_issues=step_issues,
            step_build_metrics=step_build_metrics,
        )


# Circular import guard - typing only
from ...common.models import DockerImageInfo  # noqa: E402  # isort:skip
from ...runtime import AppliedResource  # noqa: E402  # isort:skip
