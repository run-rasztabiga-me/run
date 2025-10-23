from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from ..core.models import DockerBuildMetrics, ValidationIssue
from .pipeline import ValidationContext, ValidationPipeline, ValidationPipelineResult, ValidationState, ValidationStep
from .steps import (
    DockerBuildStep,
    KubernetesApplyStep,
    LinterValidationStep,
    RuntimeValidationStep,
    SyntaxValidationStep,
)
from ...common.command_runner import CommandRunner
from ...common.models import DockerImageInfo
from ...generator.core.config import GeneratorConfig
from ...generator.core.workspace import RepositoryWorkspace
from ...generator.core.workspace_models import RunContext


class ConfigurationValidator:
    """Coordinates validation steps for generated Dockerfiles and manifests."""

    def __init__(self, workspace: RepositoryWorkspace, run_context: RunContext, config: GeneratorConfig | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.workspace = workspace
        self.run_context = run_context
        self.config = config or GeneratorConfig()
        self.command_runner = CommandRunner(self.logger)

    def _create_context(self) -> ValidationContext:
        return ValidationContext(
            workspace=self.workspace,
            run_context=self.run_context,
            config=self.config,
            command_runner=self.command_runner,
            logger=self.logger,
        )

    def _run_steps(self, steps: Sequence[ValidationStep], state: ValidationState) -> ValidationPipelineResult:
        pipeline = ValidationPipeline(steps)
        return pipeline.run(state, self._create_context())

    def validate_dockerfiles(self, dockerfile_paths: List[str]) -> List[ValidationIssue]:
        if not dockerfile_paths:
            return []

        state = ValidationState(
            repo_name=self.run_context.repo_name,
            dockerfiles=tuple(dockerfile_paths),
        )
        result = self._run_steps(
            steps=[SyntaxValidationStep(), LinterValidationStep()],
            state=state,
        )
        return result.issues

    def validate_k8s_manifests(self, manifest_paths: List[str]) -> List[ValidationIssue]:
        if not manifest_paths:
            return []

        state = ValidationState(
            repo_name=self.run_context.repo_name,
            manifests=tuple(manifest_paths),
        )
        result = self._run_steps(
            steps=[SyntaxValidationStep(), LinterValidationStep()],
            state=state,
        )
        return result.issues

    def build_docker_images(self, docker_images: List[DockerImageInfo], repo_name: str) -> Tuple[List[ValidationIssue], List[DockerBuildMetrics]]:
        if not docker_images:
            return [], []

        state = ValidationState(
            repo_name=repo_name,
            docker_images=tuple(docker_images),
        )
        result = self._run_steps(
            steps=[DockerBuildStep()],
            state=state,
        )
        return result.issues, result.build_metrics

    def apply_k8s_manifests(
        self,
        manifest_paths: List[str],
        repo_name: str,
        docker_images: Optional[List[DockerImageInfo]] = None,
    ) -> List[ValidationIssue]:
        docker_images = docker_images or []
        if not manifest_paths:
            return []

        state = ValidationState(
            repo_name=repo_name,
            manifests=tuple(manifest_paths),
            docker_images=tuple(docker_images),
        )
        result = self._run_steps(
            steps=[KubernetesApplyStep()],
            state=state,
        )
        return result.issues

    def validate_runtime_availability(
        self,
        manifest_paths: List[str],
        namespace: str,
        test_endpoint: str,
    ) -> List[ValidationIssue]:
        if not manifest_paths:
            return []

        state = ValidationState(
            repo_name=self.run_context.repo_name,
            manifests=tuple(manifest_paths),
            test_endpoint=test_endpoint,
        )
        result = self._run_steps(
            steps=[RuntimeValidationStep()],
            state=state,
        )
        return result.issues
