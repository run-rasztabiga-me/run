from __future__ import annotations

import logging
from typing import Optional, Sequence

from ..core.models import GenerationResult, ValidationIssue
from .pipeline import ValidationContext, ValidationPipeline, ValidationPipelineResult, ValidationState, ValidationStep
from .steps import (
    DockerBuildStep,
    DockerLLMJudgeStep,
    DockerfileLinterValidationStep,
    DockerfileSyntaxValidationStep,
    KubernetesApplyStep,
    KubernetesLLMJudgeStep,
    KubernetesLinterValidationStep,
    KubernetesSyntaxValidationStep,
    RuntimeValidationStep,
)
from ...common.command_runner import CommandRunner
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

    def run_steps(
        self,
        generation_result: GenerationResult,
        test_endpoints: Optional[List[str]] = None,
    ) -> ValidationPipelineResult:
        state = ValidationState(
            repo_name=self.run_context.repo_name,
            dockerfiles=tuple(generation_result.dockerfiles),
            manifests=tuple(generation_result.k8s_manifests),
            docker_images=tuple(generation_result.docker_images),
            test_endpoints=test_endpoints,
        )

        steps: list[ValidationStep] = []

        if generation_result.dockerfiles:
            steps.extend(
                [
                    DockerfileSyntaxValidationStep(),
                    DockerfileLinterValidationStep(),
                ]
            )
            if self.config.enable_llm_judge:
                steps.append(DockerLLMJudgeStep())

        if generation_result.docker_images:
            steps.append(DockerBuildStep())

        if generation_result.k8s_manifests:
            steps.extend(
                [
                    KubernetesSyntaxValidationStep(),
                    KubernetesLinterValidationStep(),
                ]
            )
            if self.config.enable_llm_judge:
                steps.append(KubernetesLLMJudgeStep())
            steps.append(KubernetesApplyStep())
            if test_endpoints:
                steps.append(RuntimeValidationStep())

        if not steps:
            self.logger.debug("No validation steps scheduled for repository %s", self.run_context.repo_name)

        return self._run_steps(steps=steps, state=state)
