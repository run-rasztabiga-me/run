from __future__ import annotations

from typing import List

from src.common.models import DockerBuildMetrics
from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import ValidationIssue, ValidationSeverity
from ..utils import runtime_issues_to_validation
from src.runtime import DockerImageBuilder


class DockerBuildStep:
    """Build Docker images and push them to the configured registry."""

    name = "docker_build"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        issues: List[ValidationIssue] = []
        metrics: List[DockerBuildMetrics] = []
        builder = DockerImageBuilder(
            command_runner=context.command_runner,
            logger=context.logger,
        )

        for image_info in state.docker_images:
            context.logger.info(
                "Building Docker image: %s from %s",
                image_info.image_tag,
                image_info.dockerfile_path,
            )

            full_image_name = context.config.get_full_image_name(
                state.repo_name,
                image_info.image_tag,
                version=context.run_context.run_id,
            )
            context.logger.info("Full image name with run-scoped tag: %s", full_image_name)

            result = builder.build_and_push(
                workspace=context.workspace,
                image=image_info,
                image_name=full_image_name,
            )
            issues.extend(
                runtime_issues_to_validation(
                    result.issues,
                    default_subject=image_info.dockerfile_path,
                )
            )
            if result.metrics:
                metrics.append(result.metrics)

        continue_pipeline = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationStepResult(
            issues=issues,
            build_metrics=metrics,
            continue_pipeline=continue_pipeline,
        )
