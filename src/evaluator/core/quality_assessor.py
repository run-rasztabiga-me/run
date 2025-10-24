from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import (
    DockerBuildMetrics,
    GenerationResult,
    QualityMetrics,
    ValidationIssue,
    ValidationSeverity,
)
from ..validators.config_validator import ConfigurationValidator


@dataclass
class QualityAssessmentResult:
    metrics: QualityMetrics
    build_metrics: List[DockerBuildMetrics]
    build_failed: bool
    runtime_issues: Optional[List[ValidationIssue]]
    runtime_success: Optional[bool]


class QualityAssessor:
    """Encapsulates Docker/Kubernetes validation and scoring logic."""
    # TODO opracować algorytm na liczenie score'u

    def __init__(self, validator: ConfigurationValidator) -> None:
        self.validator = validator

    def assess(
        self,
        generation_result: GenerationResult,
        test_endpoint: Optional[str] = None,
    ) -> QualityAssessmentResult:
        pipeline_result = self.validator.run_steps(
            generation_result=generation_result,
            test_endpoint=test_endpoint,
        )

        quality_metrics = QualityMetrics()
        quality_metrics.validation_issues.extend(pipeline_result.issues)

        build_metrics: List[DockerBuildMetrics] = list(pipeline_result.build_metrics)
        step_issues = pipeline_result.step_issues

        docker_issue_steps = ("docker_syntax", "docker_linters", "docker_build")
        docker_issues: List[ValidationIssue] = []
        for step in docker_issue_steps:
            docker_issues.extend(step_issues.get(step, []))

        if generation_result.docker_images and any(step in step_issues for step in docker_issue_steps):
            quality_metrics.dockerfile_score = self._calculate_dockerfile_score(docker_issues)

        build_failed = any(issue.severity == ValidationSeverity.ERROR for issue in step_issues.get("docker_build", []))

        k8s_issue_steps = ("k8s_syntax", "k8s_linters", "kubernetes_apply")
        k8s_issues: List[ValidationIssue] = []
        for step in k8s_issue_steps:
            k8s_issues.extend(step_issues.get(step, []))

        ran_k8s_pipeline = any(step in step_issues for step in k8s_issue_steps)
        if generation_result.k8s_manifests and not build_failed and ran_k8s_pipeline:
            quality_metrics.k8s_manifests_score = self._calculate_k8s_score(k8s_issues)

        runtime_issues = step_issues.get("runtime")
        runtime_success = pipeline_result.runtime_success

        quality_metrics.overall_score = self._calculate_overall_score(quality_metrics)
        return QualityAssessmentResult(
            metrics=quality_metrics,
            build_metrics=build_metrics,
            build_failed=build_failed,
            runtime_issues=runtime_issues,
            runtime_success=runtime_success,
        )

    def _calculate_dockerfile_score(self, issues: List) -> float:
        base_score = 100.0
        penalty_per_error = 10.0
        penalty_per_warning = 5.0

        error_count = len([i for i in issues if i.severity.value == "error"])
        warning_count = len([i for i in issues if i.severity.value == "warning"])

        score = base_score - (error_count * penalty_per_error) - (warning_count * penalty_per_warning)
        return max(0.0, score)

    def _calculate_k8s_score(self, issues: List) -> float:
        base_score = 100.0
        penalty_per_error = 8.0
        penalty_per_warning = 4.0

        error_count = len([i for i in issues if i.severity.value == "error"])
        warning_count = len([i for i in issues if i.severity.value == "warning"])

        score = base_score - (error_count * penalty_per_error) - (warning_count * penalty_per_warning)
        return max(0.0, score)

    def _calculate_overall_score(self, quality_metrics: QualityMetrics) -> float:
        scores = []
        if quality_metrics.dockerfile_score is not None:
            scores.append(quality_metrics.dockerfile_score)
        if quality_metrics.k8s_manifests_score is not None:
            scores.append(quality_metrics.k8s_manifests_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)
