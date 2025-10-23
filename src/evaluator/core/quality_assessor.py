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
        quality_metrics = QualityMetrics()
        build_metrics: List[DockerBuildMetrics] = []
        build_failed = False
        runtime_issues: Optional[List[ValidationIssue]] = None
        runtime_success: Optional[bool] = None

        if generation_result.docker_images:
            dockerfile_issues = self.validator.validate_dockerfiles(generation_result.dockerfiles)
            quality_metrics.validation_issues.extend(dockerfile_issues)

            build_issues, build_metrics = self.validator.build_docker_images(
                generation_result.docker_images,
                generation_result.repo_name,
            )
            quality_metrics.validation_issues.extend(build_issues)

            all_docker_issues = dockerfile_issues + build_issues
            quality_metrics.dockerfile_score = self._calculate_dockerfile_score(all_docker_issues)

            build_failed = any(issue.severity == ValidationSeverity.ERROR for issue in build_issues)

        if generation_result.k8s_manifests and not build_failed:
            k8s_issues = self.validator.validate_k8s_manifests(generation_result.k8s_manifests)
            quality_metrics.validation_issues.extend(k8s_issues)

            apply_issues = self.validator.apply_k8s_manifests(
                generation_result.k8s_manifests,
                generation_result.repo_name,
                generation_result.docker_images,
            )
            quality_metrics.validation_issues.extend(apply_issues)

            all_k8s_issues = k8s_issues + apply_issues
            quality_metrics.k8s_manifests_score = self._calculate_k8s_score(all_k8s_issues)

            if test_endpoint:
                runtime_result = self.validator.validate_runtime_availability(
                    generation_result.k8s_manifests,
                    generation_result.run_context.k8s_namespace,
                    test_endpoint,
                )
                runtime_issues = runtime_result
                quality_metrics.validation_issues.extend(runtime_result)
                if runtime_result:
                    runtime_success = not any(i.severity == ValidationSeverity.ERROR for i in runtime_result)
                else:
                    runtime_success = True

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
