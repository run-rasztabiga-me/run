from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ...common.models import DockerBuildMetrics
from .models import (
    GenerationResult,
    QualityMetrics,
    ValidationIssue,
    ValidationSeverity,
)
from ..validators.config_validator import ConfigurationValidator
from .scoring_model import IssueAggregationModel


@dataclass
class QualityAssessmentResult:
    metrics: QualityMetrics
    build_metrics: List[DockerBuildMetrics]
    build_failed: bool
    runtime_issues: Optional[List[ValidationIssue]]
    runtime_success: Optional[bool]


class QualityAssessor:
    """Encapsulates Docker/Kubernetes validation and scoring logic."""

    def __init__(self, validator: ConfigurationValidator) -> None:
        self.validator = validator
        self.scoring_model = IssueAggregationModel()

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
        step_metadata = pipeline_result.step_metadata
        runtime_success = pipeline_result.runtime_success

        # Use new scoring model to calculate comprehensive scores
        aggregated_score = self.scoring_model.calculate_scores(
            step_issues=step_issues,
            runtime_success=runtime_success,
        )

        # Populate quality metrics with aggregated scores
        quality_metrics.overall_score = aggregated_score.overall_score

        # Set component scores (backward compatibility)
        if aggregated_score.docker_component:
            quality_metrics.dockerfile_score = aggregated_score.docker_component.weighted_score

        if aggregated_score.k8s_component:
            quality_metrics.k8s_manifests_score = aggregated_score.k8s_component.weighted_score

        # Store detailed scoring breakdown in quality metrics
        quality_metrics.scoring_breakdown = aggregated_score.to_dict()
        llm_results = {
            step_name: metadata
            for step_name, metadata in step_metadata.items()
            if step_name in {"docker_llm_judge", "k8s_llm_judge"}
        }
        if llm_results:
            quality_metrics.llm_judge_results.update(llm_results)

        # Aggregate severity counts for H3 hypothesis verification
        all_issues = pipeline_result.issues
        quality_metrics.error_count = sum(1 for issue in all_issues if issue.severity == ValidationSeverity.ERROR)
        quality_metrics.warning_count = sum(1 for issue in all_issues if issue.severity == ValidationSeverity.WARNING)
        quality_metrics.info_count = sum(1 for issue in all_issues if issue.severity == ValidationSeverity.INFO)

        # Check syntax validation results
        quality_metrics.dockerfile_syntax_valid = self._check_dockerfile_syntax(step_issues)
        quality_metrics.k8s_syntax_valid = self._check_k8s_syntax(step_issues)

        build_failed = any(issue.severity == ValidationSeverity.ERROR for issue in step_issues.get("docker_build", []))
        runtime_issues = step_issues.get("runtime")

        return QualityAssessmentResult(
            metrics=quality_metrics,
            build_metrics=build_metrics,
            build_failed=build_failed,
            runtime_issues=runtime_issues,
            runtime_success=runtime_success,
        )

    def _check_dockerfile_syntax(self, step_issues: dict) -> Optional[bool]:
        """Check if Dockerfile syntax validation passed (no errors in syntax step)."""
        syntax_issues = step_issues.get("docker_syntax")
        if syntax_issues is None:
            # Syntax validation step never ran
            return None

        if not syntax_issues:
            # Step ran and produced no issues
            return True

        has_syntax_errors = any(issue.severity == ValidationSeverity.ERROR for issue in syntax_issues)
        return not has_syntax_errors

    def _check_k8s_syntax(self, step_issues: dict) -> Optional[bool]:
        """Check if Kubernetes manifest syntax validation passed (no errors in syntax step)."""
        syntax_issues = step_issues.get("k8s_syntax", [])
        if not syntax_issues:
            # No syntax step was run or no issues found
            return None if "k8s_syntax" not in step_issues else True
        # Check if there are any ERROR-level syntax issues
        has_syntax_errors = any(issue.severity == ValidationSeverity.ERROR for issue in syntax_issues)
        return not has_syntax_errors
