import logging
import time
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import (
    EvaluationReport, EvaluationStatus, GenerationResult, ExecutionMetrics, QualityMetrics
)
from ..validators.config_validator import ConfigurationValidator
from ..reports.reporter import EvaluationReporter
from ...generator.core.generator import ConfigurationGenerator
from ...generator.core.config import GeneratorConfig
from .integration import GeneratorIntegration
from ...utils.repository_utils import extract_repo_name


class ConfigurationEvaluator:
    """Main evaluator class for assessing configuration generation quality."""

    def __init__(
        self,
        generator_config: Optional[GeneratorConfig] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.generator_config = generator_config or GeneratorConfig()

        # Initialize components
        self.generator = ConfigurationGenerator(self.generator_config)
        self.validator = ConfigurationValidator(self.generator.get_repository_manager())
        self.reporter = EvaluationReporter()
        self.generator_integration = GeneratorIntegration(self.generator)

    def evaluate_repository(self, repo_url: str) -> EvaluationReport:
        """
        Evaluate configuration generation for a single repository.

        Args:
            repo_url: URL of the repository to evaluate

        Returns:
            Complete evaluation report
        """
        evaluation_id = str(uuid.uuid4())
        repo_name = extract_repo_name(repo_url)

        report = EvaluationReport(
            repo_url=repo_url,
            repo_name=repo_name,
            evaluation_id=evaluation_id,
            status=EvaluationStatus.IN_PROGRESS,
        )

        self.logger.info(f"Starting evaluation for {repo_url} (ID: {evaluation_id})")

        try:
            # Step 1: Generate configurations with metrics collection
            report.add_note("Starting configuration generation")
            generation_result, execution_metrics = self._generate_with_metrics(repo_url)

            report.generation_result = generation_result
            report.execution_metrics = execution_metrics

            if not generation_result.success:
                report.mark_failed(f"Generation failed: {generation_result.error_message}")
                return report

            # Step 2: Validate and assess quality
            report.add_note("Starting quality assessment")
            quality_metrics = self._assess_quality(generation_result)
            report.quality_metrics = quality_metrics

            # Step 3: Generate detailed report
            report.add_note("Generating evaluation report")
            report.mark_completed()

            self.logger.info(f"Evaluation completed for {repo_url}")

        except Exception as e:
            self.logger.error(f"Evaluation failed for {repo_url}: {str(e)}")
            report.mark_failed(str(e))

        return report

    def evaluate_batch(self, repo_urls: List[str]) -> List[EvaluationReport]:
        """
        Evaluate multiple repositories in batch.

        Args:
            repo_urls: List of repository URLs to evaluate

        Returns:
            List of evaluation reports
        """
        self.logger.info(f"Starting batch evaluation of {len(repo_urls)} repositories")
        reports = []

        for i, repo_url in enumerate(repo_urls, 1):
            self.logger.info(f"Processing repository {i}/{len(repo_urls)}: {repo_url}")

            try:
                report = self.evaluate_repository(repo_url)
                reports.append(report)
            except Exception as e:
                self.logger.error(f"Failed to evaluate {repo_url}: {str(e)}")
                # Create minimal failed report
                failed_report = EvaluationReport(
                    repo_url=repo_url,
                    repo_name=extract_repo_name(repo_url),
                    evaluation_id=str(uuid.uuid4()),
                    status=EvaluationStatus.FAILED,
                        )
                failed_report.mark_failed(str(e))
                reports.append(failed_report)

        self.logger.info(f"Batch evaluation completed. {len(reports)} reports generated")
        return reports

    def _generate_with_metrics(self, repo_url: str) -> tuple[GenerationResult, ExecutionMetrics]:
        """Generate configurations while collecting execution metrics."""
        try:
            # Use the generator integration to handle the generation process
            generation_result, execution_metrics = self.generator_integration.generate_with_monitoring(repo_url)
            return generation_result, execution_metrics

        except Exception as e:
            execution_metrics = ExecutionMetrics()
            execution_metrics.total_time = 0
            execution_metrics.error_count += 1

            generation_result = GenerationResult(
                repo_url=repo_url,
                repo_name=extract_repo_name(repo_url),
                success=False,
                error_message=str(e),
                generation_time=0
            )

            return generation_result, execution_metrics

    def _assess_quality(self, generation_result: GenerationResult) -> QualityMetrics:
        """Assess the quality of generated configurations."""
        quality_metrics = QualityMetrics()

        # TODO opracować algorytm na liczenie score'u

        try:
            # Validate Dockerfiles if they exist
            if generation_result.dockerfiles:
                dockerfile_issues = self.validator.validate_dockerfiles(generation_result.dockerfiles)
                quality_metrics.validation_issues.extend(dockerfile_issues)
                quality_metrics.dockerfile_score = self._calculate_dockerfile_score(dockerfile_issues)

            # Validate Kubernetes manifests if they exist
            if generation_result.k8s_manifests:
                k8s_issues = self.validator.validate_k8s_manifests(generation_result.k8s_manifests)
                quality_metrics.validation_issues.extend(k8s_issues)
                quality_metrics.k8s_manifests_score = self._calculate_k8s_score(k8s_issues)

            # Calculate overall metrics
            quality_metrics.overall_score = self._calculate_overall_score(quality_metrics)

        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")

        return quality_metrics

    def _calculate_dockerfile_score(self, issues: List) -> float:
        """Calculate Dockerfile quality score based on validation issues."""
        # Placeholder scoring logic
        base_score = 100.0
        penalty_per_error = 10.0
        penalty_per_warning = 5.0

        error_count = len([i for i in issues if i.severity.value == "error"])
        warning_count = len([i for i in issues if i.severity.value == "warning"])

        score = base_score - (error_count * penalty_per_error) - (warning_count * penalty_per_warning)
        return max(0.0, score)

    def _calculate_k8s_score(self, issues: List) -> float:
        """Calculate Kubernetes manifests quality score based on validation issues."""
        # Placeholder scoring logic
        base_score = 100.0
        penalty_per_error = 8.0
        penalty_per_warning = 4.0

        error_count = len([i for i in issues if i.severity.value == "error"])
        warning_count = len([i for i in issues if i.severity.value == "warning"])

        score = base_score - (error_count * penalty_per_error) - (warning_count * penalty_per_warning)
        return max(0.0, score)

    def _calculate_overall_score(self, quality_metrics: QualityMetrics) -> float:
        """Calculate overall quality score as the average of Dockerfile and Kubernetes manifest scores."""
        scores = []
        if quality_metrics.dockerfile_score is not None:
            scores.append(quality_metrics.dockerfile_score)
        if quality_metrics.k8s_manifests_score is not None:
            scores.append(quality_metrics.k8s_manifests_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)


    def save_report(self, report: EvaluationReport, output_dir: str = "./evaluation_reports") -> str:
        """Save evaluation report to file."""
        return self.reporter.save_report(report, output_dir)

    def export_batch_results(self, reports: List[EvaluationReport], output_dir: str = "./evaluation_reports") -> Dict[str, str]:
        """Export batch evaluation results in multiple formats."""
        return self.reporter.export_batch_results(reports, output_dir)

