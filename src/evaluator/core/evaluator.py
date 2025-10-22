import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .models import (
    EvaluationReport, EvaluationStatus, GenerationResult, ExecutionMetrics, QualityMetrics, ValidationSeverity
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
        self.validator = ConfigurationValidator(
            self.generator.get_repository_manager(),
            self.generator_config
        )
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
        report.build_success = None
        report.runtime_success = None

        self.logger.info(f"Starting evaluation for {repo_url} (ID: {evaluation_id})")

        try:
            # Step 1: Generate configurations with metrics collection
            report.add_note("Starting configuration generation")
            generation_result, execution_metrics = self._generate_with_metrics(repo_url)

            report.generation_result = generation_result
            report.execution_metrics = execution_metrics

            if not generation_result.success:
                report.build_success = False
                report.mark_failed(f"Generation failed: {generation_result.error_message}")
                return report

            # Add note about generated files
            report.add_note(f"Generated {len(generation_result.docker_images)} Dockerfiles and {len(generation_result.k8s_manifests)} K8s manifests")

            # Step 2: Validate and assess quality
            report.add_note("Starting quality assessment")
            quality_metrics, build_metrics = self._assess_quality(generation_result)
            report.quality_metrics = quality_metrics

            # Add build metrics to execution metrics
            if build_metrics:
                execution_metrics.docker_build_metrics.extend(build_metrics)
                # Add note about build metrics
                total_size = sum(m.image_size_mb for m in build_metrics)
                avg_time = sum(m.build_time for m in build_metrics) / len(build_metrics)
                report.add_note(f"Built {len(build_metrics)} Docker images (total: {total_size:.1f} MB, avg time: {avg_time:.1f}s)")

            # Add note about validation results
            if quality_metrics.validation_issues:
                error_count = len([i for i in quality_metrics.validation_issues if i.severity.value == "error"])
                warning_count = len([i for i in quality_metrics.validation_issues if i.severity.value == "warning"])
                if generation_result.docker_images:
                    report.add_note(f"Dockerfile validation: {error_count} errors, {warning_count} warnings")
                if generation_result.k8s_manifests:
                    k8s_issues = [i for i in quality_metrics.validation_issues if 'k8s' in i.file_path.lower() or 'kubectl' in i.rule_id.lower() or 'kube' in i.rule_id.lower()]
                    k8s_errors = len([i for i in k8s_issues if i.severity.value == "error"])
                    k8s_warnings = len([i for i in k8s_issues if i.severity.value == "warning"])
                    report.add_note(f"K8s validation: {k8s_errors} errors, {k8s_warnings} warnings")

            # Check if there were Docker build errors
            has_build_errors = False
            if generation_result.docker_images:
                docker_build_issues = [
                    i for i in quality_metrics.validation_issues
                    if i.rule_id in ['DOCKER_BUILDX_FAILED', 'DOCKER_PUSH_FAILED', 'DOCKER_BUILD_FILE_NOT_FOUND', 'DOCKER_BUILD_CONTEXT_NOT_FOUND']
                    and i.severity == ValidationSeverity.ERROR
                ]
                has_build_errors = len(docker_build_issues) > 0
                report.build_success = not has_build_errors
            else:
                report.build_success = None

            # Add note about deployment (only if no build errors)
            if generation_result.k8s_manifests and not has_build_errors:
                namespace = generation_result.repo_name.lower().replace('_', '-')
                report.add_note(f"Deployed to namespace: {namespace}")

            # Step 3: Runtime validation - test endpoint health
            # Skip if there were Docker build errors
            if generation_result.k8s_manifests and generation_result.test_endpoint and not has_build_errors:
                report.add_note(f"Testing endpoint: {generation_result.test_endpoint}")
                namespace = generation_result.repo_name.lower().replace('_', '-')
                runtime_issues = self.validator.validate_runtime_availability(
                    generation_result.k8s_manifests,
                    namespace,
                    generation_result.test_endpoint
                )
                quality_metrics.validation_issues.extend(runtime_issues)

                # Log runtime validation results
                if runtime_issues:
                    error_count = len([i for i in runtime_issues if i.severity.value == "error"])
                    warning_count = len([i for i in runtime_issues if i.severity.value == "warning"])
                    if error_count > 0:
                        report.add_note(f"Runtime validation: {error_count} errors, {warning_count} warnings")
                        report.runtime_success = False
                    elif warning_count > 0:
                        report.add_note(f"Runtime validation: {warning_count} warnings (no errors)")
                        report.runtime_success = True
                else:
                    report.add_note(f"Runtime validation passed: endpoint is healthy")
                    report.runtime_success = True
            elif has_build_errors:
                report.add_note("Runtime validation skipped due to Docker build errors")
                report.runtime_success = False

            # Step 4: Generate detailed report
            report.add_note("Generating evaluation report")
            report.mark_completed()

            self.logger.info(f"Evaluation completed for {repo_url}")

        except Exception as e:
            self.logger.error(f"Evaluation failed for {repo_url}: {str(e)}")
            report.runtime_success = False
            report.build_success = False if report.build_success is None else report.build_success
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

    def _assess_quality(self, generation_result: GenerationResult) -> Tuple[QualityMetrics, List]:
        """Assess the quality of generated configurations."""
        quality_metrics = QualityMetrics()
        build_metrics = []

        # TODO opracować algorytm na liczenie score'u

        try:
            # Validate and build Docker images if they exist
            if generation_result.docker_images:
                # First validate Dockerfiles syntax and best practices
                dockerfile_issues = self.validator.validate_dockerfiles(generation_result.dockerfiles)
                quality_metrics.validation_issues.extend(dockerfile_issues)

                # Then try to build the images (if syntax validation passed)
                build_issues, build_metrics = self.validator.build_docker_images(
                    generation_result.docker_images,
                    generation_result.repo_name
                )
                quality_metrics.validation_issues.extend(build_issues)

                # Calculate score based on all issues
                all_dockerfile_issues = dockerfile_issues + build_issues
                quality_metrics.dockerfile_score = self._calculate_dockerfile_score(all_dockerfile_issues)

                # Check if any critical build errors occurred
                has_build_errors = any(
                    issue.severity == ValidationSeverity.ERROR
                    for issue in build_issues
                )

                if has_build_errors:
                    self.logger.error("Docker image build failed. Skipping Kubernetes deployment.")
                    # Return early - don't proceed to K8s validation if images didn't build
                    quality_metrics.overall_score = self._calculate_overall_score(quality_metrics)
                    return quality_metrics, build_metrics

            # Validate Kubernetes manifests if they exist
            if generation_result.k8s_manifests:
                # Syntax validation with kubectl dry-run and static analysis
                k8s_issues = self.validator.validate_k8s_manifests(generation_result.k8s_manifests)
                quality_metrics.validation_issues.extend(k8s_issues)

                # Apply to cluster and verify deployment
                apply_issues = self.validator.apply_k8s_manifests(
                    generation_result.k8s_manifests,
                    generation_result.repo_name,
                    generation_result.docker_images
                )
                quality_metrics.validation_issues.extend(apply_issues)

                # Calculate score based on all issues
                all_k8s_issues = k8s_issues + apply_issues
                quality_metrics.k8s_manifests_score = self._calculate_k8s_score(all_k8s_issues)

            # Calculate overall metrics
            quality_metrics.overall_score = self._calculate_overall_score(quality_metrics)

        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")

        return quality_metrics, build_metrics

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
