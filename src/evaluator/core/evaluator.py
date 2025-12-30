import logging
import uuid
from typing import List, Optional, Dict, Any

from .models import (
    EvaluationReport, EvaluationStatus, GenerationResult, ExecutionMetrics, QualityMetrics, ValidationSeverity,
    count_errors, count_warnings,
)
from ..reports.reporter import EvaluationReporter
from ...generator.core.generator import ConfigurationGenerator
from ...generator.core.config import GeneratorConfig
from .integration import GeneratorIntegration
from ...generator.core.workspace import RepositoryWorkspace
from ...utils.repository_utils import extract_repo_name
from .quality_assessor import QualityAssessor, QualityAssessmentResult
from ..validators.config_validator import ConfigurationValidator


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
        self.reporter = EvaluationReporter()
        self.generator_integration = GeneratorIntegration(self.generator)
        # Validator will be created per evaluation with workspace and run_context

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
            generation_result, execution_metrics, workspace = self._generate_with_metrics(repo_url)

            report.generation_result = generation_result
            report.execution_metrics = execution_metrics

            if not generation_result.success:
                report.build_success = False
                report.runtime_success = False
                report.mark_failed(f"Generation failed: {generation_result.error_message}")
                return report

            # Check that we have run_context from generation
            if not generation_result.run_context:
                report.build_success = False
                report.runtime_success = False
                report.mark_failed("Generation result missing run_context")
                return report

            # Add note about generated files
            report.add_note(f"Generated {len(generation_result.docker_images)} Dockerfiles and {len(generation_result.k8s_manifests)} K8s manifests")

            # Step 2: Validate and assess quality
            report.add_note("Starting quality assessment")
            assessment = self._assess_quality(generation_result, workspace)
            report.quality_metrics = assessment.metrics

            # Add build metrics to execution metrics
            if assessment.build_metrics:
                execution_metrics.docker_build_metrics.extend(assessment.build_metrics)
                # Add note about build metrics
                total_size = sum(m.image_size_mb for m in assessment.build_metrics)
                avg_time = sum(m.build_time for m in assessment.build_metrics) / len(assessment.build_metrics)
                report.add_note(
                    f"Built {len(assessment.build_metrics)} Docker images (total: {total_size:.1f} MB, avg time: {avg_time:.1f}s)"
                )

            # Add note about validation results
            if assessment.metrics.validation_issues:
                error_count = count_errors(assessment.metrics.validation_issues)
                warning_count = count_warnings(assessment.metrics.validation_issues)
                if generation_result.docker_images:
                    report.add_note(f"Dockerfile validation: {error_count} errors, {warning_count} warnings")
                if generation_result.k8s_manifests:
                    k8s_issues = [
                        i for i in assessment.metrics.validation_issues
                        if 'k8s' in i.file_path.lower() or 'kubectl' in i.rule_id.lower() or 'kube' in i.rule_id.lower()
                    ]
                    k8s_errors = count_errors(k8s_issues)
                    k8s_warnings = count_warnings(k8s_issues)
                    report.add_note(f"K8s validation: {k8s_errors} errors, {k8s_warnings} warnings")

            # Check if there were Docker build errors
            report.build_success = None if not generation_result.docker_images else not assessment.build_failed

            # Add note about deployment (only if no build errors)
            if generation_result.k8s_manifests and not assessment.build_failed:
                namespace = generation_result.run_context.k8s_namespace
                report.add_note(f"Deployed to namespace: {namespace}")

            # Step 3: Runtime validation - test endpoint health
            # Skip if there were Docker build errors
            if assessment.runtime_issues is not None:
                if generation_result.test_endpoints:
                    endpoints_str = ", ".join(generation_result.test_endpoints)
                    report.add_note(f"Testing endpoints: {endpoints_str}")

                if assessment.runtime_issues:
                    error_count = count_errors(assessment.runtime_issues)
                    warning_count = count_warnings(assessment.runtime_issues)
                    if error_count > 0:
                        report.add_note(f"Runtime validation: {error_count} errors, {warning_count} warnings")
                    elif warning_count > 0:
                        report.add_note(f"Runtime validation: {warning_count} warnings (no errors)")
                else:
                    report.add_note("Runtime validation passed: endpoint is healthy")

                report.runtime_success = assessment.runtime_success
            elif assessment.build_failed:
                report.add_note("Runtime validation skipped due to Docker build errors")
                report.runtime_success = False
            else:
                # Runtime validation was not executed (no test endpoint or no k8s manifests)
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

    def _generate_with_metrics(self, repo_url: str) -> tuple[GenerationResult, ExecutionMetrics, Optional['RepositoryWorkspace']]:
        """Generate configurations while collecting execution metrics."""
        try:
            # Use the generator integration to handle the generation process
            generation_result, execution_metrics, workspace = self.generator_integration.generate_with_monitoring(repo_url)
            return generation_result, execution_metrics, workspace

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

            return generation_result, execution_metrics, None

    def _assess_quality(
        self,
        generation_result: GenerationResult,
        workspace: 'RepositoryWorkspace'
    ) -> QualityAssessmentResult:
        """Assess the quality of generated configurations."""

        # Create validator with workspace and run_context from generation result
        validator = ConfigurationValidator(workspace, generation_result.run_context, self.generator_config)
        assessor = QualityAssessor(validator)

        assessment = assessor.assess(
            generation_result,
            test_endpoints=generation_result.test_endpoints,
        )
        return assessment


    def save_report(self, report: EvaluationReport, output_dir: str = "./evaluation_reports") -> str:
        """Save evaluation report to file."""
        return self.reporter.save_report(report, output_dir)
