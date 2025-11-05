import json
import csv
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from ..core.models import EvaluationReport, ValidationSeverity


class EvaluationReporter:
    """Generates reports from evaluation results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def save_report(self, report: EvaluationReport, output_dir: str = "./evaluation_reports") -> str:
        """
        Save individual evaluation report to JSON file.

        Args:
            report: Evaluation report to save
            output_dir: Directory to save reports

        Returns:
            Path to saved report file
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = []
        if report.experiment_name:
            filename_parts.append(self._sanitize_token(report.experiment_name))
        if report.model_name:
            filename_parts.append(self._sanitize_token(report.model_name))
        if report.prompt_id:
            filename_parts.append(self._sanitize_token(report.prompt_id))
        if report.repo_name:
            filename_parts.append(self._sanitize_token(report.repo_name))
        if report.repetition_index is not None:
            filename_parts.append(f"run{report.repetition_index + 1}")

        core_name = "__".join(filter(None, filename_parts)) or "report"
        filename = f"{core_name}_{timestamp}_{report.evaluation_id[:8]}.json"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._report_to_dict(report), f, indent=2, default=str)

            self.logger.info(f"Report saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
            raise

    def _report_to_dict(self, report: EvaluationReport) -> Dict[str, Any]:
        """Convert evaluation report to dictionary for serialization."""
        extra_metadata = report.extra_metadata or {}

        return {
            "repo_url": report.repo_url,
            "repo_name": report.repo_name,
            "evaluation_id": report.evaluation_id,
            "status": report.status.value,
            "start_time": report.start_time.isoformat(),
            "end_time": report.end_time.isoformat() if report.end_time else None,
            "total_evaluation_time": report.total_evaluation_time,
            "generation_result": self._generation_result_to_dict(report.generation_result) if report.generation_result else None,
            "execution_metrics": self._execution_metrics_to_dict(report.execution_metrics) if report.execution_metrics else None,
            "quality_metrics": self._quality_metrics_to_dict(report.quality_metrics) if report.quality_metrics else None,
            "notes": report.notes,
            "experiment_name": report.experiment_name,
            "model_name": report.model_name,
            "model_provider": report.model_provider,
            "model_parameters": report.model_parameters or None,
            "repetition_index": report.repetition_index,
            "prompt_id": report.prompt_id,
            "prompt_override": report.prompt_override,
            "build_success": report.build_success,
            "runtime_success": report.runtime_success,
            "extra_metadata": extra_metadata or None,
        }

    def _generation_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert generation result to dictionary."""
        workspace_dir = None
        run_id = None
        if getattr(result, "run_context", None):
            try:
                workspace_dir = str(result.run_context.workspace_dir.resolve())
            except Exception:
                workspace_dir = str(result.run_context.workspace_dir)
            run_id = result.run_context.run_id
        return {
            "repo_url": result.repo_url,
            "repo_name": result.repo_name,
            "success": result.success,
            "docker_images": [
                {
                    "dockerfile_path": img.dockerfile_path,
                    "image_tag": img.image_tag,
                    "build_context": img.build_context
                }
                for img in result.docker_images
            ],
            "k8s_manifests": result.k8s_manifests,
            "generation_time": result.generation_time,
            "error_message": result.error_message,
            "timestamp": result.timestamp.isoformat(),
            "workspace_dir": workspace_dir,
            "run_id": run_id,
        }

    def _execution_metrics_to_dict(self, metrics) -> Dict[str, Any]:
        """Convert execution metrics to dictionary."""
        return {
            "total_time": metrics.total_time,
            "tool_calls_count": metrics.tool_calls_count,
            "tool_calls_breakdown": metrics.tool_calls_breakdown,
            "tokens_used": metrics.tokens_used,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "error_count": metrics.error_count,
            "run_id": metrics.run_id,
            "docker_build_metrics": [
                {
                    "image_tag": m.image_tag,
                    "build_time": m.build_time,
                    "image_size_mb": m.image_size_mb,
                    "layers_count": m.layers_count
                }
                for m in metrics.docker_build_metrics
            ]
        }

    def _quality_metrics_to_dict(self, metrics) -> Dict[str, Any]:
        """Convert quality metrics to dictionary."""
        result = {
            "dockerfile_score": metrics.dockerfile_score,
            "k8s_manifests_score": metrics.k8s_manifests_score,
            "overall_score": metrics.overall_score,
            "validation_issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "rule_id": issue.rule_id
                }
                for issue in metrics.validation_issues
            ],
            # H3 hypothesis verification metrics
            "error_count": metrics.error_count,
            "warning_count": metrics.warning_count,
            "info_count": metrics.info_count,
            "has_errors": metrics.has_errors,
            "is_clean": metrics.is_clean,
            "dockerfile_syntax_valid": metrics.dockerfile_syntax_valid,
            "k8s_syntax_valid": metrics.k8s_syntax_valid,
        }

        # Add detailed scoring breakdown if available
        if hasattr(metrics, 'scoring_breakdown') and metrics.scoring_breakdown:
            result["scoring_breakdown"] = metrics.scoring_breakdown

        if getattr(metrics, "llm_judge_results", None):
            result["llm_judge_results"] = metrics.llm_judge_results

        return result

    def _sanitize_token(self, value: str) -> str:
        """Sanitize text for filesystem usage."""
        sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in (value or ""))
        sanitized = "-".join(filter(None, sanitized.split("-")))
        return sanitized or "value"
