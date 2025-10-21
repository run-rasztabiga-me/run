"""Experiment runner orchestrating evaluator executions."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.evaluator import ConfigurationEvaluator
from ..core.models import EvaluationReport
from ...generator.core.config import GeneratorConfig
from ...utils.repository_utils import extract_repo_name
from .config import ExperimentDefinition, ExperimentSuite, ModelSpec, load_experiment_suite


logger = logging.getLogger(__name__)


@dataclass
class _RunContext:
    """Internal helper capturing run-level metadata."""

    experiment: ExperimentDefinition
    model: ModelSpec
    repo_url: str
    repetition_index: int
    timestamp: datetime

    @property
    def repetition_label(self) -> str:
        return f"run{self.repetition_index + 1}"


class ExperimentRunner:
    """Coordinate experiment execution across repositories and models."""

    def __init__(
        self,
        suite: ExperimentSuite,
        output_dir: Optional[str] = None,
    ):
        self.suite = suite
        base_output = output_dir or suite.output_dir or "./evaluation_reports"
        self.base_dir = Path(base_output) / "experiments"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Execute all experiments in the suite sequentially."""
        for experiment in self.suite.experiments:
            self._run_experiment(experiment)

    def _run_experiment(self, experiment: ExperimentDefinition) -> None:
        experiment_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.base_dir / experiment.safe_name / experiment_timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting experiment '%s' (%s)", experiment.name, experiment_dir)

        summary_rows: List[Dict[str, Any]] = []

        for repo_url in experiment.repos:
            repo_name = extract_repo_name(repo_url)

            for model in experiment.models:
                model_dir = experiment_dir / model.identifier / repo_name
                model_dir.mkdir(parents=True, exist_ok=True)

                for repetition in range(experiment.repetitions):
                    run_context = _RunContext(
                        experiment=experiment,
                        model=model,
                        repo_url=repo_url,
                        repetition_index=repetition,
                        timestamp=datetime.utcnow(),
                    )

                    report_path, report, summary_row = self._execute_run(
                        context=run_context,
                        output_dir=model_dir,
                    )
                    summary_rows.append(summary_row)
                    logger.info(
                        "Completed %s | repo=%s model=%s repetition=%s -> %s",
                        experiment.name,
                        repo_name,
                        model.identifier,
                        run_context.repetition_label,
                        report_path,
                    )

        self._write_summary_artifacts(experiment_dir, summary_rows)
        logger.info("Experiment '%s' finished. %d runs recorded.", experiment.name, len(summary_rows))

    def _execute_run(
        self,
        context: _RunContext,
        output_dir: Path,
    ) -> tuple[str, EvaluationReport, Dict[str, Any]]:
        generator_config = self._build_generator_config(context.experiment, context.model)
        evaluator = ConfigurationEvaluator(generator_config=generator_config)

        report = evaluator.evaluate_repository(context.repo_url)
        report.experiment_name = context.experiment.name
        report.model_name = context.model.name
        report.model_provider = context.model.provider
        report.repetition_index = context.repetition_index
        report.model_parameters = {
            "temperature": context.model.temperature,
            "seed": context.model.seed,
            **context.model.parameters,
        }
        report.extra_metadata.update({
            "generator_overrides": context.experiment.generator_overrides,
            "run_overrides": context.experiment.run_overrides,
        })

        report_path = evaluator.save_report(report, output_dir=str(output_dir))
        summary_row = self._build_summary_row(report, context, report_path, output_dir.parent.parent)
        return report_path, report, summary_row

    def _build_generator_config(self, experiment: ExperimentDefinition, model: ModelSpec) -> GeneratorConfig:
        config_payload: Dict[str, Any] = dict(experiment.generator_overrides)
        config_payload["model_name"] = model.name
        config_payload["model_provider"] = model.provider

        if model.temperature is not None:
            config_payload["temperature"] = model.temperature
        if model.seed is not None:
            config_payload["seed"] = model.seed

        # Allow arbitrary additional overrides from the model spec
        for key, value in model.parameters.items():
            config_payload[key] = value

        return GeneratorConfig(**config_payload)

    def _build_summary_row(
        self,
        report: EvaluationReport,
        context: _RunContext,
        report_path: str,
        experiment_dir: Path,
    ) -> Dict[str, Any]:
        repo_name = extract_repo_name(context.repo_url)
        relative_path = Path(report_path).relative_to(experiment_dir)
        exec_metrics = report.execution_metrics
        generation_result = report.generation_result
        quality_metrics = report.quality_metrics

        return {
            "experiment": context.experiment.name,
            "timestamp": context.timestamp.isoformat(),
            "repo_url": context.repo_url,
            "repo_name": repo_name,
            "model_provider": context.model.provider,
            "model_name": context.model.name,
            "model_label": context.model.label or context.model.identifier,
            "temperature": context.model.temperature,
            "seed": context.model.seed,
            "repetition": context.repetition_index,
            "status": report.status.value,
            "generation_success": bool(generation_result.success) if generation_result else False,
            "generation_time": generation_result.generation_time if generation_result else None,
            "overall_score": quality_metrics.overall_score if quality_metrics else None,
            "dockerfile_score": quality_metrics.dockerfile_score if quality_metrics else None,
            "k8s_score": quality_metrics.k8s_manifests_score if quality_metrics else None,
            "tool_calls": exec_metrics.tool_calls_count if exec_metrics else None,
            "tokens_used": exec_metrics.tokens_used if exec_metrics else None,
            "report_path": relative_path.as_posix(),
        }

    def _write_summary_artifacts(self, experiment_dir: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        summary_csv = experiment_dir / "summary.csv"
        summary_json = experiment_dir / "summary.json"

        fieldnames = [
            "experiment",
            "timestamp",
            "repo_url",
            "repo_name",
            "model_provider",
            "model_name",
            "model_label",
            "temperature",
            "seed",
            "repetition",
            "status",
            "generation_success",
            "generation_time",
            "overall_score",
            "dockerfile_score",
            "k8s_score",
            "tool_calls",
            "tokens_used",
            "report_path",
        ]

        with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        payload = {
            "generated_at": datetime.utcnow().isoformat(),
            "runs": rows,
        }
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_from_file(config_path: str | Path, output_dir: Optional[str] = None) -> None:
    """Convenience helper to run experiments directly from a config path."""
    suite = load_experiment_suite(config_path)
    runner = ExperimentRunner(suite, output_dir=output_dir)
    runner.run()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run configuration generation experiments.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiments YAML/JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional base directory for experiment outputs (defaults to suite or evaluation_reports/experiments).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("src").setLevel(logging.INFO)
    for noisy_logger in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    run_from_file(args.config, output_dir=args.output_dir)


__all__ = ["ExperimentRunner", "run_from_file", "main"]
