import logging
import time

from .models import GenerationResult, ExecutionMetrics
from ...generator.core.generator import ConfigurationGenerator
from ...utils.repository_utils import extract_repo_name
from ..metrics.langsmith_collector import LangSmithMetricsCollector


class GeneratorIntegration:
    """
    Integration layer between evaluator and generator to capture
    metrics and structured results.
    """

    def __init__(self, generator: ConfigurationGenerator):
        self.generator = generator
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = LangSmithMetricsCollector()


    def generate_with_monitoring(self, repo_url: str) -> tuple[GenerationResult, ExecutionMetrics]:
        """
        Generate configurations with full monitoring and structured result capture.

        Args:
            repo_url: Repository URL to process

        Returns:
            Tuple of generation result and execution metrics
        """
        repo_name = extract_repo_name(repo_url)
        start_time = time.time()

        try:
            # Call the actual generator
            self.logger.info(f"Starting real generation for {repo_url}")
            config_output, messages, run_id = self.generator.generate(repo_url)

            # Create basic execution metrics
            generation_time = time.time() - start_time
            execution_metrics = ExecutionMetrics()
            execution_metrics.total_time = generation_time
            execution_metrics.run_id = run_id

            # Extract metrics from LangSmith if run_id is available
            if run_id:
                self.logger.info(f"Fetching metrics from LangSmith for run_id: {run_id}")
                langsmith_metrics = self.metrics_collector.fetch_metrics(run_id)
                if langsmith_metrics:
                    execution_metrics.tool_calls_count = langsmith_metrics.get("tool_calls_count", 0)
                    execution_metrics.tool_calls_breakdown = langsmith_metrics.get("tool_calls_breakdown", {})
                    execution_metrics.tokens_used = langsmith_metrics.get("total_tokens", 0)
                    execution_metrics.input_tokens = langsmith_metrics.get("input_tokens", 0)
                    execution_metrics.output_tokens = langsmith_metrics.get("output_tokens", 0)
            else:
                self.logger.warning("No run_id available, metrics will not be collected from LangSmith")

            # Use structured output from ConfigurationOutput
            self.logger.info("Using structured output from ConfigurationOutput")
            total_files = len(config_output.dockerfiles) + len(config_output.kubernetes_files)
            success = total_files > 0

            generation_result = GenerationResult(
                repo_url=repo_url,
                repo_name=repo_name,
                success=success,
                dockerfiles=config_output.dockerfiles,
                k8s_manifests=config_output.kubernetes_files,
                generation_time=generation_time
            )


        except Exception as e:
            generation_time = time.time() - start_time
            execution_metrics = ExecutionMetrics()
            execution_metrics.total_time = generation_time
            execution_metrics.error_count += 1

            generation_result = GenerationResult(
                repo_url=repo_url,
                repo_name=repo_name,
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time
            )

        return generation_result, execution_metrics
