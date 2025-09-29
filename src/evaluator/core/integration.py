import logging
import time
from typing import Dict, Any, Optional, List

from .models import GenerationResult, ExecutionMetrics
from ...generator.core.generator import ConfigurationGenerator
from ...utils.repository_utils import extract_repo_name
from ..metrics.parser import StructuredResultParser


class GeneratorIntegration:
    """
    Integration layer between evaluator and generator to capture
    metrics and structured results.
    """

    def __init__(self, generator: ConfigurationGenerator):
        self.generator = generator
        self.logger = logging.getLogger(__name__)


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
            config_output, messages = self.generator.generate(repo_url)

            # Create basic execution metrics
            generation_time = time.time() - start_time
            execution_metrics = ExecutionMetrics()
            execution_metrics.total_time = generation_time

            # Extract metrics from messages if needed
            if messages:
                final_message = messages[-1].content
                parser = StructuredResultParser()
                agent_metrics = parser.extract_metrics_from_output(str(final_message))
                if agent_metrics:
                    execution_metrics.tool_calls_count = agent_metrics.get("tool_calls_count", 0)
                    execution_metrics.tool_calls_breakdown = agent_metrics.get("tool_calls_breakdown", {})
                    execution_metrics.tokens_used = agent_metrics.get("total_tokens", 0)
                    execution_metrics.input_tokens = agent_metrics.get("input_tokens", 0)
                    execution_metrics.output_tokens = agent_metrics.get("output_tokens", 0)

            # Use structured output from ConfigurationOutput
            self.logger.info("Using structured output from ConfigurationOutput")
            total_files = len(config_output.dockerfiles) + len(config_output.kubernetes_files)
            success = total_files > 0
            execution_metrics.success_detected_via_done = True

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
