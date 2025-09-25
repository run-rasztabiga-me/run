import json
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager

from .models import GenerationResult, ExecutionMetrics
from ...generator.core.generator import ConfigurationGenerator
from ..metrics.collector import MetricsCollector
from ...utils.repository_utils import extract_repo_name


class GeneratorIntegration:
    """
    Integration layer between evaluator and generator to capture
    metrics and structured results.
    """

    def __init__(self, generator: ConfigurationGenerator, metrics_collector: MetricsCollector):
        self.generator = generator
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)

        # Hooks for capturing generation events
        self.pre_generation_hooks: List[Callable] = []
        self.post_generation_hooks: List[Callable] = []
        self.tool_call_hooks: List[Callable] = []

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

        # Start metrics collection
        self.metrics_collector.start_collection()

        # Execute pre-generation hooks
        for hook in self.pre_generation_hooks:
            try:
                hook(repo_url)
            except Exception as e:
                self.logger.warning(f"Pre-generation hook failed: {str(e)}")

        try:
            # Call the actual generator and capture output
            with self._capture_generation_context():
                self.logger.info(f"Starting real generation for {repo_url}")
                agent_output = self.generator.generate(repo_url)

            # Stop metrics collection
            execution_metrics = self.metrics_collector.stop_collection()
            generation_time = time.time() - start_time
            execution_metrics.total_time = generation_time

            # Parse agent output for structured results
            parser = StructuredResultParser()
            parsed_results = parser.parse_agent_output(agent_output)

            # Extract metrics from agent output
            agent_metrics = parser.extract_metrics_from_output(agent_output)
            if agent_metrics:
                execution_metrics.tool_calls_count = agent_metrics.get("tool_calls_count", 0)
                execution_metrics.tool_calls_breakdown = agent_metrics.get("tool_calls_breakdown", {})
                execution_metrics.tokens_used = agent_metrics.get("total_tokens", 0)
                execution_metrics.input_tokens = agent_metrics.get("input_tokens", 0)
                execution_metrics.output_tokens = agent_metrics.get("output_tokens", 0)

            # Use parsed results if available, otherwise fallback to filesystem detection
            if parsed_results:
                self.logger.info("Using parsed results from agent output")
                success = parsed_results["success"]
                generated_files = parsed_results["generated_files"]
                execution_metrics.success_detected_via_done = True
            else:
                self.logger.info("Falling back to filesystem detection")
                generated_files = self._detect_generated_files(repo_name)
                success = len(generated_files) > 0
                execution_metrics.success_detected_via_done = False

            generation_result = GenerationResult(
                repo_url=repo_url,
                repo_name=repo_name,
                success=success,
                generated_files=generated_files,
                dockerfile_path=self._find_dockerfile(repo_name),
                k8s_manifests_path=self._find_k8s_manifests(repo_name),
                generation_time=generation_time
            )

            # Execute post-generation hooks
            for hook in self.post_generation_hooks:
                try:
                    hook(repo_url, generation_result)
                except Exception as e:
                    self.logger.warning(f"Post-generation hook failed: {str(e)}")

        except Exception as e:
            execution_metrics = self.metrics_collector.stop_collection()
            execution_metrics.total_time = time.time() - start_time
            execution_metrics.error_count += 1

            generation_result = GenerationResult(
                repo_url=repo_url,
                repo_name=repo_name,
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time
            )

        return generation_result, execution_metrics

    @contextmanager
    def _capture_generation_context(self):
        """Context manager to capture generation process events."""
        self.logger.debug("Starting generation context capture")

        try:
            # TODO: Set up monitoring hooks here
            # - Intercept agent tool calls
            # - Monitor file system operations
            # - Capture stdout/stderr if needed
            yield

        finally:
            self.logger.debug("Ending generation context capture")
            # TODO: Clean up monitoring hooks

    def _detect_generated_files(self, repo_name: str) -> List[str]:
        """Detect files that were generated during the process."""
        import os

        tmp_dir = f"./tmp/{repo_name}"
        generated_files = []

        if not os.path.exists(tmp_dir):
            return generated_files

        # Look for common generated files
        potential_files = [
            os.path.join(tmp_dir, "Dockerfile"),
            os.path.join(tmp_dir, "k8s.yaml"),
            os.path.join(tmp_dir, "deployment.yaml"),
            os.path.join(tmp_dir, "service.yaml"),
            os.path.join(tmp_dir, "ingress.yaml"),
        ]

        # Check for k8s directory
        k8s_dir = os.path.join(tmp_dir, "k8s")
        if os.path.exists(k8s_dir):
            for file in os.listdir(k8s_dir):
                if file.endswith(('.yaml', '.yml')):
                    potential_files.append(os.path.join(k8s_dir, file))

        # Filter existing files
        for file_path in potential_files:
            if os.path.exists(file_path):
                generated_files.append(file_path)

        return generated_files

    def _find_dockerfile(self, repo_name: str) -> Optional[str]:
        """Find the generated Dockerfile."""
        import os

        dockerfile_path = f"./tmp/{repo_name}/Dockerfile"
        return dockerfile_path if os.path.exists(dockerfile_path) else None

    def _find_k8s_manifests(self, repo_name: str) -> Optional[str]:
        """Find the generated Kubernetes manifests directory or file."""
        import os

        tmp_dir = f"./tmp/{repo_name}"

        # Check for k8s directory
        k8s_dir = os.path.join(tmp_dir, "k8s")
        if os.path.exists(k8s_dir):
            return k8s_dir

        # Check for individual manifest files
        manifest_files = [
            os.path.join(tmp_dir, f)
            for f in ["deployment.yaml", "service.yaml", "ingress.yaml", "k8s.yaml"]
        ]

        existing_manifests = [f for f in manifest_files if os.path.exists(f)]
        if existing_manifests:
            return tmp_dir  # Return directory containing manifests

        return None


    def add_pre_generation_hook(self, hook: Callable) -> None:
        """Add a hook to run before generation starts."""
        self.pre_generation_hooks.append(hook)

    def add_post_generation_hook(self, hook: Callable) -> None:
        """Add a hook to run after generation completes."""
        self.post_generation_hooks.append(hook)

    def add_tool_call_hook(self, hook: Callable) -> None:
        """Add a hook to run when tools are called."""
        self.tool_call_hooks.append(hook)


class StructuredResultParser:
    """
    Parser to extract structured results from generator output.
    This is a placeholder for when the generator is modified to return JSON.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_agent_output(self, output: str) -> Optional[Dict[str, Any]]:
        """
        Parse agent output for structured results, specifically looking for DONE message.

        Args:
            output: Complete agent output string

        Returns:
            Structured results if DONE message found, None otherwise
        """
        self.logger.debug("Parsing agent output for DONE message")

        # Check if agent completed successfully by finding DONE message
        if "DONE" not in output:
            self.logger.warning("No DONE message found in agent output")
            return None

        # Extract file paths mentioned in the output
        generated_files = self.extract_file_paths(output)

        # Look for success indicators in the output
        success_indicators = [
            "successfully",  # Matches "Created file ... successfully"
            "created file",  # Matches file creation messages
            "generated",
            "completed the task",
            "done"
        ]

        success = any(indicator.lower() in output.lower() for indicator in success_indicators)

        # If DONE message is present and we have generated files, consider it successful
        final_success = success and "DONE" in output and len(generated_files) > 0

        return {
            "success": final_success,
            "generated_files": generated_files,
            "completion_message": self._extract_done_message(output),
            "has_dockerfile": any("dockerfile" in path.lower() for path in generated_files),
            "has_k8s_manifests": any(path.lower().endswith(('.yaml', '.yml')) for path in generated_files)
        }

    def extract_file_paths(self, output: str) -> List[str]:
        """Extract file paths from agent output."""
        import re

        # Look for file path patterns in output
        file_patterns = [
            r'Created file ([^\s]+)',
            r'Modified file ([^\s]+)',
            r'Generated ([^\s]+)',
            r'Saved to ([^\s]+)'
        ]

        extracted_paths = []
        for pattern in file_patterns:
            matches = re.findall(pattern, output)
            extracted_paths.extend(matches)

        return extracted_paths

    def _extract_done_message(self, output: str) -> str:
        """Extract the completion message containing DONE."""
        import re

        # Look for lines containing DONE and surrounding context
        lines = output.split('\n')
        done_lines = []

        for i, line in enumerate(lines):
            if "DONE" in line:
                # Get the line with DONE and some context
                start_idx = max(0, i-2)
                end_idx = min(len(lines), i+3)
                done_lines.extend(lines[start_idx:end_idx])
                break

        return '\n'.join(done_lines) if done_lines else "DONE message found"

    def extract_metrics_from_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Extract metrics from agent output including token usage and tool calls."""
        import re

        metrics = {
            "tool_calls_count": 0,
            "tool_calls_breakdown": {},
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # Extract tool calls from the output
        # Look for tool call IDs to avoid double counting (each tool call has a unique ID)
        tool_call_id_pattern = r"'id': '(call_[^']+)', 'function': \{'arguments': '[^']*', 'name': '(\w+)'\}"
        tool_call_matches = re.findall(tool_call_id_pattern, output)

        # Use set to track unique tool call IDs to avoid double counting
        unique_tool_calls = set()
        for call_id, tool_name in tool_call_matches:
            if call_id not in unique_tool_calls:
                unique_tool_calls.add(call_id)
                metrics["tool_calls_breakdown"][tool_name] = metrics["tool_calls_breakdown"].get(tool_name, 0) + 1
                metrics["tool_calls_count"] += 1

        # Extract token usage from the output
        # Look for patterns like 'completion_tokens': 31, 'prompt_tokens': 1032
        token_patterns = {
            "input_tokens": r"'prompt_tokens': (\d+)",
            "output_tokens": r"'completion_tokens': (\d+)"
        }

        for token_type, pattern in token_patterns.items():
            matches = re.findall(pattern, output)
            if matches:
                # Sum all token counts found in the output
                metrics[token_type] = sum(int(match) for match in matches)

        # Calculate total tokens outside the loop
        metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]

        self.logger.info(f"Extracted metrics from agent output: {metrics['tool_calls_count']} tool calls, {metrics['total_tokens']} tokens")

        return metrics if metrics["tool_calls_count"] > 0 or metrics["total_tokens"] > 0 else None


# Future integration points for enhanced monitoring:

class LangGraphMonitor:
    """
    Monitor for LangGraph agent execution.
    This would integrate with LangGraph's callback system when available.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # TODO: Initialize LangGraph monitoring hooks

    def setup_monitoring(self, agent):
        """Setup monitoring hooks for LangGraph agent."""
        # TODO: Implement when LangGraph provides callback mechanisms
        pass


class LangSmithIntegration:
    """
    Integration with LangSmith for advanced metrics and tracing.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        # TODO: Initialize LangSmith client

    def start_trace(self, repo_url: str) -> str:
        """Start a LangSmith trace session."""
        # TODO: Implement LangSmith tracing
        return "trace_id_placeholder"

    def get_trace_metrics(self, trace_id: str) -> Dict[str, Any]:
        """Retrieve metrics from LangSmith trace."""
        # TODO: Implement metrics retrieval
        return {}