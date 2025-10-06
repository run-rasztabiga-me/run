import logging
import time
from typing import Dict, Any, Optional
from langsmith import Client


class LangSmithMetricsCollector:
    """Collects execution metrics from LangSmith traces."""

    def __init__(self, project_name: str = "run"):
        self.logger = logging.getLogger(__name__)
        self.client = Client()
        self.project_name = project_name

    def fetch_metrics(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch metrics from LangSmith for a given run_id.

        Args:
            run_id: The LangSmith run identifier

        Returns:
            Dictionary containing aggregated metrics or None if fetch fails
        """
        try:
            # Fetch the main run, retrying if status is still pending
            run = self._wait_for_run(run_id)

            if run is None:
                self.logger.error(f"Failed to fetch run {run_id} or run is still pending")
                return None

            # Initialize metrics
            total_tokens = 0
            input_tokens = 0
            output_tokens = 0
            tool_calls_breakdown = {}

            # Aggregate token metrics from the run
            if run.prompt_tokens:
                input_tokens += run.prompt_tokens
            if run.completion_tokens:
                output_tokens += run.completion_tokens
            if run.total_tokens:
                total_tokens += run.total_tokens

            # TODO trzeba to robic? sam run zawiera messages juz
            # Fetch child runs (tool calls) for the same trace
            # Note: We need to fetch all runs in the trace, not just direct children
            # LangSmith API doesn't support parent_run_id filter, so we fetch by trace_id
            child_runs = list(self.client.list_runs(
                project_name=self.project_name,
                trace_id=run.trace_id,
                is_root=False  # Get only child runs, not the root
            ))

            # Count tool calls by name
            for child_run in child_runs:
                if child_run.run_type == "tool":
                    tool_name = child_run.name
                    tool_calls_breakdown[tool_name] = tool_calls_breakdown.get(tool_name, 0) + 1

            # Calculate total if not provided
            if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
                total_tokens = input_tokens + output_tokens

            self.logger.info(
                f"Fetched metrics from LangSmith: {sum(tool_calls_breakdown.values())} tool calls, "
                f"{total_tokens} tokens"
            )

            return {
                "tool_calls_count": sum(tool_calls_breakdown.values()),
                "tool_calls_breakdown": tool_calls_breakdown,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch metrics from LangSmith: {str(e)}")
            return None

    def _wait_for_run(self, run_id: str, max_retries: int = 5, retry_delay: int = 5):
        """
        Wait for a run to complete, retrying if status is still pending.

        Args:
            run_id: The LangSmith run identifier
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries

        Returns:
            The run object if successful, None if still pending after retries
        """
        for attempt in range(max_retries):
            run = self.client.read_run(run_id)

            if run.status != "pending":
                return run

            if attempt < max_retries - 1:
                self.logger.info(
                    f"Run {run_id} is still pending, retrying in {retry_delay} seconds... "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
            else:
                self.logger.warning(f"Run {run_id} is still pending after {max_retries} attempts")

        return None
