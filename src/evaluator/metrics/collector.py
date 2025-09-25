import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

from ..core.models import ExecutionMetrics


class MetricsCollector:
    """Collects execution metrics during configuration generation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._start_time: Optional[float] = None
        self._tool_calls: Dict[str, int] = {}
        self._total_tool_calls = 0
        self._error_count = 0
        self._retry_count = 0
        self._collecting = False

    def start_collection(self) -> None:
        """Start collecting metrics."""
        self._start_time = time.time()
        self._tool_calls.clear()
        self._total_tool_calls = 0
        self._error_count = 0
        self._retry_count = 0
        self._collecting = True
        self.logger.debug("Started metrics collection")

    def stop_collection(self) -> ExecutionMetrics:
        """Stop collecting metrics and return the collected data."""
        if not self._collecting:
            self.logger.warning("Metrics collection was not started")
            return ExecutionMetrics(total_time=0, tool_calls_count=0)

        total_time = time.time() - self._start_time if self._start_time else 0
        self._collecting = False

        metrics = ExecutionMetrics(
            total_time=total_time,
            tool_calls_count=self._total_tool_calls,
            tool_calls_breakdown=self._tool_calls.copy(),
            error_count=self._error_count,
            retry_count=self._retry_count
        )

        self.logger.debug(f"Stopped metrics collection. Total time: {total_time:.2f}s, Tool calls: {self._total_tool_calls}")
        return metrics

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        if not self._collecting:
            return

        self._tool_calls[tool_name] = self._tool_calls.get(tool_name, 0) + 1
        self._total_tool_calls += 1
        self.logger.debug(f"Recorded tool call: {tool_name}")

    def record_error(self) -> None:
        """Record an error."""
        if not self._collecting:
            return

        self._error_count += 1
        self.logger.debug(f"Recorded error. Total errors: {self._error_count}")

    def record_retry(self) -> None:
        """Record a retry attempt."""
        if not self._collecting:
            return

        self._retry_count += 1
        self.logger.debug(f"Recorded retry. Total retries: {self._retry_count}")

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage (placeholder for future LangSmith integration)."""
        if not self._collecting:
            return

        # This will be implemented when integrating with LangSmith
        # For now, just log the information
        self.logger.debug(f"Token usage: {input_tokens} input, {output_tokens} output")

    def record_cost(self, cost: float) -> None:
        """Record estimated cost (placeholder for future implementation)."""
        if not self._collecting:
            return

        # This will be implemented when integrating with cost calculation
        self.logger.debug(f"Estimated cost: ${cost:.4f}")

    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager to measure the duration of an operation."""
        if not self._collecting:
            yield
            return

        start = time.time()
        self.logger.debug(f"Starting operation: {operation_name}")

        try:
            yield
        except Exception as e:
            self.record_error()
            self.logger.debug(f"Operation {operation_name} failed: {str(e)}")
            raise
        finally:
            duration = time.time() - start
            self.logger.debug(f"Operation {operation_name} completed in {duration:.2f}s")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if not self._collecting:
            return {}

        current_time = time.time() - self._start_time if self._start_time else 0
        return {
            "elapsed_time": current_time,
            "tool_calls_count": self._total_tool_calls,
            "tool_calls_breakdown": self._tool_calls.copy(),
            "error_count": self._error_count,
            "retry_count": self._retry_count,
            "collecting": self._collecting
        }

    def is_collecting(self) -> bool:
        """Check if metrics collection is active."""
        return self._collecting


class LangSmithMetricsCollector(MetricsCollector):
    """Extended metrics collector with LangSmith integration.

    TODO: Improve LangSmith integration to properly fetch metrics from traces.
    Current implementation falls back to parsing agent output directly.
    Future enhancements:
    - Use proper LangSmith session/trace tracking
    - Integrate with LangSmith callbacks during agent execution
    - Fetch detailed metrics from LangSmith API with correct session context
    """

    def __init__(self, langsmith_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.langsmith_config = langsmith_config or {}
        self.trace_id: Optional[str] = None
        self.session_id: Optional[str] = None

        # TODO: Improve LangSmith client initialization and session management
        try:
            from langsmith import Client
            self.client = Client()
            self.logger.info("LangSmith client initialized (basic functionality)")
        except ImportError:
            self.logger.warning("LangSmith not available, falling back to basic metrics")
            self.client = None

    def start_collection(self) -> None:
        """Start collecting metrics with LangSmith tracing."""
        super().start_collection()

        if self.client:
            # LangSmith tracing is handled automatically via environment variables
            # We'll capture the session after execution
            self.logger.debug("LangSmith tracing active")

    def stop_collection(self) -> ExecutionMetrics:
        """Stop collecting metrics and fetch data from LangSmith."""
        metrics = super().stop_collection()

        # TODO: Implement proper LangSmith metrics fetching
        # For now, metrics will be extracted from agent output in the integration layer
        if self.client:
            self.logger.debug("LangSmith tracing completed - metrics will be parsed from agent output")

        return metrics

    def get_langsmith_trace_url(self) -> Optional[str]:
        """Get LangSmith trace URL for the current session."""
        if self.client and self.trace_id:
            return f"https://smith.langchain.com/o/{self.client.info.tenant_id}/projects/default/r/{self.trace_id}"
        return None