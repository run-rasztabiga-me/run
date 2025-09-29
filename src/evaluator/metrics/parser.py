import logging
import re
from typing import Dict, Any, Optional, List


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
        # Look for lines containing DONE and surrounding context
        lines = output.split('\n')
        done_lines = []

        for i, line in enumerate(lines):
            if "DONE" in line:
                # Get the line with DONE and some context
                start_idx = max(0, i - 2)
                end_idx = min(len(lines), i + 3)
                done_lines.extend(lines[start_idx:end_idx])
                break

        return '\n'.join(done_lines) if done_lines else "DONE message found"

    def extract_metrics_from_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Extract metrics from agent output including token usage and tool calls."""
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

        self.logger.info(
            f"Extracted metrics from agent output: {metrics['tool_calls_count']} tool calls, {metrics['total_tokens']} tokens")

        return metrics if metrics["tool_calls_count"] > 0 or metrics["total_tokens"] > 0 else None