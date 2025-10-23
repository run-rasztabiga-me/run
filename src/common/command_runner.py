from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


@dataclass(slots=True)
class CommandResult:
    """Represents the outcome of an external command execution."""

    command: Sequence[str]
    return_code: Optional[int]
    stdout: str
    stderr: str
    duration: float
    timed_out: bool
    tool_available: bool
    exception: Optional[BaseException] = None

    def succeeded(self) -> bool:
        """Return True when the command finished successfully."""
        return self.return_code == 0 and not self.timed_out and self.tool_available


class CommandRunner:
    """Thin wrapper over subprocess that captures execution metadata."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[float] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        """Execute a command and capture stdout, stderr, timings, and failures."""
        start = time.time()
        try:
            self.logger.debug("Executing command: %s (cwd=%s)", " ".join(command), cwd)
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env=dict(env) if env else None,
            )
            duration = time.time() - start
            return CommandResult(
                command=command,
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration=duration,
                timed_out=False,
                tool_available=True,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start
            self.logger.warning("Command timed out after %.2fs: %s", duration, " ".join(command))
            return CommandResult(
                command=command,
                return_code=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                duration=duration,
                timed_out=True,
                tool_available=True,
                exception=exc,
            )
        except FileNotFoundError as exc:
            duration = time.time() - start
            self.logger.error("Command not found: %s", command[0])
            return CommandResult(
                command=command,
                return_code=None,
                stdout="",
                stderr=f"Command not found: {command[0]}",
                duration=duration,
                timed_out=False,
                tool_available=False,
                exception=exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            duration = time.time() - start
            self.logger.error("Command execution failed: %s", exc)
            return CommandResult(
                command=command,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration=duration,
                timed_out=False,
                tool_available=True,
                exception=exc,
            )

