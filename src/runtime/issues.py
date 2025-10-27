"""Shared issue representation for runtime operations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

IssueSeverity = Literal["error", "warning", "info"]


@dataclass(slots=True)
class RuntimeIssue:
    """Lightweight issue representation for runtime operations."""

    code: str
    message: str
    severity: IssueSeverity = "error"
    subject: Optional[str] = None
    details: Optional[str] = None

    def is_error(self) -> bool:
        """Return True when the issue is considered an error."""
        return self.severity == "error"
