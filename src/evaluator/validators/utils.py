"""Shared helpers for evaluator validation steps."""
from __future__ import annotations

from typing import List

from ..core.models import ValidationIssue, ValidationSeverity
from src.runtime import RuntimeIssue


def runtime_issues_to_validation(
    issues: List[RuntimeIssue],
    *,
    default_subject: str,
) -> List[ValidationIssue]:
    """Convert runtime issues emitted by the PaaS runtime helpers into validator issues."""
    severity_map = {
        "error": ValidationSeverity.ERROR,
        "warning": ValidationSeverity.WARNING,
        "info": ValidationSeverity.INFO,
    }
    converted: List[ValidationIssue] = []

    for issue in issues:
        severity = severity_map.get(issue.severity, ValidationSeverity.ERROR)
        converted.append(
            ValidationIssue(
                file_path=issue.subject or default_subject,
                line_number=None,
                severity=severity,
                message=issue.message,
                rule_id=issue.code,
            )
        )

    return converted
