from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import ValidationIssue, ValidationSeverity, has_error_issues


class DockerfileLinterValidationStep:
    """Run Hadolint against generated Dockerfiles."""

    name = "docker_linters"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        if not state.dockerfiles:
            return ValidationStepResult()

        issues: List[ValidationIssue] = []
        for dockerfile_path in state.dockerfiles:
            issues.extend(_run_hadolint(context, dockerfile_path))

        return ValidationStepResult(issues=issues, continue_pipeline=not has_error_issues(issues))


class KubernetesLinterValidationStep:
    """Run kube-linter against generated manifests."""

    name = "k8s_linters"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        if not state.manifests:
            return ValidationStepResult()

        issues: List[ValidationIssue] = []
        for manifest_path in state.manifests:
            issues.extend(_run_kube_linter(context, manifest_path))

        return ValidationStepResult(issues=issues, continue_pipeline=not has_error_issues(issues))


def _run_hadolint(context: ValidationContext, dockerfile_path: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    dockerfile_full_path = context.workspace.get_full_path(dockerfile_path)

    if not dockerfile_full_path.exists():
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Dockerfile not found for Hadolint analysis: {dockerfile_path}",
                rule_id="HADOLINT_FILE_NOT_FOUND",
            )
        )
        return issues

    result = context.command_runner.run(
        ["hadolint", "--format", "json", str(dockerfile_full_path)],
        timeout=30,
    )

    if result.timed_out:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Hadolint analysis timed out",
                rule_id="HADOLINT_TIMEOUT",
            )
        )
        return issues

    if not result.tool_available:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Hadolint not installed - static analysis skipped",
                rule_id="HADOLINT_NOT_FOUND",
            )
        )
        return issues

    if result.stdout:
        try:
            hadolint_issues = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to parse Hadolint output: {exc}",
                    rule_id="HADOLINT_PARSE_ERROR",
                )
            )
        else:
            for entry in hadolint_issues:
                severity = _map_hadolint_severity(entry.get("level", "error"))
                issues.append(
                    ValidationIssue(
                        file_path=dockerfile_path,
                        line_number=entry.get("line"),
                        severity=severity,
                        message=entry.get("message", ""),
                        rule_id=entry.get("code", "HADOLINT"),
                    )
                )
    elif (result.return_code or 0) != 0:
        error_msg = result.stderr.strip() or "Hadolint failed without output"
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=error_msg,
                rule_id="HADOLINT_ERROR",
            )
        )

    return issues


def _run_kube_linter(context: ValidationContext, manifest_path: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    manifest_full_path = context.workspace.get_full_path(manifest_path)

    # Skip non-YAML files
    if manifest_full_path.suffix.lower() not in [".yaml", ".yml"]:
        context.logger.debug("Skipping non-YAML file: %s", manifest_path)
        return issues

    config_path = Path(__file__).resolve().parent.parent / ".kube-linter.yaml"
    cmd = ["kube-linter", "lint", "--format", "json"]
    if config_path.exists():
        cmd.extend(["--config", str(config_path)])
    cmd.append(str(manifest_full_path))

    result = context.command_runner.run(cmd, timeout=30)

    if result.timed_out:
        issues.append(
            ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Kube-linter analysis timed out",
                rule_id="KUBE_LINTER_TIMEOUT",
            )
        )
        return issues

    if not result.tool_available:
        issues.append(
            ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Kube-linter not installed - static analysis skipped",
                rule_id="KUBE_LINTER_NOT_FOUND",
            )
        )
        return issues

    if result.stdout:
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    file_path=manifest_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to parse kube-linter output: {exc}",
                    rule_id="KUBE_LINTER_PARSE_ERROR",
                )
            )
        else:
            reports = output.get("Reports") or []
            for report in reports:
                diagnostic = report.get("Diagnostic", {})
                message = diagnostic.get("Message", "")
                severity = _map_kube_linter_severity(diagnostic.get("Severity", "warning"))
                issues.append(
                    ValidationIssue(
                        file_path=manifest_path,
                        line_number=None,
                        severity=severity,
                        message=message,
                        rule_id=report.get("Check", "KUBE_LINTER"),
                    )
                )
    elif (result.return_code or 0) != 0:
        issues.append(
            ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=result.stderr.strip() or "kube-linter failed without output",
                rule_id="KUBE_LINTER_ERROR",
            )
        )

    return issues


def _map_hadolint_severity(hadolint_level: str) -> ValidationSeverity:
    mapping = {
        "error": ValidationSeverity.ERROR,
        "warning": ValidationSeverity.WARNING,
        "info": ValidationSeverity.INFO,
        "style": ValidationSeverity.INFO,
    }
    return mapping.get(hadolint_level.lower(), ValidationSeverity.WARNING)


def _map_kube_linter_severity(kube_linter_level: str) -> ValidationSeverity:
    mapping = {
        "error": ValidationSeverity.ERROR,
        "warning": ValidationSeverity.WARNING,
        "info": ValidationSeverity.INFO,
    }
    return mapping.get(kube_linter_level.lower(), ValidationSeverity.WARNING)
