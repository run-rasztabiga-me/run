from __future__ import annotations

from typing import List

from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import ValidationIssue, ValidationSeverity, has_error_issues


class DockerfileSyntaxValidationStep:
    """Validate Dockerfile syntax using `docker build --check`."""

    name = "docker_syntax"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        if not state.dockerfiles:
            return ValidationStepResult()

        issues: List[ValidationIssue] = []
        for dockerfile_path in state.dockerfiles:
            issues.extend(_validate_dockerfile_syntax(context, dockerfile_path))

        return ValidationStepResult(issues=issues, continue_pipeline=not has_error_issues(issues))


class KubernetesSyntaxValidationStep:
    """Validate Kubernetes manifest syntax via `kubectl --dry-run`."""

    name = "k8s_syntax"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        if not state.manifests:
            return ValidationStepResult()

        issues: List[ValidationIssue] = []
        for manifest_path in state.manifests:
            issues.extend(_validate_manifest_syntax(context, manifest_path))

        return ValidationStepResult(issues=issues, continue_pipeline=not has_error_issues(issues))


def _validate_dockerfile_syntax(context: ValidationContext, dockerfile_path: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    dockerfile_full_path = context.workspace.get_full_path(dockerfile_path)

    if not dockerfile_full_path.exists():
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Dockerfile not found at {dockerfile_path}",
                rule_id="DOCKER_NOT_FOUND",
            )
        )
        return issues

    result = context.command_runner.run(
        [
            "docker",
            "build",
            "--check",
            "-f",
            str(dockerfile_full_path),
            ".",
        ],
        cwd=dockerfile_full_path.parent,
        timeout=30,
    )

    if result.timed_out:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Dockerfile syntax validation timed out",
                rule_id="DOCKER_TIMEOUT",
            )
        )
        return issues

    if not result.tool_available:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Docker CLI not available - cannot validate Dockerfile syntax",
                rule_id="DOCKER_CLI_NOT_FOUND",
            )
        )
        return issues

    if result.stdout:
        issues.extend(_parse_docker_check_output(result.stdout, dockerfile_path))

    if (result.return_code or 0) != 0 and not issues:
        error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Dockerfile validation failed: {error_msg}",
                rule_id="DOCKER_VALIDATION_FAILED",
            )
        )

    if result.exception and not issues:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate Dockerfile syntax: {result.stderr or str(result.exception)}",
                rule_id="DOCKER_ERROR",
            )
        )

    return issues


def _validate_manifest_syntax(context: ValidationContext, manifest_path: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    manifest_full_path = context.workspace.get_full_path(manifest_path)

    # Skip non-YAML files
    if manifest_full_path.suffix.lower() not in [".yaml", ".yml"]:
        context.logger.debug("Skipping non-YAML file: %s", manifest_path)
        return issues

    result = context.command_runner.run(
        ["kubectl", "apply", "--dry-run=server", "-f", str(manifest_full_path)],
        timeout=30,
    )

    if result.timed_out:
        issues.append(
            ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="kubectl validation timed out",
                rule_id="KUBECTL_TIMEOUT",
            )
        )
        return issues

    if not result.tool_available:
        issues.append(
            ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="kubectl not installed - Kubernetes validation skipped",
                rule_id="KUBECTL_NOT_FOUND",
            )
        )
        return issues

    if (result.return_code or 0) != 0:
        error_msg = result.stderr.strip() or "Unknown kubectl error"
        issues.append(
            ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Kubernetes validation failed: {error_msg}",
                rule_id="KUBECTL_VALIDATION_FAILED",
            )
        )

    if result.exception and not issues:
        issues.append(
            ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate Kubernetes manifest: {result.stderr or str(result.exception)}",
                rule_id="KUBECTL_ERROR",
            )
        )

    return issues


def _parse_docker_check_output(output: str, dockerfile_path: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    lines = output.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if not (line.startswith("WARNING:") or line.startswith("ERROR:")):
            i += 1
            continue

        severity = ValidationSeverity.WARNING if line.startswith("WARNING:") else ValidationSeverity.ERROR

        parts = line.split(" - ", 1)
        if len(parts) >= 2:
            rule_info = parts[0].split(":", 1)[1].strip()
            description = parts[1]
        else:
            rule_info = line.split(":", 1)[1].strip() if ":" in line else "DOCKER_CHECK"
            description = ""

        line_number = None
        message_parts = [description] if description else []

        j = i + 1
        while j < len(lines) and j < i + 5:
            next_line = lines[j].strip()
            if next_line.startswith("Dockerfile:"):
                try:
                    line_number = int(next_line.split(":", 1)[1])
                except (IndexError, ValueError):
                    pass
            elif next_line and not next_line.startswith("---"):
                message_parts.append(next_line)
            j += 1

        message = " ".join(message_parts).strip() if message_parts else rule_info

        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=line_number,
                severity=severity,
                message=message,
                rule_id=rule_info.split()[0] if rule_info else "DOCKER_CHECK",
            )
        )

        i = j

    return issues
