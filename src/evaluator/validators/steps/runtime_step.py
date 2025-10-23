from __future__ import annotations

import json
import time
from typing import List
from urllib.parse import urlparse

import requests
from requests import RequestException

from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import ValidationIssue, ValidationSeverity


class RuntimeValidationStep:
    """Validate runtime availability via ingress endpoint checks."""

    name = "runtime"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        if not state.test_endpoint:
            return ValidationStepResult()

        namespace = context.run_context.k8s_namespace
        issues = _perform_runtime_validation(context, namespace, state.test_endpoint)
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        runtime_success = None if not issues else not has_errors
        if not issues:
            runtime_success = True

        return ValidationStepResult(
            issues=issues,
            runtime_success=runtime_success,
        )


def _perform_runtime_validation(context: ValidationContext, namespace: str, test_endpoint: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    ingress_url = _get_ingress_url(context, namespace)

    if not ingress_url:
        issues.append(
            ValidationIssue(
                file_path="runtime",
                line_number=None,
                severity=ValidationSeverity.WARNING,
                message="No ingress URL found in cluster, skipping runtime health check",
                rule_id="NO_INGRESS_URL",
            )
        )
        return issues

    context.logger.info("Found ingress URL: %s", ingress_url)

    parsed_url = urlparse(ingress_url)
    hostname = parsed_url.hostname

    if not hostname:
        issues.append(
            ValidationIssue(
                file_path="runtime",
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to extract hostname from ingress URL: {ingress_url}",
                rule_id="INVALID_INGRESS_URL",
            )
        )
        return issues

    _ensure_hosts_entry(context, hostname)

    context.logger.info("Waiting for ingress to be fully configured...")
    time.sleep(context.config.k8s_ingress_timeout)

    issues.extend(_check_endpoint_health(context, ingress_url, test_endpoint))
    return issues


def _get_ingress_url(context: ValidationContext, namespace: str) -> str | None:
    result = context.command_runner.run(
        ["kubectl", "get", "ingress", "-n", namespace, "-o", "json"],
        timeout=30,
    )

    if result.timed_out:
        context.logger.warning("Timeout while getting ingress from cluster")
        return None

    if not result.tool_available:
        context.logger.warning("kubectl not available - cannot get ingress URL")
        return None

    if (result.return_code or 0) != 0:
        context.logger.warning("Failed to get ingresses from cluster: %s", result.stderr)
        return None

    try:
        data = json.loads(result.stdout) if result.stdout else {}
    except Exception as exc:
        context.logger.warning("Failed to parse kubectl ingress output: %s", exc)
        return None

    ingresses = (data or {}).get("items", [])
    if not ingresses:
        context.logger.warning("No ingresses found in namespace %s", namespace)
        return None

    ingress = ingresses[0]
    rules = ingress.get("spec", {}).get("rules", [])
    if not rules:
        context.logger.warning("Ingress has no rules in namespace %s", namespace)
        return None

    host = rules[0].get("host")
    if not host:
        context.logger.warning("Ingress rule missing host in namespace %s", namespace)
        return None

    tls = ingress.get("spec", {}).get("tls", [])
    protocol = "https" if tls else "http"

    ingress_name = ingress.get("metadata", {}).get("name", "unknown")
    context.logger.info("Found ingress '%s' with host: %s", ingress_name, host)

    return f"{protocol}://{host}"


def _check_endpoint_health(context: ValidationContext, base_url: str, endpoint_path: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    retries = 5
    request_timeout = 30

    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path

    full_url = f"{base_url}{endpoint_path}"
    context.logger.info("Testing endpoint: %s", full_url)

    last_error = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt
                context.logger.info("Retry %s/%s after %ss...", attempt + 1, retries, wait_time)
                time.sleep(wait_time)

            response = requests.get(full_url, timeout=request_timeout, verify=False)

            if 200 <= response.status_code < 300:
                context.logger.info("Endpoint %s is healthy (status %s)", full_url, response.status_code)
                return []
            last_error = f"Endpoint returned non-2xx status: {response.status_code}"
            context.logger.warning("Attempt %s: %s", attempt + 1, last_error)

        except RequestException as exc:
            last_error = str(exc)
            context.logger.warning("Attempt %s: Connection failed - %s", attempt + 1, last_error)

    issues.append(
        ValidationIssue(
            file_path="runtime",
            line_number=None,
            severity=ValidationSeverity.ERROR,
            message=f"Endpoint {full_url} not accessible after {retries} attempts: {last_error}",
            rule_id="ENDPOINT_HEALTH_CHECK_FAILED",
        )
    )

    return issues


def _ensure_hosts_entry(context: ValidationContext, hostname: str) -> None:
    ip_address = context.config.k8s_cluster_ip
    entry = f"{ip_address}   {hostname}"

    try:
        with open("/etc/hosts", "r") as handle:
            for line in handle:
                if hostname in line and not line.strip().startswith("#"):
                    context.logger.info("Entry for %s already exists in /etc/hosts", hostname)
                    return

        context.logger.info("Adding /etc/hosts entry: %s", entry)
        result = context.command_runner.run(
            ["sudo", "sh", "-c", f'echo "{entry}" >> /etc/hosts'],
            timeout=10,
        )

        if result.timed_out or (result.return_code or 0) != 0:
            context.logger.warning("Failed to add /etc/hosts entry: %s", result.stderr)
        else:
            context.logger.info("Successfully added /etc/hosts entry for %s", hostname)

    except Exception as exc:
        context.logger.warning("Error ensuring /etc/hosts entry: %s", exc)
