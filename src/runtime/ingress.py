"""Ingress runtime health checks reusable across evaluator and deployment flows."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

import requests
from requests import RequestException

from src.common.command_runner import CommandRunner
from src.generator.core.config import GeneratorConfig

from .issues import RuntimeIssue


@dataclass(slots=True)
class RuntimeCheckResult:
    """Result of performing runtime ingress checks."""

    issues: List[RuntimeIssue]
    ingress_url: Optional[str] = None

    @property
    def success(self) -> bool:
        return not any(issue.is_error() for issue in self.issues)


class IngressRuntimeChecker:
    """Validate ingress availability and endpoint health."""

    def __init__(self, command_runner: CommandRunner, config: GeneratorConfig, logger: Optional[logging.Logger] = None):
        self.command_runner = command_runner
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def check(self, namespace: str, test_endpoint: str) -> RuntimeCheckResult:
        issues: List[RuntimeIssue] = []

        ingress_url = self._get_ingress_url(namespace)
        if not ingress_url:
            issues.append(
                RuntimeIssue(
                    code="NO_INGRESS_URL",
                    message="No ingress URL found in cluster, skipping runtime health check",
                    severity="error",
                    subject=namespace,
                )
            )
            return RuntimeCheckResult(issues=issues, ingress_url=None)

        parsed_url = urlparse(ingress_url)
        hostname = parsed_url.hostname

        if not hostname:
            issues.append(
                RuntimeIssue(
                    code="INVALID_INGRESS_URL",
                    message=f"Failed to extract hostname from ingress URL: {ingress_url}",
                    subject=namespace,
                )
            )
            return RuntimeCheckResult(issues=issues, ingress_url=ingress_url)

        self._ensure_hosts_entry(hostname)

        self.logger.info("Waiting for ingress to be fully configured...")
        time.sleep(self.config.k8s_ingress_timeout)

        issues.extend(self._check_endpoint_health(ingress_url, test_endpoint))
        return RuntimeCheckResult(issues=issues, ingress_url=ingress_url)

    def _get_ingress_url(self, namespace: str) -> Optional[str]:
        result = self.command_runner.run(
            ["kubectl", "get", "ingress", "-n", namespace, "-o", "json"],
            timeout=30,
        )

        if result.timed_out:
            self.logger.warning("Timeout while getting ingress from cluster")
            return None

        if not result.tool_available:
            self.logger.warning("kubectl not available - cannot get ingress URL")
            return None

        if (result.return_code or 0) != 0:
            self.logger.warning("Failed to get ingresses from cluster: %s", result.stderr)
            return None

        try:
            data = json.loads(result.stdout) if result.stdout else {}
        except Exception as exc:
            self.logger.warning("Failed to parse kubectl ingress output: %s", exc)
            return None

        ingresses = (data or {}).get("items", [])
        if not ingresses:
            self.logger.warning("No ingresses found in namespace %s", namespace)
            return None

        ingress = ingresses[0]
        rules = ingress.get("spec", {}).get("rules", [])
        if not rules:
            self.logger.warning("Ingress has no rules in namespace %s", namespace)
            return None

        host = rules[0].get("host")
        if not host:
            self.logger.warning("Ingress rule missing host in namespace %s", namespace)
            return None

        tls = ingress.get("spec", {}).get("tls", [])
        protocol = "https" if tls else "http"

        ingress_name = ingress.get("metadata", {}).get("name", "unknown")
        self.logger.info("Found ingress '%s' with host: %s", ingress_name, host)

        return f"{protocol}://{host}"

    def _check_endpoint_health(self, base_url: str, endpoint_path: str) -> List[RuntimeIssue]:
        issues: List[RuntimeIssue] = []
        retries = 5
        request_timeout = 30

        if not endpoint_path.startswith("/"):
            endpoint_path = "/" + endpoint_path

        full_url = f"{base_url}{endpoint_path}"
        self.logger.info("Testing endpoint: %s", full_url)

        last_error = None
        for attempt in range(retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt
                    self.logger.info("Retry %s/%s after %ss...", attempt + 1, retries, wait_time)
                    time.sleep(wait_time)

                response = requests.get(full_url, timeout=request_timeout, verify=False)

                if 200 <= response.status_code < 300:
                    self.logger.info("Endpoint %s is healthy (status %s)", full_url, response.status_code)
                    return []
                last_error = f"Endpoint returned non-2xx status: {response.status_code}"
                self.logger.warning("Attempt %s: %s", attempt + 1, last_error)

            except RequestException as exc:
                last_error = str(exc)
                self.logger.warning("Attempt %s: Connection failed - %s", attempt + 1, last_error)

        issues.append(
            RuntimeIssue(
                code="ENDPOINT_HEALTH_CHECK_FAILED",
                message=f"Endpoint {full_url} not accessible after {retries} attempts: {last_error}",
                subject=full_url,
            )
        )

        return issues

    def _ensure_hosts_entry(self, hostname: str) -> None:
        ip_address = self.config.k8s_cluster_ip
        entry = f"{ip_address}   {hostname}"

        try:
            with open("/etc/hosts", "r") as handle:
                for line in handle:
                    if hostname in line and not line.strip().startswith("#"):
                        self.logger.info("Entry for %s already exists in /etc/hosts", hostname)
                        return

            self.logger.info("Adding /etc/hosts entry: %s", entry)
            result = self.command_runner.run(
                ["sudo", "sh", "-c", f'echo "{entry}" >> /etc/hosts'],
                timeout=10,
            )

            if result.timed_out or (result.return_code or 0) != 0:
                self.logger.warning("Failed to add /etc/hosts entry: %s", result.stderr)
            else:
                self.logger.info("Successfully added /etc/hosts entry for %s", hostname)

        except Exception as exc:
            self.logger.warning("Error ensuring /etc/hosts entry: %s", exc)
