import os
import logging
from typing import List

from ..core.models import ValidationIssue, ValidationSeverity


class ConfigurationValidator:
    """Validates generated Docker and Kubernetes configurations using external linters.

    TODO: Integrate with external linters as described in Chapter 4:
    - Hadolint for Dockerfile validation
    - Kube-linter for Kubernetes manifest validation

    Current implementation provides basic file existence checks and placeholder scoring.
    Custom validation rules have been removed in favor of industry-standard linters.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_dockerfile(self, dockerfile_path: str) -> List[ValidationIssue]:
        """
        Validate a Dockerfile using Hadolint (external linter).

        TODO: Integrate with Hadolint as described in Chapter 4:
        - Install Hadolint binary or use Docker image
        - Execute Hadolint on the Dockerfile: hadolint --format json <dockerfile_path>
        - Parse Hadolint JSON output and convert to ValidationIssue objects
        - Support different output formats (JSON, checkstyle, etc.)
        - Map Hadolint rule codes (DL3xxx, DL4xxx, etc.) to rule_id field
        - Convert Hadolint severity levels to ValidationSeverity enum
        - Handle cases where Hadolint is not installed

        Args:
            dockerfile_path: Path to the Dockerfile

        Returns:
            List of validation issues found by Hadolint
        """
        if not os.path.exists(dockerfile_path):
            return [ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Dockerfile not found",
                rule_id="FILE_NOT_FOUND",
                category="file_missing"
            )]

        # TODO: Replace with actual Hadolint integration
        self.logger.info(f"TODO: Run Hadolint validation on {dockerfile_path}")
        return []

    def validate_k8s_manifests(self, manifests_path: str) -> List[ValidationIssue]:
        """
        Validate Kubernetes manifests using Kube-linter.

        TODO: Integrate with Kube-linter as described in Chapter 4:
        - Install Kube-linter binary
        - Execute Kube-linter on manifest files/directories: kube-linter lint --format json <manifests_path>
        - Parse Kube-linter JSON/SARIF output and convert to ValidationIssue objects
        - Handle both single files and directories
        - Map Kube-linter check names to rule_id field
        - Convert Kube-linter severity levels to ValidationSeverity enum
        - Handle cases where Kube-linter is not installed

        Args:
            manifests_path: Path to Kubernetes manifests (file or directory)

        Returns:
            List of validation issues found by Kube-linter
        """
        if not os.path.exists(manifests_path):
            return [ValidationIssue(
                file_path=manifests_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Kubernetes manifests not found",
                rule_id="FILE_NOT_FOUND",
                category="file_missing"
            )]

        # TODO: Replace with actual Kube-linter integration
        self.logger.info(f"TODO: Run Kube-linter validation on {manifests_path}")
        return []


class DockerLinter:
    """Specialized Docker linter using Hadolint.

    TODO: Implement Hadolint integration as described in Chapter 4:
    - Use subprocess to call Hadolint binary: hadolint --format json <dockerfile>
    - Parse JSON output and convert to ValidationIssue objects
    - Handle different severity levels (error, warning, info, style)
    - Support Hadolint configuration files (.hadolint.yml)
    - Handle cases where Hadolint is not installed
    - Consider using Docker image if binary not available: docker run --rm -i hadolint/hadolint
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def lint_with_hadolint(self, dockerfile_path: str) -> List[ValidationIssue]:
        """
        Run Hadolint on Dockerfile and return validation issues.

        TODO: Implement Hadolint execution:
        Example command: hadolint --format json /path/to/Dockerfile
        Example Docker command: docker run --rm -i hadolint/hadolint < /path/to/Dockerfile

        Expected JSON output format:
        [
          {
            "file": "/path/to/Dockerfile",
            "line": 1,
            "column": 1,
            "level": "error",
            "code": "DL3006",
            "message": "Always tag the version of an image explicitly."
          }
        ]

        Args:
            dockerfile_path: Path to the Dockerfile to validate

        Returns:
            List of validation issues from Hadolint
        """
        self.logger.info(f"TODO: Run Hadolint on {dockerfile_path}")
        return []


class KubernetesLinter:
    """Specialized Kubernetes linter using Kube-linter and other tools.

    TODO: Integrate with Kubernetes linters as described in Chapter 4:
    - Kube-linter for security and best practices
    - Optional: kubeval for schema validation
    - Optional: kube-score for additional recommendations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def lint_with_kube_linter(self, manifest_path: str) -> List[ValidationIssue]:
        """
        Run Kube-linter on Kubernetes manifests.

        TODO: Implement Kube-linter execution:
        Example command: kube-linter lint --format json /path/to/manifests

        Expected JSON output format includes checks for:
        - Security best practices
        - Resource limits and requests
        - Health checks (liveness/readiness probes)
        - Image pull policies
        - Service account configurations

        Args:
            manifest_path: Path to Kubernetes manifest file or directory

        Returns:
            List of validation issues from Kube-linter
        """
        self.logger.info(f"TODO: Run Kube-linter on {manifest_path}")
        return []

    def lint_with_kubeval(self, manifest_path: str) -> List[ValidationIssue]:
        """
        Run kubeval for Kubernetes schema validation.

        TODO: Optional integration with kubeval:
        - Validates manifests against Kubernetes API schemas
        - Useful for catching structural/syntax errors
        - Command: kubeval /path/to/manifests/*.yaml

        Args:
            manifest_path: Path to manifest file

        Returns:
            List of validation issues from kubeval
        """
        self.logger.info(f"TODO: Run kubeval on {manifest_path}")
        return []

    def lint_with_kube_score(self, manifest_path: str) -> List[ValidationIssue]:
        """
        Run kube-score for additional Kubernetes recommendations.

        TODO: Optional integration with kube-score:
        - Provides recommendations for production-ready configurations
        - Command: kube-score score /path/to/manifests/*.yaml

        Args:
            manifest_path: Path to manifest file

        Returns:
            List of validation issues from kube-score
        """
        self.logger.info(f"TODO: Run kube-score on {manifest_path}")
        return []