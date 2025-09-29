import json
import logging
import subprocess
import yaml
from typing import List
from pathlib import Path

from ..core.models import ValidationIssue, ValidationSeverity
from ...generator.core.repository import RepositoryManager


class ConfigurationValidator:
    """Validates generated Docker and Kubernetes configurations using external linters and build tests.

    Implements validation tasks:
    7. Dockerfile syntax validation
    8. Static analysis with Hadolint
    9. Docker image building (TODO)
    10. Kubernetes manifest syntax validation
    11. Static analysis with Kube-linter
    12. Kubernetes manifest application in Kind (TODO)
    13. Runtime validation - application availability (TODO)
    """

    def __init__(self, repository_manager: RepositoryManager):
        self.logger = logging.getLogger(__name__)
        self.repository_manager = repository_manager

    def validate_dockerfiles(self, dockerfile_paths: List[str]) -> List[ValidationIssue]:
        """
        Validate Dockerfiles with syntax and static analysis.

        Args:
            dockerfile_paths: List of paths to Dockerfiles

        Returns:
            List of validation issues from all Dockerfiles
        """
        all_issues = []
        for dockerfile_path in dockerfile_paths:
            # 7. Dockerfile syntax validation
            all_issues.extend(self._validate_dockerfile_syntax(dockerfile_path))

            # 8. Static analysis with Hadolint
            all_issues.extend(self._run_hadolint(dockerfile_path))

        return all_issues

    def validate_k8s_manifests(self, manifest_paths: List[str]) -> List[ValidationIssue]:
        """
        Validate Kubernetes manifests with syntax and static analysis.

        Args:
            manifest_paths: List of paths to Kubernetes manifest files

        Returns:
            List of validation issues
        """
        issues = []

        for manifest_path in manifest_paths:
            # 10. Kubernetes manifest syntax validation
            issues.extend(self._validate_k8s_syntax(manifest_path))

            # 11. Static analysis with Kube-linter
            issues.extend(self._run_kube_linter(manifest_path))

        return issues

    def _validate_dockerfile_syntax(self, dockerfile_path: str) -> List[ValidationIssue]:
        """
        7. Dockerfile syntax validation using docker build --check.
        """
        issues = []
        try:
            # Use repository manager to get absolute path
            dockerfile_full_path = self.repository_manager.get_full_path(dockerfile_path)

            if not dockerfile_full_path.exists():
                issues.append(ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Dockerfile not found at {dockerfile_path}",
                    rule_id="DOCKERFILE_NOT_FOUND",
                    category="file_missing"
                ))
                return issues

            # Use docker build --check to validate syntax without building
            # Run from repository directory for proper build context
            result = subprocess.run([
                'docker', 'build', '--check', '-f', str(dockerfile_full_path), '.'
            ], capture_output=True, text=True, timeout=30, cwd=str(dockerfile_full_path.parent))

            if result.returncode != 0:
                # Parse error from docker build output
                issues.append(ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Dockerfile syntax error: {result.stderr.strip()}",
                    rule_id="DOCKERFILE_SYNTAX",
                    category="syntax"
                ))
        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Dockerfile syntax validation timed out",
                rule_id="DOCKERFILE_TIMEOUT",
                category="syntax"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate Dockerfile syntax: {str(e)}",
                rule_id="DOCKERFILE_VALIDATION_ERROR",
                category="syntax"
            ))

        return issues

    def _run_hadolint(self, dockerfile_path: str) -> List[ValidationIssue]:
        """
        8. Static analysis with Hadolint.
        """
        issues = []
        try:
            # Use repository manager to get absolute path
            dockerfile_full_path = self.repository_manager.get_full_path(dockerfile_path)

            if not dockerfile_full_path.exists():
                issues.append(ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.WARNING,
                    message=f"Dockerfile not found for Hadolint analysis: {dockerfile_path}",
                    rule_id="DOCKERFILE_NOT_FOUND",
                    category="file_missing"
                ))
                return issues

            # Run hadolint with JSON output
            result = subprocess.run([
                'hadolint', '--format', 'json', str(dockerfile_full_path)
            ], capture_output=True, text=True, timeout=30)

            if result.stdout:
                # Parse hadolint JSON output
                hadolint_issues = json.loads(result.stdout)
                for issue in hadolint_issues:
                    severity = self._map_hadolint_severity(issue.get('level', 'error'))
                    issues.append(ValidationIssue(
                        file_path=dockerfile_path,
                        line_number=issue.get('line'),
                        severity=severity,
                        message=issue.get('message', ''),
                        rule_id=issue.get('code', 'HADOLINT'),
                        category="best_practices"
                    ))

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.WARNING,
                message="Hadolint analysis timed out",
                rule_id="HADOLINT_TIMEOUT",
                category="tool_error"
            ))
        except FileNotFoundError:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.WARNING,
                message="Hadolint not installed - static analysis skipped",
                rule_id="HADOLINT_NOT_FOUND",
                category="tool_error"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.WARNING,
                message=f"Hadolint analysis failed: {str(e)}",
                rule_id="HADOLINT_ERROR",
                category="tool_error"
            ))

        return issues

    def _validate_k8s_syntax(self, manifest_path: str) -> List[ValidationIssue]:
        """
        10. Kubernetes manifest syntax validation using YAML parser and basic structure checks.
        """
        issues = []
        try:
            # Use repository manager to get absolute path
            manifest_full_path = self.repository_manager.get_full_path(manifest_path)

            with open(manifest_full_path, 'r') as f:
                docs = list(yaml.safe_load_all(f))

            for i, doc in enumerate(docs):
                if doc is None:
                    continue

                # Check required Kubernetes fields
                if not isinstance(doc, dict):
                    issues.append(ValidationIssue(
                        file_path=manifest_path,
                        line_number=None,
                        severity=ValidationSeverity.ERROR,
                        message=f"Document {i+1} is not a valid Kubernetes object",
                        rule_id="K8S_INVALID_OBJECT",
                        category="syntax"
                    ))
                    continue

                # Check for required fields
                required_fields = ['apiVersion', 'kind']
                for field in required_fields:
                    if field not in doc:
                        issues.append(ValidationIssue(
                            file_path=manifest_path,
                            line_number=None,
                            severity=ValidationSeverity.ERROR,
                            message=f"Missing required field '{field}' in document {i+1}",
                            rule_id="K8S_MISSING_FIELD",
                            category="syntax"
                        ))

        except yaml.YAMLError as e:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=getattr(e, 'problem_mark', {}).get('line', None),
                severity=ValidationSeverity.ERROR,
                message=f"YAML syntax error: {str(e)}",
                rule_id="K8S_YAML_SYNTAX",
                category="syntax"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate Kubernetes manifest syntax: {str(e)}",
                rule_id="K8S_VALIDATION_ERROR",
                category="syntax"
            ))

        return issues

    def _run_kube_linter(self, manifest_path: str) -> List[ValidationIssue]:
        """
        11. Static analysis with Kube-linter.
        """
        issues = []
        try:
            # Use repository manager to get absolute path
            manifest_full_path = self.repository_manager.get_full_path(manifest_path)

            # Run kube-linter with JSON output
            result = subprocess.run([
                'kube-linter', 'lint', '--format', 'json', str(manifest_full_path)
            ], capture_output=True, text=True, timeout=30)

            if result.stdout:
                # Parse kube-linter JSON output
                output = json.loads(result.stdout)
                reports = output.get('Reports', [])

                for report in reports:
                    issues.append(ValidationIssue(
                        file_path=manifest_path,
                        line_number=None,  # kube-linter doesn't provide line numbers
                        severity=self._map_kube_linter_severity(report.get('Level', 'warning')),
                        message=report.get('Message', ''),
                        rule_id=report.get('Check', 'KUBE_LINTER'),
                        category="best_practices"
                    ))

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.WARNING,
                message="Kube-linter analysis timed out",
                rule_id="KUBE_LINTER_TIMEOUT",
                category="tool_error"
            ))
        except FileNotFoundError:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.WARNING,
                message="Kube-linter not installed - static analysis skipped",
                rule_id="KUBE_LINTER_NOT_FOUND",
                category="tool_error"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.WARNING,
                message=f"Kube-linter analysis failed: {str(e)}",
                rule_id="KUBE_LINTER_ERROR",
                category="tool_error"
            ))

        return issues

    def _map_hadolint_severity(self, hadolint_level: str) -> ValidationSeverity:
        """Map Hadolint severity levels to ValidationSeverity."""
        mapping = {
            'error': ValidationSeverity.ERROR,
            'warning': ValidationSeverity.WARNING,
            'info': ValidationSeverity.INFO,
            'style': ValidationSeverity.INFO
        }
        return mapping.get(hadolint_level.lower(), ValidationSeverity.WARNING)

    def _map_kube_linter_severity(self, kube_linter_level: str) -> ValidationSeverity:
        """Map Kube-linter severity levels to ValidationSeverity."""
        mapping = {
            'error': ValidationSeverity.ERROR,
            'warning': ValidationSeverity.WARNING,
            'info': ValidationSeverity.INFO
        }
        return mapping.get(kube_linter_level.lower(), ValidationSeverity.WARNING)

    # TODO: Implement remaining validation tasks

    def build_docker_images(self, dockerfile_paths: List[str]) -> List[ValidationIssue]:
        """
        9. Docker image building validation.

        TODO: Implement Docker build testing:
        - For each Dockerfile, attempt to build the image
        - Use docker build command with temporary tags
        - Capture build output and errors
        - Clean up created images after validation
        - Report build failures as validation issues
        """
        self.logger.info("TODO: Implement Docker image building validation")
        return []

    def apply_k8s_manifests_in_kind(self, manifest_paths: List[str]) -> List[ValidationIssue]:
        """
        12. Kubernetes manifest application in Kind environment.

        TODO: Implement Kind cluster testing:
        - Ensure Kind cluster is running
        - Apply manifests using kubectl apply
        - Check for application errors
        - Verify resources are created successfully
        - Clean up applied resources after validation
        - Report application failures as validation issues
        """
        self.logger.info("TODO: Implement Kind cluster manifest application")
        return []

    def validate_runtime_availability(self, manifest_paths: List[str]) -> List[ValidationIssue]:
        """
        13. Runtime validation - application availability through load balancer.

        TODO: Implement runtime availability checks:
        - Wait for deployments to be ready
        - Find services and ingresses with external access
        - Test HTTP endpoints for connectivity
        - Verify application responds correctly
        - Check health check endpoints if available
        - Report connectivity failures as validation issues
        """
        self.logger.info("TODO: Implement runtime availability validation")
        return []