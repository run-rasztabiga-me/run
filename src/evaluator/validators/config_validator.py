import json
import logging
import subprocess
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
                    rule_id="DOCKER_NOT_FOUND"
                ))
                return issues

            # Use docker build --check to validate syntax without building
            # Run from repository directory for proper build context
            result = subprocess.run([
                'docker', 'build', '--check', '-f', str(dockerfile_full_path), '.'
            ], capture_output=True, text=True, timeout=30, cwd=str(dockerfile_full_path.parent))

            # Parse docker build --check output from stdout
            if result.stdout:
                issues.extend(self._parse_docker_check_output(result.stdout, dockerfile_path))

            # If return code is non-zero and no issues were parsed, add a generic error
            if result.returncode != 0 and not issues:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                issues.append(ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Dockerfile validation failed: {error_msg}",
                    rule_id="DOCKER_VALIDATION_FAILED"
                ))
        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Dockerfile syntax validation timed out",
                rule_id="DOCKER_TIMEOUT"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate Dockerfile syntax: {str(e)}",
                rule_id="DOCKER_ERROR"
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
                    severity=ValidationSeverity.ERROR,
                    message=f"Dockerfile not found for Hadolint analysis: {dockerfile_path}",
                    rule_id="HADOLINT_FILE_NOT_FOUND"
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
                        rule_id=issue.get('code', 'HADOLINT')
                    ))

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Hadolint analysis timed out",
                rule_id="HADOLINT_TIMEOUT"
            ))
        except FileNotFoundError:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Hadolint not installed - static analysis skipped",
                rule_id="HADOLINT_NOT_FOUND"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Hadolint analysis failed: {str(e)}",
                rule_id="HADOLINT_ERROR"
            ))

        return issues

    def _validate_k8s_syntax(self, manifest_path: str) -> List[ValidationIssue]:
        """
        10. Kubernetes manifest syntax validation using kubectl dry-run.

        Uses kubectl apply --dry-run=server for proper schema validation against Kubernetes API.
        """
        issues = []
        try:
            # Use repository manager to get absolute path
            manifest_full_path = self.repository_manager.get_full_path(manifest_path)

            # Use kubectl dry-run for validation
            result = subprocess.run([
                'kubectl', 'apply', '--dry-run=server', '-f', str(manifest_full_path)
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                # Parse kubectl error output
                error_msg = result.stderr.strip()
                issues.append(ValidationIssue(
                    file_path=manifest_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Kubernetes validation failed: {error_msg}",
                    rule_id="KUBECTL_VALIDATION_FAILED"
                ))

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="kubectl validation timed out",
                rule_id="KUBECTL_TIMEOUT"
            ))
        except FileNotFoundError:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="kubectl not installed - Kubernetes validation skipped",
                rule_id="KUBECTL_NOT_FOUND"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate Kubernetes manifest: {str(e)}",
                rule_id="KUBECTL_ERROR"
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

            # Get config file path using importlib.resources
            # The config file is located in the same package as this module
            config_path = Path(__file__).parent / '.kube-linter.yaml'

            # Run kube-linter with JSON output and config file if it exists
            cmd = ['kube-linter', 'lint', '--format', 'json']
            if config_path.exists():
                cmd.extend(['--config', str(config_path)])
            cmd.append(str(manifest_full_path))

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.stdout:
                # Parse kube-linter JSON output
                output = json.loads(result.stdout)
                reports = output.get('Reports') or []

                for report in reports:
                    # Extract message from Diagnostic.Message
                    diagnostic = report.get('Diagnostic', {})
                    message = diagnostic.get('Message', '')

                    issues.append(ValidationIssue(
                        file_path=manifest_path,
                        line_number=None,  # kube-linter doesn't provide line numbers
                        severity=ValidationSeverity.WARNING,  # kube-linter reports are typically warnings
                        message=message,
                        rule_id=report.get('Check', 'KUBE_LINTER')
                    ))

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Kube-linter analysis timed out",
                rule_id="KUBE_LINTER_TIMEOUT"
            ))
        except FileNotFoundError:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Kube-linter not installed - static analysis skipped",
                rule_id="KUBE_LINTER_NOT_FOUND"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Kube-linter analysis failed: {str(e)}",
                rule_id="KUBE_LINTER_ERROR"
            ))

        return issues

    def _parse_docker_check_output(self, output: str, dockerfile_path: str) -> List[ValidationIssue]:
        """Parse docker build --check output to extract warnings and errors."""
        issues = []
        lines = output.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for WARNING or ERROR lines
            if line.startswith('WARNING:') or line.startswith('ERROR:'):
                severity = ValidationSeverity.WARNING if line.startswith('WARNING:') else ValidationSeverity.ERROR

                # Extract rule ID and description
                parts = line.split(' - ', 1)
                if len(parts) >= 2:
                    rule_info = parts[0].split(':', 1)[1].strip()  # Get part after WARNING:/ERROR:
                    description = parts[1] if len(parts) > 1 else ""
                else:
                    rule_info = line.split(':', 1)[1].strip() if ':' in line else "DOCKER_CHECK"
                    description = ""

                # Try to extract line number from the next lines (look for Dockerfile:N)
                line_number = None
                message_parts = [description] if description else []

                # Read next few lines to get more context
                j = i + 1
                while j < len(lines) and j < i + 5:
                    next_line = lines[j].strip()
                    if next_line.startswith('Dockerfile:'):
                        try:
                            line_number = int(next_line.split(':')[1])
                        except (IndexError, ValueError):
                            pass
                    elif next_line and not next_line.startswith('---'):
                        message_parts.append(next_line)
                    j += 1

                message = ' '.join(message_parts) if message_parts else rule_info

                issues.append(ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=line_number,
                    severity=severity,
                    message=message.strip(),
                    rule_id=rule_info.split()[0] if rule_info else "DOCKER_CHECK"
                ))

                i = j  # Skip the lines we've already processed
            else:
                i += 1

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
