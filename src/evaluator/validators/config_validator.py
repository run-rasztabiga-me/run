import json
import logging
import subprocess
import yaml
import re
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from ..core.models import ValidationIssue, ValidationSeverity, DockerBuildMetrics
from ...generator.core.repository import RepositoryManager
from ...generator.core.config import GeneratorConfig
from ...common.models import DockerImageInfo


class ConfigurationValidator:
    """Validates generated Docker and Kubernetes configurations using external linters and build tests.

    Implements validation tasks:
    7. Dockerfile syntax validation
    8. Static analysis with Hadolint
    9. Docker image building
    10. Kubernetes manifest syntax validation
    11. Static analysis with Kube-linter
    12. Kubernetes manifest application in K8S cluster
    13. Runtime validation - application availability (TODO)
    """

    def __init__(self, repository_manager: RepositoryManager, config: GeneratorConfig = None):
        self.logger = logging.getLogger(__name__)
        self.repository_manager = repository_manager
        self.config = config or GeneratorConfig()

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

    def build_docker_images(self, docker_images: List[DockerImageInfo], repo_name: str) -> Tuple[List[ValidationIssue], List[DockerBuildMetrics]]:
        """
        9. Docker image building validation.

        Builds Docker images from Dockerfiles and pushes them to the configured private registry.

        Args:
            docker_images: List of DockerImageInfo objects with build metadata
            repo_name: Name of the repository being validated

        Returns:
            Tuple of (validation issues, build metrics)
        """
        issues = []
        build_metrics = []

        for image_info in docker_images:
            self.logger.info(f"Building Docker image: {image_info.image_tag} from {image_info.dockerfile_path}")

            # Generate full image name with registry
            full_image_name = self.config.get_full_image_name(repo_name, image_info.image_tag)

            # Build and push the image using buildx with insecure registry support
            build_push_issues, metrics = self._build_and_push_image(
                dockerfile_path=image_info.dockerfile_path,
                build_context=image_info.build_context,
                image_name=full_image_name,
                image_tag=image_info.image_tag
            )
            issues.extend(build_push_issues)
            if metrics:
                build_metrics.append(metrics)

        return issues, build_metrics

    def apply_k8s_manifests(self, manifest_paths: List[str], repo_name: str, docker_images: List[DockerImageInfo] = None) -> List[ValidationIssue]:
        """
        12. Kubernetes manifest application.

        Applies K8s manifests to configured cluster in a dedicated namespace.

        Args:
            manifest_paths: List of paths to Kubernetes manifest files
            repo_name: Repository name (used for namespace creation)
            docker_images: List of Docker images generated by LLM (for patching image names)

        Returns:
            List of validation issues encountered during application
        """
        if docker_images is None:
            docker_images = []
        issues = []

        # Generate namespace name from repo_name (sanitize for K8s naming)
        namespace = self._sanitize_namespace_name(repo_name)

        try:
            # 1. Delete old namespace if exists (cleanup previous runs)
            self._delete_namespace(namespace)

            # 2. Create new namespace
            create_issues = self._create_namespace(namespace)
            issues.extend(create_issues)

            if any(issue.severity == ValidationSeverity.ERROR for issue in create_issues):
                return issues  # Can't proceed without namespace

            # 3. Apply each manifest to the namespace
            applied_resources = []
            for manifest_path in manifest_paths:
                apply_issues, resources = self._apply_manifest(manifest_path, namespace, repo_name, docker_images)
                issues.extend(apply_issues)
                applied_resources.extend(resources)

            # 4. Wait for resources to be ready
            if applied_resources:
                ready_issues = self._wait_for_resources_ready(applied_resources, namespace)
                issues.extend(ready_issues)

        except Exception as e:
            issues.append(ValidationIssue(
                file_path="cluster",
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to apply manifests: {str(e)}",
                rule_id="K8S_APPLY_ERROR"
            ))

        return issues

    def get_ingress_url(self, namespace: str) -> Optional[str]:
        """
        Get the ingress URL from the Kubernetes cluster.

        Args:
            namespace: Kubernetes namespace where resources are deployed

        Returns:
            Ingress URL if found, None otherwise
        """
        try:
            # Get all ingresses in the namespace
            result = subprocess.run([
                'kubectl', 'get', 'ingress',
                '-n', namespace,
                '-o', 'json'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                self.logger.warning(f"Failed to get ingresses from cluster: {result.stderr}")
                return None

            import json
            ingresses_data = json.loads(result.stdout)
            ingresses = ingresses_data.get('items', [])

            if not ingresses:
                self.logger.warning(f"No ingresses found in namespace {namespace}")
                return None

            # Take the first ingress
            ingress = ingresses[0]

            # Extract host from ingress spec
            rules = ingress.get('spec', {}).get('rules', [])
            if rules and len(rules) > 0:
                host = rules[0].get('host')
                if host:
                    # Check if TLS is configured
                    tls = ingress.get('spec', {}).get('tls', [])
                    protocol = 'https' if tls else 'http'

                    ingress_name = ingress.get('metadata', {}).get('name', 'unknown')
                    self.logger.info(f"Found ingress '{ingress_name}' with host: {host}")

                    return f"{protocol}://{host}"

        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout while getting ingress from cluster")
        except Exception as e:
            self.logger.warning(f"Failed to get ingress URL from cluster: {str(e)}")

        return None

    def check_endpoint_health(self, base_url: str, endpoint_path: str, timeout: int = 30, retries: int = 5) -> List[ValidationIssue]:
        """
        Check if an application endpoint is accessible and returns 2xx status.

        Args:
            base_url: Base URL of the application (e.g., "http://app.example.com")
            endpoint_path: Relative path to test (e.g., "/health", "/")
            timeout: Timeout for each request in seconds
            retries: Number of retry attempts with exponential backoff

        Returns:
            List of validation issues if endpoint is not accessible
        """
        import requests
        from requests.exceptions import RequestException

        issues = []

        # Ensure endpoint_path starts with /
        if not endpoint_path.startswith('/'):
            endpoint_path = '/' + endpoint_path

        full_url = f"{base_url}{endpoint_path}"
        self.logger.info(f"Testing endpoint: {full_url}")

        last_error = None
        for attempt in range(retries):
            try:
                # Wait with exponential backoff (except first attempt)
                if attempt > 0:
                    wait_time = 2 ** attempt  # 2, 4, 8, 16 seconds
                    self.logger.info(f"Retry {attempt + 1}/{retries} after {wait_time}s...")
                    time.sleep(wait_time)

                response = requests.get(full_url, timeout=timeout, verify=False)

                if 200 <= response.status_code < 300:
                    self.logger.info(f"Endpoint {full_url} is healthy (status {response.status_code})")
                    return []  # Success!
                else:
                    last_error = f"Endpoint returned non-2xx status: {response.status_code}"
                    self.logger.warning(f"Attempt {attempt + 1}: {last_error}")

            except RequestException as e:
                last_error = str(e)
                self.logger.warning(f"Attempt {attempt + 1}: Connection failed - {last_error}")

        # All retries failed
        issues.append(ValidationIssue(
            file_path="runtime",
            line_number=None,
            severity=ValidationSeverity.ERROR,
            message=f"Endpoint {full_url} not accessible after {retries} attempts: {last_error}",
            rule_id="ENDPOINT_HEALTH_CHECK_FAILED"
        ))

        return issues

    def _ensure_hosts_entry(self, hostname: str) -> None:
        """
        Ensure entry exists in /etc/hosts for hostname resolution.
        Does not add duplicate entries if hostname already exists.

        Args:
            hostname: Hostname to ensure (e.g., "poc2-fastapi.rasztabiga.me")
        """
        ip_address = self.config.k8s_cluster_ip
        entry = f"{ip_address}   {hostname}"

        try:
            # Check if entry already exists
            with open('/etc/hosts', 'r') as f:
                content = f.read()
                # Check if hostname is already mapped (anywhere in the file)
                for line in content.splitlines():
                    if hostname in line and not line.strip().startswith('#'):
                        self.logger.info(f"Entry for {hostname} already exists in /etc/hosts")
                        return

            # Add entry using sudo (requires passwordless sudo or user interaction)
            self.logger.info(f"Adding /etc/hosts entry: {entry}")
            result = subprocess.run([
                'sudo', 'sh', '-c', f'echo "{entry}" >> /etc/hosts'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                self.logger.warning(f"Failed to add /etc/hosts entry: {result.stderr}")
            else:
                self.logger.info(f"Successfully added /etc/hosts entry for {hostname}")

        except Exception as e:
            self.logger.warning(f"Error ensuring /etc/hosts entry: {str(e)}")

    def validate_runtime_availability(self, manifest_paths: List[str], namespace: str, test_endpoint: str) -> List[ValidationIssue]:
        """
        13. Runtime validation - application availability through ingress.

        Args:
            manifest_paths: List of paths to Kubernetes manifest files
            namespace: Kubernetes namespace where resources are deployed
            test_endpoint: Relative path to test endpoint (e.g., "/health")

        Returns:
            List of validation issues if application is not accessible
        """
        issues = []

        # Get ingress URL
        ingress_url = self.get_ingress_url(namespace)

        if not ingress_url:
            issues.append(ValidationIssue(
                file_path="runtime",
                line_number=None,
                severity=ValidationSeverity.WARNING,
                message="No ingress URL found in cluster, skipping runtime health check",
                rule_id="NO_INGRESS_URL"
            ))
            return issues

        self.logger.info(f"Found ingress URL: {ingress_url}")

        # Extract hostname from URL for /etc/hosts entry
        from urllib.parse import urlparse
        parsed_url = urlparse(ingress_url)
        hostname = parsed_url.hostname

        if not hostname:
            issues.append(ValidationIssue(
                file_path="runtime",
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to extract hostname from ingress URL: {ingress_url}",
                rule_id="INVALID_INGRESS_URL"
            ))
            return issues

        # Ensure /etc/hosts entry exists for DNS resolution
        self._ensure_hosts_entry(hostname)

        # Wait a bit for ingress to be fully configured
        self.logger.info("Waiting for ingress to be fully configured...")
        time.sleep(self.config.k8s_ingress_timeout)

        # Check endpoint health
        health_issues = self.check_endpoint_health(ingress_url, test_endpoint)
        issues.extend(health_issues)

        return issues

    def _build_and_push_image(self, dockerfile_path: str, build_context: str, image_name: str, image_tag: str) -> Tuple[List[ValidationIssue], Optional[DockerBuildMetrics]]:
        """
        Build and push a Docker image using buildx with insecure registry support.

        Args:
            dockerfile_path: Path to Dockerfile relative to repository root
            build_context: Build context path relative to repository root
            image_name: Full image name including registry and tag
            image_tag: Image tag (for metrics tracking)

        Returns:
            Tuple of (validation issues, build metrics)
        """
        issues = []
        metrics = None
        try:
            # Get absolute paths
            dockerfile_full_path = self.repository_manager.get_full_path(dockerfile_path)
            context_full_path = self.repository_manager.get_full_path(build_context)

            if not dockerfile_full_path.exists():
                issues.append(ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Dockerfile not found at {dockerfile_path}",
                    rule_id="DOCKER_BUILD_FILE_NOT_FOUND"
                ))
                return issues, None

            if not context_full_path.exists():
                issues.append(ValidationIssue(
                    file_path=build_context,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Build context directory not found at {build_context}",
                    rule_id="DOCKER_BUILD_CONTEXT_NOT_FOUND"
                ))
                return issues, None

            # Build the image using buildx for linux/amd64 platform
            # Then push separately using regular docker push (which supports insecure registries)
            self.logger.info(f"Building Docker image with buildx: {image_name}")

            # Start timing
            build_start = time.time()

            # Step 1: Build with buildx (load into local docker)
            result = subprocess.run([
                'docker', 'buildx', 'build',
                '--platform', 'linux/amd64',
                '--no-cache',
                '--load',  # Load into local docker instead of pushing
                '--progress=plain',
                '-t', image_name,
                '-f', str(dockerfile_full_path),
                str(context_full_path)
            ], capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                issues.append(ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Docker buildx build failed: {error_msg}",
                    rule_id="DOCKER_BUILDX_FAILED"
                ))
                return issues, None

            self.logger.info(f"✓ Build completed successfully for image: {image_name}")

            # Log build output for debugging
            if result.stdout:
                self.logger.debug(f"Build output:\n{result.stdout}")
            if result.stderr:
                self.logger.debug(f"Build stderr:\n{result.stderr}")

            # Step 2: Push to registry using regular docker push (supports insecure registries)
            self.logger.info(f"Pushing image to registry: {image_name}")
            push_result = subprocess.run([
                'docker', 'push', image_name
            ], capture_output=True, text=True, timeout=300)

            # Calculate build time (includes push)
            build_time = time.time() - build_start

            if push_result.returncode != 0:
                error_msg = push_result.stderr.strip() if push_result.stderr else push_result.stdout.strip()
                issues.append(ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Docker push failed: {error_msg}",
                    rule_id="DOCKER_PUSH_FAILED"
                ))
                return issues, None

            self.logger.info(f"✓ Successfully pushed image to registry: {image_name}")

            # Step 3: Verify the image exists in the registry using HTTP API
            # Extract registry address and image path from full image name
            # Format: registry/repo-name-tag:version
            if '/' in image_name:
                registry_and_image = image_name.split('/', 1)
                registry = registry_and_image[0]
                image_path = registry_and_image[1]

                # Split image path into name and tag
                if ':' in image_path:
                    image_repo, image_tag_version = image_path.rsplit(':', 1)
                else:
                    image_repo = image_path
                    image_tag_version = 'latest'

                # Verify using HTTP registry API
                # Give registry a moment to index the pushed image
                time.sleep(1)

                import requests
                verify_url = f"http://{registry}/v2/{image_repo}/manifests/{image_tag_version}"
                self.logger.info(f"Verifying image in registry: {image_name}")
                self.logger.debug(f"Verification URL: {verify_url}")

                # Retry verification a few times in case registry is still indexing
                verified = False
                last_error = None
                for attempt in range(3):
                    try:
                        self.logger.debug(f"Verification attempt {attempt + 1}/3: HEAD {verify_url}")
                        verify_response = requests.head(verify_url, timeout=10)
                        if verify_response.status_code == 200:
                            self.logger.info(f"✓ Successfully verified image in registry: {image_name}")
                            verified = True
                            break
                        else:
                            last_error = f"registry returned status {verify_response.status_code}"
                            self.logger.warning(f"Verification attempt {attempt + 1}/3 failed: {last_error}")
                            if attempt < 2:  # Don't sleep after last attempt
                                time.sleep(1)
                    except Exception as e:
                        last_error = str(e)
                        self.logger.warning(f"Verification attempt {attempt + 1}/3 failed: {last_error}")
                        if attempt < 2:  # Don't sleep after last attempt
                            time.sleep(1)

                if not verified:
                    issues.append(ValidationIssue(
                        file_path=dockerfile_path,
                        line_number=None,
                        severity=ValidationSeverity.ERROR,
                        message=f"Image pushed but verification failed after 3 attempts (URL: {verify_url}): {last_error}",
                        rule_id="DOCKER_PUSH_VERIFICATION_FAILED"
                    ))
                    return issues, None

            # Get image size and layers using docker inspect
            try:
                # Get size
                size_result = subprocess.run([
                    'docker', 'image', 'inspect', image_name, '--format', '{{.Size}}'
                ], capture_output=True, text=True, timeout=10)

                # Get layers count
                layers_result = subprocess.run([
                    'docker', 'image', 'inspect', image_name, '--format', '{{len .RootFS.Layers}}'
                ], capture_output=True, text=True, timeout=10)

                if size_result.returncode == 0 and layers_result.returncode == 0:
                    image_size_bytes = int(size_result.stdout.strip())
                    image_size_mb = image_size_bytes / (1024 * 1024)
                    layers_count = int(layers_result.stdout.strip())

                    metrics = DockerBuildMetrics(
                        image_tag=image_tag,
                        build_time=build_time,
                        image_size_mb=round(image_size_mb, 2),
                        layers_count=layers_count
                    )
                    self.logger.info(f"Image metrics: {image_size_mb:.2f} MB, {layers_count} layers, built in {build_time:.1f}s")
                else:
                    self.logger.warning(f"Could not inspect image metrics for {image_name}")
            except Exception as e:
                self.logger.warning(f"Failed to collect image metrics: {str(e)}")

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Docker buildx build/push timed out (>10 minutes)",
                rule_id="DOCKER_BUILDX_TIMEOUT"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Docker buildx error: {str(e)}",
                rule_id="DOCKER_BUILDX_ERROR"
            ))

        return issues, metrics

    def _sanitize_namespace_name(self, repo_name: str) -> str:
        """
        Sanitize repository name for use as Kubernetes namespace.

        K8s namespace rules:
        - Must be DNS label (RFC 1123)
        - Lowercase alphanumeric + hyphens
        - Max 63 chars
        - Start/end with alphanumeric

        Args:
            repo_name: Repository name to sanitize

        Returns:
            Sanitized namespace name
        """
        # Convert to lowercase
        namespace = repo_name.lower()

        # Replace invalid characters with hyphens
        namespace = re.sub(r'[^a-z0-9-]', '-', namespace)

        # Remove leading/trailing hyphens
        namespace = namespace.strip('-')

        # Truncate to 63 chars
        namespace = namespace[:63]

        # Ensure it ends with alphanumeric (remove trailing hyphens after truncation)
        namespace = namespace.rstrip('-')

        return namespace

    def _delete_namespace(self, namespace: str) -> None:
        """
        Delete namespace if it exists (cleanup from previous runs).

        Uses --ignore-not-found so it doesn't fail if namespace doesn't exist.
        Also waits for namespace to be fully deleted before returning.

        Args:
            namespace: Namespace name to delete
        """
        self.logger.info(f"Cleaning up old namespace: {namespace}")

        try:
            # Delete namespace (ignore if doesn't exist)
            subprocess.run([
                'kubectl', 'delete', 'namespace', namespace, '--ignore-not-found=true'
            ], capture_output=True, text=True, timeout=30)

            # Wait for namespace to be fully deleted (max 60s)
            # This is important - if we create namespace too quickly, it might still be terminating
            subprocess.run([
                'kubectl', 'wait', '--for=delete', f'namespace/{namespace}',
                '--timeout=60s'
            ], capture_output=True, text=True, timeout=70)

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout waiting for namespace {namespace} deletion")
        except Exception as e:
            self.logger.warning(f"Error deleting namespace {namespace}: {str(e)}")

    def _create_namespace(self, namespace: str) -> List[ValidationIssue]:
        """
        Create a new Kubernetes namespace.

        Args:
            namespace: Namespace name to create

        Returns:
            List of validation issues if creation failed
        """
        issues = []

        self.logger.info(f"Creating namespace: {namespace}")

        try:
            result = subprocess.run([
                'kubectl', 'create', 'namespace', namespace
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                issues.append(ValidationIssue(
                    file_path=namespace,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to create namespace: {result.stderr}",
                    rule_id="K8S_NAMESPACE_CREATE_FAILED"
                ))
            else:
                self.logger.info(f"Successfully created namespace: {namespace}")

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=namespace,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Namespace creation timed out",
                rule_id="K8S_NAMESPACE_TIMEOUT"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=namespace,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Error creating namespace: {str(e)}",
                rule_id="K8S_NAMESPACE_ERROR"
            ))

        return issues

    def _extract_resource_names(self, manifest_path: Path) -> List[Dict[str, str]]:
        """
        Extract resource kind and name from manifest YAML for status checking.

        Args:
            manifest_path: Path to manifest file

        Returns:
            List of dicts: [{"kind": "Deployment", "name": "app-name"}, ...]
        """
        resources = []

        try:
            with open(manifest_path, 'r') as f:
                # YAML file can contain multiple documents
                docs = list(yaml.safe_load_all(f))

            for doc in docs:
                if doc and isinstance(doc, dict):
                    kind = doc.get('kind')
                    name = doc.get('metadata', {}).get('name')

                    if kind and name:
                        resources.append({
                            'kind': kind,
                            'name': name
                        })

        except Exception as e:
            self.logger.warning(f"Failed to extract resource names from {manifest_path}: {str(e)}")

        return resources

    def _patch_image_names(self, manifest_path: Path, repo_name: str, docker_images: List[DockerImageInfo]) -> Path:
        """
        Patch image names in manifest to use full registry path.

        Transforms short image names (e.g., 'poc1-fastapi:latest') to full registry paths
        (e.g., '192.168.0.124:32000/poc1-fastapi-backend:latest').

        Only patches images that were generated by LLM (present in docker_images list).
        Official images (postgres, redis, etc.) are left unchanged.

        Args:
            manifest_path: Path to original manifest file
            repo_name: Repository name for building full image names
            docker_images: List of Docker images generated by LLM (only these will be patched)

        Returns:
            Path to patched manifest file (temporary file)
        """

        # Build set of image tags that should be patched (from LLM output)
        llm_image_tags = {img.image_tag for img in docker_images}

        try:
            # Load manifest documents
            with open(manifest_path, 'r') as f:
                docs = list(yaml.safe_load_all(f))

            # Patch each document
            for doc in docs:
                if not doc or not isinstance(doc, dict):
                    continue

                kind = doc.get('kind')
                if kind not in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
                    continue

                # Get containers from the appropriate path based on kind
                if kind in ['Deployment', 'StatefulSet', 'DaemonSet']:
                    containers = doc.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
                elif kind == 'Job':
                    containers = doc.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
                elif kind == 'CronJob':
                    containers = doc.get('spec', {}).get('jobTemplate', {}).get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
                else:
                    containers = []

                # Patch each container's image
                for container in containers:
                    if 'image' not in container:
                        continue

                    image = container['image']

                    # Check if image already has registry prefix
                    if '/' in image and image.startswith(self.config.docker_registry):
                        # Already has correct registry, skip
                        self.logger.debug(f"Image already has registry prefix: {image}")
                        continue

                    # Extract image tag (part before ':' or whole string if no ':')
                    if ':' in image:
                        image_tag, version = image.split(':', 1)
                    else:
                        image_tag = image
                        version = self.config.default_image_tag

                    # Remove any registry prefix from image_tag
                    if '/' in image_tag:
                        image_tag = image_tag.split('/')[-1]

                    # Only patch if this image was generated by LLM
                    if image_tag not in llm_image_tags:
                        self.logger.debug(f"Skipping official/external image: {image} (not in LLM output)")
                        continue

                    # Build full image name
                    full_image = self.config.get_full_image_name(repo_name, image_tag, version)

                    self.logger.info(f"Patching image name: {image} → {full_image}")
                    container['image'] = full_image

            # Write patched manifest to temporary file
            temp_path = manifest_path.with_suffix('.patched.yaml')
            with open(temp_path, 'w') as f:
                yaml.safe_dump_all(docs, f)

            self.logger.debug(f"Created patched manifest: {temp_path}")
            return temp_path

        except Exception as e:
            self.logger.error(f"Failed to patch image names in {manifest_path}: {str(e)}")
            # Return original path if patching fails
            return manifest_path

    def _apply_manifest(self, manifest_path: str, namespace: str, repo_name: str, docker_images: List[DockerImageInfo]) -> Tuple[List[ValidationIssue], List[Dict[str, str]]]:
        """
        Apply a single manifest file to cluster in specified namespace.

        Automatically patches image names to use full registry path before applying.

        Args:
            manifest_path: Path to manifest file relative to repository root
            namespace: Kubernetes namespace to apply to
            repo_name: Repository name for patching image names
            docker_images: List of Docker images generated by LLM (for selective patching)

        Returns:
            Tuple of (issues, applied_resources)
            - issues: List of ValidationIssues
            - applied_resources: List of dicts with resource info for status checking
        """
        issues = []
        applied_resources = []
        patched_manifest_path = None

        try:
            manifest_full_path = self.repository_manager.get_full_path(manifest_path)

            if not manifest_full_path.exists():
                issues.append(ValidationIssue(
                    file_path=manifest_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Manifest file not found: {manifest_path}",
                    rule_id="K8S_MANIFEST_NOT_FOUND"
                ))
                return issues, applied_resources

            # Patch image names to use full registry path
            patched_manifest_path = self._patch_image_names(manifest_full_path, repo_name, docker_images)

            # Apply patched manifest to namespace
            self.logger.info(f"Applying manifest {manifest_path} to namespace {namespace}")
            result = subprocess.run([
                'kubectl', 'apply', '-f', str(patched_manifest_path), '-n', namespace
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                issues.append(ValidationIssue(
                    file_path=manifest_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"kubectl apply failed: {result.stderr.strip()}",
                    rule_id="K8S_APPLY_FAILED"
                ))
            else:
                self.logger.info(f"Successfully applied {manifest_path}")

                # Extract resource names for status checking (from original manifest)
                applied_resources = self._extract_resource_names(manifest_full_path)

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="kubectl apply timed out",
                rule_id="K8S_APPLY_TIMEOUT"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Error applying manifest: {str(e)}",
                rule_id="K8S_APPLY_ERROR"
            ))
        finally:
            # Cleanup patched manifest file
            if patched_manifest_path and patched_manifest_path != manifest_full_path:
                try:
                    patched_manifest_path.unlink(missing_ok=True)
                    self.logger.debug(f"Cleaned up patched manifest: {patched_manifest_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup patched manifest {patched_manifest_path}: {str(e)}")

        return issues, applied_resources

    def _wait_for_resources_ready(self, resources: List[Dict[str, str]], namespace: str, timeout: int = 120) -> List[ValidationIssue]:
        """
        Wait for deployments/statefulsets to be ready.

        Args:
            resources: List of resource dicts with 'kind' and 'name'
            namespace: Kubernetes namespace
            timeout: Timeout in seconds (default 120s = 2min)

        Returns:
            List of validation issues for resources that didn't become ready
        """
        issues = []

        for resource in resources:
            kind = resource['kind']
            name = resource['name']

            # Only wait for Deployments and StatefulSets (they have rollout status)
            if kind not in ['Deployment', 'StatefulSet']:
                continue

            self.logger.info(f"Waiting for {kind}/{name} to be ready in namespace {namespace}")

            try:
                # kubectl wait --for=condition=available deployment/name -n namespace --timeout=300s
                # For StatefulSets, use JSONPath since they don't have a 'ready' condition
                if kind == 'Deployment':
                    result = subprocess.run([
                        'kubectl', 'wait',
                        f'--for=condition=available',
                        f'{kind.lower()}/{name}',
                        '-n', namespace,
                        f'--timeout={timeout}s'
                    ], capture_output=True, text=True, timeout=timeout + 10)
                elif kind == 'StatefulSet':
                    # First get the desired replica count
                    get_replicas_result = subprocess.run([
                        'kubectl', 'get', f'{kind.lower()}/{name}',
                        '-n', namespace,
                        '-o', 'jsonpath={.spec.replicas}'
                    ], capture_output=True, text=True, timeout=10)

                    if get_replicas_result.returncode != 0:
                        raise Exception(f"Failed to get replica count: {get_replicas_result.stderr}")

                    replica_count = get_replicas_result.stdout.strip() or '1'

                    # Wait for readyReplicas to match desired count
                    result = subprocess.run([
                        'kubectl', 'wait',
                        f'--for=jsonpath={{.status.readyReplicas}}={replica_count}',
                        f'{kind.lower()}/{name}',
                        '-n', namespace,
                        f'--timeout={timeout}s'
                    ], capture_output=True, text=True, timeout=timeout + 10)
                else:
                    # Skip other resource types
                    continue

                if result.returncode != 0:
                    issues.append(ValidationIssue(
                        file_path=name,
                        line_number=None,
                        severity=ValidationSeverity.ERROR,
                        message=f"{kind} {name} not ready within {timeout}s: {result.stderr.strip()}",
                        rule_id="K8S_RESOURCE_NOT_READY"
                    ))
                else:
                    self.logger.info(f"{kind}/{name} is ready")

            except subprocess.TimeoutExpired:
                issues.append(ValidationIssue(
                    file_path=name,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"{kind} {name} readiness check timed out",
                    rule_id="K8S_READY_TIMEOUT"
                ))
            except Exception as e:
                issues.append(ValidationIssue(
                    file_path=name,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Error checking {kind} {name} status: {str(e)}",
                    rule_id="K8S_READY_CHECK_ERROR"
                ))

        return issues
