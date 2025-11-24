"""Docker build and push helpers reusable by runtime workflows."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

from src.common.command_runner import CommandResult, CommandRunner
from src.common.models import DockerBuildMetrics, DockerImageInfo
from src.generator.core.workspace import RepositoryWorkspace

from .issues import RuntimeIssue


@dataclass(slots=True)
class DockerBuildResult:
    """Aggregate result of building and pushing a Docker image."""

    image_name: str
    issues: List[RuntimeIssue]
    metrics: Optional[DockerBuildMetrics] = None

    @property
    def success(self) -> bool:
        return not any(issue.is_error() for issue in self.issues)


class DockerImageBuilder:
    """Build Docker images and push them to a registry."""

    def __init__(self, command_runner: CommandRunner, logger: Optional[logging.Logger] = None) -> None:
        self.command_runner = command_runner
        self.logger = logger or logging.getLogger(__name__)

    def build_and_push(
        self,
        *,
        workspace: RepositoryWorkspace,
        image: DockerImageInfo,
        image_name: str,
        verify_registry: bool = True,
        max_push_retries: int = 3,
    ) -> DockerBuildResult:
        """
        Build a Docker image from the provided Dockerfile and push it to the registry.

        Args:
            workspace: Repository workspace with checked-out sources.
            image: Docker image metadata produced by the generator.
            image_name: Fully-qualified image name (including registry and tag).
            verify_registry: Whether to verify the image exists in the registry after push.
            max_push_retries: Maximum number of retries for docker push operations.
        """
        issues: List[RuntimeIssue] = []

        dockerfile_full_path = workspace.get_full_path(image.dockerfile_path)
        build_context_path = workspace.get_full_path(image.build_context)

        if not dockerfile_full_path.exists():
            issues.append(
                RuntimeIssue(
                    code="DOCKER_BUILD_FILE_NOT_FOUND",
                    message=f"Dockerfile not found at {image.dockerfile_path}",
                    subject=image.dockerfile_path,
                )
            )
            return DockerBuildResult(image_name=image_name, issues=issues)

        if not build_context_path.exists():
            issues.append(
                RuntimeIssue(
                    code="DOCKER_BUILD_CONTEXT_NOT_FOUND",
                    message=f"Build context directory not found at {image.build_context}",
                    subject=image.build_context,
                )
            )
            return DockerBuildResult(image_name=image_name, issues=issues)

        build_cmd = [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--load",
            "--progress=plain",
            "-t",
            image_name,
            "-f",
            str(dockerfile_full_path),
            str(build_context_path),
        ]

        self.logger.info("Building Docker image %s with buildx", image_name)
        build_result = self.command_runner.run(build_cmd, timeout=600)

        issues.extend(self._handle_build_result(build_result, image, image_name))
        if any(issue.is_error() for issue in issues):
            return DockerBuildResult(image_name=image_name, issues=issues)

        push_result, push_issues = self._push_image(image_name, max_push_retries)
        issues.extend(push_issues)

        if any(issue.is_error() for issue in issues) or not push_result:
            return DockerBuildResult(image_name=image_name, issues=issues)

        verify_issues = (
            self._verify_image_in_registry(image_name) if verify_registry and "/" in image_name else []
        )
        issues.extend(verify_issues)

        metrics = self._collect_metrics(image_name, image.image_tag, build_result, push_result)

        return DockerBuildResult(image_name=image_name, issues=issues, metrics=metrics)

    def _handle_build_result(
        self,
        build_result: CommandResult,
        image: DockerImageInfo,
        image_name: str,
    ) -> List[RuntimeIssue]:
        issues: List[RuntimeIssue] = []

        if build_result.timed_out:
            issues.append(
                RuntimeIssue(
                    code="DOCKER_BUILDX_TIMEOUT",
                    message="Docker buildx build timed out (>10 minutes)",
                    subject=image.dockerfile_path,
                )
            )
            return issues

        if not build_result.tool_available:
            issues.append(
                RuntimeIssue(
                    code="DOCKER_BUILDX_NOT_FOUND",
                    message="Docker buildx not available - install Docker with buildx support",
                    subject=image.dockerfile_path,
                )
            )
            return issues

        if (build_result.return_code or 0) != 0:
            error_msg = build_result.stderr.strip() or build_result.stdout.strip() or "Unknown buildx error"
            issues.append(
                RuntimeIssue(
                    code="DOCKER_BUILDX_FAILED",
                    message=f"Docker buildx build failed: {error_msg}",
                    subject=image.dockerfile_path,
                )
            )
            self.logger.error("Docker buildx failed for %s: %s", image_name, error_msg)
        else:
            self.logger.info("Build completed successfully for %s", image_name)

        return issues

    def _push_image(
        self,
        image_name: str,
        max_push_retries: int,
    ) -> tuple[Optional[CommandResult], List[RuntimeIssue]]:
        push_result: Optional[CommandResult] = None
        issues: List[RuntimeIssue] = []
        push_success = False

        for attempt in range(max_push_retries):
            if attempt > 0:
                self.logger.info("Retrying docker push (attempt %d/%d)...", attempt + 1, max_push_retries)
                time.sleep(2 ** attempt)

            push_result = self.command_runner.run(["docker", "push", image_name], timeout=300)

            if push_result.timed_out:
                self.logger.warning(
                    "Docker push timed out on attempt %d/%d for %s", attempt + 1, max_push_retries, image_name
                )
                continue

            if not push_result.tool_available:
                issues.append(
                    RuntimeIssue(
                        code="DOCKER_CLI_NOT_FOUND",
                        message="Docker CLI not available for push operation",
                        subject=image_name,
                    )
                )
                break

            if (push_result.return_code or 0) == 0:
                push_success = True
                break

            error_msg = push_result.stderr.strip() or push_result.stdout.strip() or ""
            is_retryable = any(
                keyword in error_msg.lower()
                for keyword in ["timeout", "i/o timeout", "connection", "network", "temporary failure", "proxyconnect"]
            )

            if is_retryable:
                self.logger.warning(
                    "Docker push failed with retryable error on attempt %d/%d: %s",
                    attempt + 1,
                    max_push_retries,
                    error_msg[:200],
                )
            else:
                self.logger.error("Docker push failed with non-retryable error: %s", error_msg[:200])
                issues.append(
                    RuntimeIssue(
                        code="DOCKER_PUSH_FAILED",
                        message=f"Docker push failed: {error_msg}",
                        subject=image_name,
                    )
                )
                break

        if not push_result:
            issues.append(
                RuntimeIssue(
                    code="DOCKER_PUSH_FAILED",
                    message="Docker push failed: no result after retries",
                    subject=image_name,
                )
            )
            return None, issues

        if push_result.timed_out and push_success is False:
            issues.append(
                RuntimeIssue(
                    code="DOCKER_PUSH_TIMEOUT",
                    message=f"Docker push timed out after {max_push_retries} attempts",
                    subject=image_name,
                )
            )
            return push_result, issues

        if not push_success:
            error_msg = push_result.stderr.strip() or push_result.stdout.strip() or "Unknown docker push error"
            issues.append(
                RuntimeIssue(
                    code="DOCKER_PUSH_FAILED",
                    message=f"Docker push failed after {max_push_retries} attempts: {error_msg}",
                    subject=image_name,
                )
            )
            return push_result, issues

        self.logger.info("Successfully pushed image to registry: %s", image_name)
        return push_result, issues

    def _verify_image_in_registry(self, image_name: str) -> List[RuntimeIssue]:
        issues: List[RuntimeIssue] = []

        registry, image_path = image_name.split("/", 1)
        if ":" in image_path:
            image_repo, image_tag_version = image_path.rsplit(":", 1)
        else:
            image_repo = image_path
            image_tag_version = "latest"

        verify_url = f"http://{registry}/v2/{image_repo}/tags/list"
        self.logger.info("Verifying image in registry: %s", image_name)

        verified = False
        last_error: Optional[str] = None
        for attempt in range(3):
            try:
                self.logger.debug("Verification attempt %s/3: GET %s", attempt + 1, verify_url)
                verify_response = requests.get(verify_url, timeout=10)
                if verify_response.status_code == 200:
                    tags = verify_response.json().get("tags", [])
                    if tags and image_tag_version in tags:
                        self.logger.info("Successfully verified image in registry: %s", image_name)
                        verified = True
                        break
                    last_error = f"tag '{image_tag_version}' not found in tags list: {tags}"
                else:
                    last_error = f"registry returned status {verify_response.status_code}"
            except Exception as exc:  # pragma: no cover - network variability
                last_error = str(exc)

            if attempt < 2:
                time.sleep(1)

        if not verified:
            issues.append(
                RuntimeIssue(
                    code="DOCKER_PUSH_VERIFICATION_FAILED",
                    message=f"Image pushed but verification failed after 3 attempts (URL: {verify_url}): {last_error}",
                    subject=image_name,
                )
            )

        return issues

    def _collect_metrics(
        self,
        image_name: str,
        image_tag: str,
        build_result: CommandResult,
        push_result: CommandResult,
    ) -> Optional[DockerBuildMetrics]:
        total_build_time = build_result.duration + (push_result.duration if push_result.return_code == 0 else 0.0)

        size_result = self.command_runner.run(
            ["docker", "image", "inspect", image_name, "--format", "{{.Size}}"],
            timeout=10,
        )
        layers_result = self.command_runner.run(
            ["docker", "image", "inspect", image_name, "--format", "{{len .RootFS.Layers}}"],
            timeout=10,
        )

        if size_result.succeeded() and layers_result.succeeded():
            try:
                image_size_bytes = int(size_result.stdout.strip())
                image_size_mb = image_size_bytes / (1024 * 1024)
                layers_count = int(layers_result.stdout.strip())
                metrics = DockerBuildMetrics(
                    image_tag=image_tag,
                    build_time=total_build_time,
                    image_size_mb=round(image_size_mb, 2),
                    layers_count=layers_count,
                    image_name=image_name,
                )
                self.logger.info(
                    "Image metrics: %.2f MB, %s layers, built in %.1fs",
                    metrics.image_size_mb,
                    metrics.layers_count,
                    metrics.build_time,
                )
                return metrics
            except ValueError:
                self.logger.warning("Could not parse docker inspect output for %s", image_name)
        else:
            self.logger.warning("Could not inspect image metrics for %s", image_name)

        return None
