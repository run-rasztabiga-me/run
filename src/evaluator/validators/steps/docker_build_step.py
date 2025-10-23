from __future__ import annotations

import time
from typing import List, Optional, Tuple

import requests

from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import DockerBuildMetrics, ValidationIssue, ValidationSeverity


class DockerBuildStep:
    """Build Docker images and push them to the configured registry."""

    name = "docker_build"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        issues: List[ValidationIssue] = []
        metrics: List[DockerBuildMetrics] = []

        for image_info in state.docker_images:
            context.logger.info(
                "Building Docker image: %s from %s",
                image_info.image_tag,
                image_info.dockerfile_path,
            )

            full_image_name = context.config.get_full_image_name(
                state.repo_name,
                image_info.image_tag,
                version=context.run_context.run_id,
            )
            context.logger.info("Full image name with run-scoped tag: %s", full_image_name)

            build_issues, metric = _build_and_push_image(
                context=context,
                dockerfile_path=image_info.dockerfile_path,
                build_context=image_info.build_context,
                image_name=full_image_name,
                image_tag=image_info.image_tag,
            )
            issues.extend(build_issues)
            if metric:
                metrics.append(metric)

        continue_pipeline = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationStepResult(
            issues=issues,
            build_metrics=metrics,
            continue_pipeline=continue_pipeline,
        )


def _build_and_push_image(
    *,
    context: ValidationContext,
    dockerfile_path: str,
    build_context: str,
    image_name: str,
    image_tag: str,
) -> Tuple[List[ValidationIssue], Optional[DockerBuildMetrics]]:
    issues: List[ValidationIssue] = []
    metrics: Optional[DockerBuildMetrics] = None

    dockerfile_full_path = context.workspace.get_full_path(dockerfile_path)
    context_full_path = context.workspace.get_full_path(build_context)

    if not dockerfile_full_path.exists():
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Dockerfile not found at {dockerfile_path}",
                rule_id="DOCKER_BUILD_FILE_NOT_FOUND",
            )
        )
        return issues, None

    if not context_full_path.exists():
        issues.append(
            ValidationIssue(
                file_path=build_context,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Build context directory not found at {build_context}",
                rule_id="DOCKER_BUILD_CONTEXT_NOT_FOUND",
            )
        )
        return issues, None

    context.logger.info("Building Docker image with buildx: %s", image_name)

    build_cmd = [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/amd64",
        "--no-cache",
        "--pull",
        "--load",
        "--progress=plain",
        "-t",
        image_name,
        "-f",
        str(dockerfile_full_path),
        str(context_full_path),
    ]
    build_result = context.command_runner.run(build_cmd, timeout=600)

    if build_result.timed_out:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Docker buildx build timed out (>10 minutes)",
                rule_id="DOCKER_BUILDX_TIMEOUT",
            )
        )
        return issues, None

    if not build_result.tool_available:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Docker buildx not available - install Docker with buildx support",
                rule_id="DOCKER_BUILDX_NOT_FOUND",
            )
        )
        return issues, None

    if (build_result.return_code or 0) != 0:
        error_msg = build_result.stderr.strip() or build_result.stdout.strip() or "Unknown buildx error"
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Docker buildx build failed: {error_msg}",
                rule_id="DOCKER_BUILDX_FAILED",
            )
        )
        return issues, None

    context.logger.info("✓ Build completed successfully for image: %s", image_name)

    context.logger.info("Pushing image to registry: %s", image_name)
    push_result = context.command_runner.run(["docker", "push", image_name], timeout=300)

    total_build_time = build_result.duration + (push_result.duration if push_result.return_code == 0 else 0.0)

    if push_result.timed_out:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Docker push timed out",
                rule_id="DOCKER_PUSH_TIMEOUT",
            )
        )
        return issues, None

    if not push_result.tool_available:
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Docker CLI not available for push operation",
                rule_id="DOCKER_CLI_NOT_FOUND",
            )
        )
        return issues, None

    if (push_result.return_code or 0) != 0:
        error_msg = push_result.stderr.strip() or push_result.stdout.strip() or "Unknown docker push error"
        issues.append(
            ValidationIssue(
                file_path=dockerfile_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Docker push failed: {error_msg}",
                rule_id="DOCKER_PUSH_FAILED",
            )
        )
        return issues, None

    context.logger.info("✓ Successfully pushed image to registry: %s", image_name)

    if "/" in image_name:
        registry, image_path = image_name.split("/", 1)
        if ":" in image_path:
            image_repo, image_tag_version = image_path.rsplit(":", 1)
        else:
            image_repo = image_path
            image_tag_version = "latest"

        verify_url = f"http://{registry}/v2/{image_repo}/tags/list"
        context.logger.info("Verifying image in registry: %s", image_name)

        verified = False
        last_error = None
        for attempt in range(3):
            try:
                context.logger.debug("Verification attempt %s/3: GET %s", attempt + 1, verify_url)
                verify_response = requests.get(verify_url, timeout=10)
                if verify_response.status_code == 200:
                    tags = verify_response.json().get("tags", [])
                    if tags and image_tag_version in tags:
                        context.logger.info("✓ Successfully verified image in registry: %s", image_name)
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
                ValidationIssue(
                    file_path=dockerfile_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Image pushed but verification failed after 3 attempts (URL: {verify_url}): {last_error}",
                    rule_id="DOCKER_PUSH_VERIFICATION_FAILED",
                )
            )
            return issues, None

    size_result = context.command_runner.run(
        ["docker", "image", "inspect", image_name, "--format", "{{.Size}}"],
        timeout=10,
    )
    layers_result = context.command_runner.run(
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
            )
            context.logger.info(
                "Image metrics: %.2f MB, %s layers, built in %.1fs",
                image_size_mb,
                layers_count,
                total_build_time,
            )
        except ValueError:
            context.logger.warning("Could not parse docker inspect output for %s", image_name)
    else:
        context.logger.warning("Could not inspect image metrics for %s", image_name)

    return issues, metrics
