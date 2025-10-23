from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import ValidationIssue, ValidationSeverity
from src.common.models import DockerImageInfo


class KubernetesApplyStep:
    """Apply manifests to the cluster and wait for workloads to become ready."""

    name = "kubernetes_apply"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        issues: List[ValidationIssue] = []
        namespace = context.run_context.k8s_namespace
        context.logger.info("Using run-scoped namespace: %s", namespace)

        try:
            _delete_namespace(context, namespace)
            create_issues = _create_namespace(context, namespace)
            issues.extend(create_issues)

            if any(issue.severity == ValidationSeverity.ERROR for issue in create_issues):
                return ValidationStepResult(issues=issues, continue_pipeline=False)

            applied_resources: List[Dict[str, str]] = []
            for manifest_path in state.manifests:
                apply_issues, resources = _apply_manifest(
                    context=context,
                    manifest_path=manifest_path,
                    namespace=namespace,
                    repo_name=state.repo_name,
                    docker_images=list(state.docker_images),
                )
                issues.extend(apply_issues)
                applied_resources.extend(resources)

            if applied_resources:
                issues.extend(_wait_for_resources_ready(context, applied_resources, namespace))

        except Exception as exc:  # pragma: no cover - defensive
            issues.append(
                ValidationIssue(
                    file_path="cluster",
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to apply manifests: {exc}",
                    rule_id="K8S_APPLY_ERROR",
                )
            )

        return ValidationStepResult(issues=issues)


def _delete_namespace(context: ValidationContext, namespace: str) -> None:
    context.logger.info(f"Cleaning up old namespace: {namespace}")

    delete_result = context.command_runner.run(
        ["kubectl", "delete", "namespace", namespace, "--ignore-not-found=true"],
        timeout=30,
    )
    if delete_result.timed_out:
        context.logger.warning("Timeout deleting namespace %s", namespace)

    wait_result = context.command_runner.run(
        ["kubectl", "wait", "--for=delete", f"namespace/{namespace}", "--timeout=60s"],
        timeout=70,
    )
    if wait_result.timed_out:
        context.logger.warning("Timeout waiting for namespace %s deletion", namespace)
    if (wait_result.return_code or 0) != 0 and not wait_result.timed_out:
        context.logger.debug("kubectl wait returned %s: %s", wait_result.return_code, wait_result.stderr.strip())


def _create_namespace(context: ValidationContext, namespace: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    context.logger.info(f"Creating namespace: {namespace}")

    result = context.command_runner.run(
        ["kubectl", "create", "namespace", namespace],
        timeout=30,
    )

    if result.timed_out:
        issues.append(
            ValidationIssue(
                file_path=namespace,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="Namespace creation timed out",
                rule_id="K8S_NAMESPACE_TIMEOUT",
            )
        )
        return issues

    if not result.tool_available:
        issues.append(
            ValidationIssue(
                file_path=namespace,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message="kubectl not available - cannot create namespace",
                rule_id="K8S_NAMESPACE_ERROR",
            )
        )
        return issues

    if (result.return_code or 0) != 0:
        issues.append(
            ValidationIssue(
                file_path=namespace,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to create namespace: {result.stderr}",
                rule_id="K8S_NAMESPACE_CREATE_FAILED",
            )
        )
    else:
        context.logger.info("Successfully created namespace: %s", namespace)

    return issues


def _apply_manifest(
    *,
    context: ValidationContext,
    manifest_path: str,
    namespace: str,
    repo_name: str,
    docker_images: Sequence[DockerImageInfo],
) -> Tuple[List[ValidationIssue], List[Dict[str, str]]]:
    issues: List[ValidationIssue] = []
    applied_resources: List[Dict[str, str]] = []
    patched_manifest_path: Optional[Path] = None

    try:
        manifest_full_path = context.workspace.get_full_path(manifest_path)

        if not manifest_full_path.exists():
            issues.append(
                ValidationIssue(
                    file_path=manifest_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Manifest file not found: {manifest_path}",
                    rule_id="K8S_MANIFEST_NOT_FOUND",
                )
            )
            return issues, applied_resources

        if manifest_full_path.suffix.lower() not in [".yaml", ".yml"]:
            context.logger.info("Skipping non-YAML file: %s", manifest_path)
            return issues, applied_resources

        patched_manifest_path = _patch_image_names(
            context=context,
            manifest_path=manifest_full_path,
            repo_name=repo_name,
            docker_images=docker_images,
        )

        context.logger.info("Applying manifest %s to namespace %s", manifest_path, namespace)
        result = context.command_runner.run(
            ["kubectl", "apply", "-f", str(patched_manifest_path), "-n", namespace],
            timeout=60,
        )

        if result.timed_out:
            issues.append(
                ValidationIssue(
                    file_path=manifest_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message="kubectl apply timed out",
                    rule_id="K8S_APPLY_TIMEOUT",
                )
            )
        elif not result.tool_available:
            issues.append(
                ValidationIssue(
                    file_path=manifest_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message="kubectl not available - cannot apply manifest",
                    rule_id="K8S_APPLY_ERROR",
                )
            )
        elif (result.return_code or 0) != 0:
            issues.append(
                ValidationIssue(
                    file_path=manifest_path,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"kubectl apply failed: {result.stderr.strip()}",
                    rule_id="K8S_APPLY_FAILED",
                )
            )
        else:
            context.logger.info("Successfully applied %s", manifest_path)
            applied_resources = _extract_resource_names(context, manifest_full_path)

    except Exception as exc:
        issues.append(
            ValidationIssue(
                file_path=manifest_path,
                line_number=None,
                severity=ValidationSeverity.ERROR,
                message=f"Error applying manifest: {exc}",
                rule_id="K8S_APPLY_ERROR",
            )
        )
    finally:
        if (
            patched_manifest_path
            and patched_manifest_path != manifest_full_path
            and patched_manifest_path.exists()
        ):
            try:
                patched_manifest_path.unlink(missing_ok=True)
                context.logger.debug("Cleaned up patched manifest: %s", patched_manifest_path)
            except Exception as exc:  # pragma: no cover - filesystem warning
                context.logger.warning("Failed to cleanup patched manifest %s: %s", patched_manifest_path, exc)

    return issues, applied_resources


def _wait_for_resources_ready(
    context: ValidationContext,
    resources: List[Dict[str, str]],
    namespace: str,
    timeout: int = 120,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    for resource in resources:
        kind = resource["kind"]
        name = resource["name"]

        if kind not in ["Deployment", "StatefulSet"]:
            continue

        context.logger.info("Waiting for %s/%s to be ready in namespace %s", kind, name, namespace)

        try:
            wait_result = None
            if kind == "Deployment":
                wait_result = context.command_runner.run(
                    [
                        "kubectl",
                        "wait",
                        "--for=condition=available",
                        f'{kind.lower()}/{name}',
                        "-n",
                        namespace,
                        f"--timeout={timeout}s",
                    ],
                    timeout=timeout + 10,
                )
            elif kind == "StatefulSet":
                replicas_result = context.command_runner.run(
                    [
                        "kubectl",
                        "get",
                        f'{kind.lower()}/{name}',
                        "-n",
                        namespace,
                        "-o",
                        "jsonpath={.spec.replicas}",
                    ],
                    timeout=10,
                )
                if not replicas_result.succeeded():
                    raise RuntimeError(f"Failed to get replica count: {replicas_result.stderr}")

                replica_count = replicas_result.stdout.strip() or "1"
                wait_result = context.command_runner.run(
                    [
                        "kubectl",
                        "wait",
                        f"--for=jsonpath={{.status.readyReplicas}}={replica_count}",
                        f'{kind.lower()}/{name}',
                        "-n",
                        namespace,
                        f"--timeout={timeout}s",
                    ],
                    timeout=timeout + 10,
                )

            if wait_result is None:
                continue

            if wait_result.timed_out:
                issues.append(
                    ValidationIssue(
                        file_path=name,
                        line_number=None,
                        severity=ValidationSeverity.ERROR,
                        message=f"{kind} {name} readiness check timed out",
                        rule_id="K8S_READY_TIMEOUT",
                    )
                )
            elif (wait_result.return_code or 0) != 0:
                issues.append(
                    ValidationIssue(
                        file_path=name,
                        line_number=None,
                        severity=ValidationSeverity.ERROR,
                        message=f"{kind} {name} not ready within {timeout}s: {wait_result.stderr.strip()}",
                        rule_id="K8S_RESOURCE_NOT_READY",
                    )
                )
            else:
                context.logger.info("%s/%s is ready", kind, name)

        except Exception as exc:
            issues.append(
                ValidationIssue(
                    file_path=name,
                    line_number=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Error checking {kind} {name} status: {exc}",
                    rule_id="K8S_READY_CHECK_ERROR",
                )
            )

    return issues


def _patch_image_names(
    *,
    context: ValidationContext,
    manifest_path: Path,
    repo_name: str,
    docker_images: Sequence[DockerImageInfo],
) -> Path:
    llm_image_tags = set()
    for img in docker_images:
        tag = img.image_tag.split(":", 1)[0]
        llm_image_tags.add(tag)
    context.logger.info("LLM-generated image tags to patch: %s", llm_image_tags)

    try:
        with open(manifest_path, "r") as handle:
            docs = list(yaml.safe_load_all(handle))

        for doc in docs:
            if not doc or not isinstance(doc, dict):
                continue

            kind = doc.get("kind")
            if kind == "Ingress":
                _patch_ingress_hosts(context, doc)
                continue

            if kind not in ["Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"]:
                continue

            containers = _extract_containers(doc, kind)
            for container in containers:
                image = container.get("image")
                if not image:
                    context.logger.debug("Container has no 'image' field, skipping")
                    continue

                if "/" in image and image.startswith(context.config.docker_registry):
                    context.logger.info("Image already has registry prefix, skipping: %s", image)
                    continue

                image_tag = image.split(":", 1)[0] if ":" in image else image
                if "/" in image_tag:
                    image_tag = image_tag.split("/")[-1]

                if image_tag not in llm_image_tags:
                    context.logger.info(
                        "Skipping official/external image: %s (tag '%s' not in LLM image tags)",
                        image,
                        image_tag,
                    )
                    continue

                full_image = context.config.get_full_image_name(repo_name, image_tag, version=context.run_context.run_id)
                context.logger.info("✓ Patching image name: %s → %s", image, full_image)
                container["image"] = full_image

        temp_path = manifest_path.with_suffix(".patched.yaml")
        with open(temp_path, "w") as handle:
            yaml.safe_dump_all(docs, handle)

        return temp_path

    except Exception as exc:
        context.logger.error("Failed to patch image names in %s: %s", manifest_path, exc, exc_info=True)
        context.logger.warning("Falling back to original manifest path: %s", manifest_path)
        return manifest_path


def _patch_ingress_hosts(context: ValidationContext, doc: Dict[str, object]) -> None:
    rules = doc.get("spec", {}).get("rules", [])  # type: ignore[index]
    for rule in rules:
        if not isinstance(rule, dict) or "host" not in rule:
            continue
        original_host = rule["host"]
        run_prefix = f"run-{context.run_context.run_id[:8]}"
        patched_host = f"{run_prefix}.{original_host}"
        rule["host"] = patched_host
        context.logger.info("✓ Patching ingress host: %s → %s", original_host, patched_host)

    tls_configs = doc.get("spec", {}).get("tls", [])  # type: ignore[index]
    for tls_config in tls_configs:
        if not isinstance(tls_config, dict) or "hosts" not in tls_config:
            continue
        original_hosts = tls_config["hosts"]
        patched_hosts = []
        for original_host in original_hosts:
            run_prefix = f"run-{context.run_context.run_id[:8]}"
            patched_hosts.append(f"{run_prefix}.{original_host}")
        tls_config["hosts"] = patched_hosts
        context.logger.info("✓ Patching TLS hosts: %s → %s", original_hosts, patched_hosts)


def _extract_containers(doc: Dict[str, object], kind: str) -> List[Dict[str, object]]:
    if kind in ["Deployment", "StatefulSet", "DaemonSet"]:
        return doc.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])  # type: ignore[index]
    if kind == "Job":
        return doc.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])  # type: ignore[index]
    if kind == "CronJob":
        return (
            doc.get("spec", {})
            .get("jobTemplate", {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )  # type: ignore[index]
    return []


def _extract_resource_names(context: ValidationContext, manifest_path: Path) -> List[Dict[str, str]]:
    resources: List[Dict[str, str]] = []
    try:
        with open(manifest_path, "r") as handle:
            docs = list(yaml.safe_load_all(handle))

        for doc in docs:
            if doc and isinstance(doc, dict):
                kind = doc.get("kind")
                name = doc.get("metadata", {}).get("name")
                if kind and name:
                    resources.append({"kind": kind, "name": name})
    except Exception as exc:
        context.logger.warning("Failed to extract resource names from %s: %s", manifest_path, exc)

    return resources
