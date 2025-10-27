"""Kubernetes deployment helpers shared between evaluator and runtime automation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import yaml

from src.common.command_runner import CommandRunner
from src.common.models import DockerImageInfo
from src.generator.core.config import GeneratorConfig
from src.generator.core.workspace import RepositoryWorkspace
from src.generator.core.workspace_models import RunContext

from .issues import RuntimeIssue


@dataclass(slots=True)
class AppliedResource:
    """Represents a Kubernetes resource applied to the cluster."""

    kind: str
    name: str


@dataclass(slots=True)
class NamespacePreparationResult:
    """Outcome of namespace preparation."""

    issues: List[RuntimeIssue]

    @property
    def success(self) -> bool:
        return not any(issue.is_error() for issue in self.issues)


@dataclass(slots=True)
class ManifestApplyResult:
    """Outcome of applying a manifest."""

    issues: List[RuntimeIssue]
    resources: List[AppliedResource]

    @property
    def success(self) -> bool:
        return not any(issue.is_error() for issue in self.issues)


class KubernetesDeployer:
    """Apply Kubernetes manifests with run-scoped patching and readiness checks."""

    def __init__(
        self,
        *,
        workspace: RepositoryWorkspace,
        command_runner: CommandRunner,
        config: GeneratorConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.workspace = workspace
        self.command_runner = command_runner
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def prepare_namespace(self, namespace: str, cleanup_existing: bool = True) -> NamespacePreparationResult:
        issues: List[RuntimeIssue] = []

        if cleanup_existing:
            self._delete_namespace(namespace)

        issues.extend(self._create_namespace(namespace))
        return NamespacePreparationResult(issues=issues)

    def apply_manifest(
        self,
        *,
        manifest_path: str,
        namespace: str,
        repo_name: str,
        run_context: RunContext,
        llm_image_tags: Set[str],
    ) -> ManifestApplyResult:
        issues: List[RuntimeIssue] = []
        resources: List[AppliedResource] = []
        patched_manifest_path: Optional[Path] = None

        try:
            manifest_full_path = self.workspace.get_full_path(manifest_path)

            if not manifest_full_path.exists():
                issues.append(
                    RuntimeIssue(
                        code="K8S_MANIFEST_NOT_FOUND",
                        message=f"Manifest file not found: {manifest_path}",
                        subject=manifest_path,
                    )
                )
                return ManifestApplyResult(issues=issues, resources=resources)

            if manifest_full_path.suffix.lower() not in [".yaml", ".yml"]:
                self.logger.info("Skipping non-YAML file: %s", manifest_path)
                return ManifestApplyResult(issues=issues, resources=resources)

            patched_manifest_path = self._patch_image_names(
                manifest_path=manifest_full_path,
                repo_name=repo_name,
                run_context=run_context,
                llm_image_tags=llm_image_tags,
            )

            self.logger.info("Applying manifest %s to namespace %s", manifest_path, namespace)
            result = self.command_runner.run(
                ["kubectl", "apply", "-f", str(patched_manifest_path), "-n", namespace],
                timeout=60,
            )

            if result.timed_out:
                issues.append(
                    RuntimeIssue(
                        code="K8S_APPLY_TIMEOUT",
                        message="kubectl apply timed out",
                        subject=manifest_path,
                    )
                )
            elif not result.tool_available:
                issues.append(
                    RuntimeIssue(
                        code="K8S_APPLY_ERROR",
                        message="kubectl not available - cannot apply manifest",
                        subject=manifest_path,
                    )
                )
            elif (result.return_code or 0) != 0:
                issues.append(
                    RuntimeIssue(
                        code="K8S_APPLY_FAILED",
                        message=f"kubectl apply failed: {result.stderr.strip()}",
                        subject=manifest_path,
                    )
                )
            else:
                self.logger.info("Successfully applied %s", manifest_path)
                resources = self._extract_resource_names(manifest_full_path)

        except Exception as exc:
            issues.append(
                RuntimeIssue(
                    code="K8S_APPLY_ERROR",
                    message=f"Error applying manifest: {exc}",
                    subject=manifest_path,
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
                    self.logger.debug("Cleaned up patched manifest: %s", patched_manifest_path)
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    self.logger.warning("Failed to cleanup patched manifest %s: %s", patched_manifest_path, exc)

        return ManifestApplyResult(issues=issues, resources=resources)

    def wait_for_resources_ready(
        self,
        *,
        namespace: str,
        resources: Sequence[AppliedResource],
        timeout: int = 120,
    ) -> List[RuntimeIssue]:
        issues: List[RuntimeIssue] = []

        for resource in resources:
            kind = resource.kind
            name = resource.name

            if kind not in ["Deployment", "StatefulSet"]:
                continue

            self.logger.info("Waiting for %s/%s to be ready in namespace %s", kind, name, namespace)

            try:
                wait_result = self._wait_for_resource(kind, name, namespace, timeout)

                if wait_result is None:
                    continue

                if wait_result.timed_out:
                    issues.append(
                        RuntimeIssue(
                            code="K8S_READY_TIMEOUT",
                            message=f"{kind} {name} readiness check timed out",
                            subject=name,
                        )
                    )
                    self.logger.warning("%s/%s did not become ready within %ss", kind, name, timeout)
                elif (wait_result.return_code or 0) != 0:
                    issues.append(
                        RuntimeIssue(
                            code="K8S_RESOURCE_NOT_READY",
                            message=f"{kind} {name} not ready within {timeout}s: {(wait_result.stderr or '').strip()}",
                            subject=name,
                        )
                    )
                    self.logger.warning(
                        "%s/%s failed readiness check within %ss: %s",
                        kind,
                        name,
                        timeout,
                        (wait_result.stderr or "").strip(),
                    )
                else:
                    self.logger.info("%s/%s is ready", kind, name)

            except Exception as exc:
                issues.append(
                    RuntimeIssue(
                        code="K8S_READY_CHECK_ERROR",
                        message=f"Error checking {kind} {name} status: {exc}",
                        subject=name,
                    )
                )
                self.logger.error("Failed to check readiness for %s/%s: %s", kind, name, exc)

        return issues

    @staticmethod
    def collect_llm_image_tags(docker_images: Sequence[DockerImageInfo]) -> Set[str]:
        tags: Set[str] = set()
        for image in docker_images:
            image_tag = image.image_tag.split(":", 1)[0]
            tags.add(image_tag)
        return tags

    def _delete_namespace(self, namespace: str) -> None:
        self.logger.info("Cleaning up old namespace: %s", namespace)

        delete_result = self.command_runner.run(
            ["kubectl", "delete", "namespace", namespace, "--ignore-not-found=true"],
            timeout=30,
        )
        if delete_result.timed_out:
            self.logger.warning("Timeout deleting namespace %s", namespace)

        wait_result = self.command_runner.run(
            ["kubectl", "wait", "--for=delete", f"namespace/{namespace}", "--timeout=60s"],
            timeout=70,
        )
        if wait_result.timed_out:
            self.logger.warning("Timeout waiting for namespace %s deletion", namespace)
        if (wait_result.return_code or 0) != 0 and not wait_result.timed_out:
            self.logger.debug(
                "kubectl wait returned %s: %s", wait_result.return_code, wait_result.stderr.strip()
            )

    def _create_namespace(self, namespace: str) -> List[RuntimeIssue]:
        issues: List[RuntimeIssue] = []

        self.logger.info("Creating namespace: %s", namespace)
        result = self.command_runner.run(
            ["kubectl", "create", "namespace", namespace],
            timeout=30,
        )

        if result.timed_out:
            issues.append(
                RuntimeIssue(
                    code="K8S_NAMESPACE_TIMEOUT",
                    message="Namespace creation timed out",
                    subject=namespace,
                )
            )
            return issues

        if not result.tool_available:
            issues.append(
                RuntimeIssue(
                    code="K8S_NAMESPACE_ERROR",
                    message="kubectl not available - cannot create namespace",
                    subject=namespace,
                )
            )
            return issues

        if (result.return_code or 0) != 0:
            issues.append(
                RuntimeIssue(
                    code="K8S_NAMESPACE_CREATE_FAILED",
                    message=f"Failed to create namespace: {result.stderr}",
                    subject=namespace,
                )
            )
        else:
            self.logger.info("Successfully created namespace: %s", namespace)

        return issues

    def _wait_for_resource(
        self,
        kind: str,
        name: str,
        namespace: str,
        timeout: int,
    ):
        if kind == "Deployment":
            return self.command_runner.run(
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

        if kind == "StatefulSet":
            replicas_result = self.command_runner.run(
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
            return self.command_runner.run(
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

        return None

    def _patch_image_names(
        self,
        *,
        manifest_path: Path,
        repo_name: str,
        run_context: RunContext,
        llm_image_tags: Set[str],
    ) -> Path:
        if not llm_image_tags:
            return manifest_path

        try:
            with open(manifest_path, "r") as handle:
                docs = list(yaml.safe_load_all(handle))

            for doc in docs:
                if not doc or not isinstance(doc, dict):
                    continue

                kind = doc.get("kind")
                if kind == "Ingress":
                    self._patch_ingress_hosts(doc, run_context)
                    continue

                if kind not in ["Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"]:
                    continue

                containers = self._extract_containers(doc, kind)
                for container in containers:
                    image = container.get("image")
                    if not image:
                        self.logger.debug("Container has no 'image' field, skipping")
                        continue

                    if "/" in image and image.startswith(self.config.docker_registry):
                        self.logger.info("Image already has registry prefix, skipping: %s", image)
                        continue

                    image_tag = image.split(":", 1)[0] if ":" in image else image
                    if "/" in image_tag:
                        image_tag = image_tag.split("/")[-1]

                    if image_tag not in llm_image_tags:
                        self.logger.info(
                            "Skipping official/external image: %s (tag '%s' not in LLM image tags)",
                            image,
                            image_tag,
                        )
                        continue

                    full_image = self.config.get_full_image_name(
                        repo_name,
                        image_tag,
                        version=run_context.run_id,
                    )
                    self.logger.info("Patching image name: %s → %s", image, full_image)
                    container["image"] = full_image

            temp_path = manifest_path.with_suffix(".patched.yaml")
            with open(temp_path, "w") as handle:
                yaml.safe_dump_all(docs, handle)

            return temp_path

        except Exception as exc:
            self.logger.error("Failed to patch image names in %s: %s", manifest_path, exc, exc_info=True)
            self.logger.warning("Falling back to original manifest path: %s", manifest_path)
            return manifest_path

    def _patch_ingress_hosts(self, doc: Dict[str, object], run_context: RunContext) -> None:
        rules = doc.get("spec", {}).get("rules", [])  # type: ignore[index]
        for rule in rules:
            if not isinstance(rule, dict) or "host" not in rule:
                continue
            original_host = rule["host"]
            run_prefix = f"run-{run_context.run_id[:8]}"
            patched_host = f"{run_prefix}.{original_host}"
            rule["host"] = patched_host
            self.logger.info("Patching ingress host: %s → %s", original_host, patched_host)

        tls_configs = doc.get("spec", {}).get("tls", [])  # type: ignore[index]
        for tls_config in tls_configs:
            if not isinstance(tls_config, dict) or "hosts" not in tls_config:
                continue
            original_hosts = tls_config["hosts"]
            patched_hosts = []
            for original_host in original_hosts:
                run_prefix = f"run-{run_context.run_id[:8]}"
                patched_hosts.append(f"{run_prefix}.{original_host}")
            tls_config["hosts"] = patched_hosts
            self.logger.info("Patching TLS hosts: %s → %s", original_hosts, patched_hosts)

    @staticmethod
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

    def _extract_resource_names(self, manifest_path: Path) -> List[AppliedResource]:
        resources: List[AppliedResource] = []
        try:
            with open(manifest_path, "r") as handle:
                docs = list(yaml.safe_load_all(handle))

            for doc in docs:
                if doc and isinstance(doc, dict):
                    kind = doc.get("kind")
                    name = doc.get("metadata", {}).get("name")
                    if kind and name:
                        resources.append(AppliedResource(kind=kind, name=name))
        except Exception as exc:
            self.logger.warning("Failed to extract resource names from %s: %s", manifest_path, exc)

        return resources
