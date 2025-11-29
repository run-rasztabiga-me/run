from __future__ import annotations

from typing import List

from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import ValidationIssue, ValidationSeverity, has_error_issues
from ..utils import runtime_issues_to_validation
from src.runtime import AppliedResource, KubernetesDeployer


class KubernetesApplyStep:
    """Apply manifests to the cluster and wait for workloads to become ready."""

    name = "kubernetes_apply"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        issues: List[ValidationIssue] = []
        namespace = context.run_context.k8s_namespace

        if not state.manifests:
            context.logger.info("No manifests provided; skipping Kubernetes apply step.")
            return ValidationStepResult()

        context.logger.info("Using run-scoped namespace: %s", namespace)
        deployer = KubernetesDeployer(
            workspace=context.workspace,
            command_runner=context.command_runner,
            config=context.config,
            logger=context.logger,
        )

        try:
            prep_result = deployer.prepare_namespace(namespace)
            issues.extend(runtime_issues_to_validation(prep_result.issues, default_subject=namespace))

            if has_error_issues(issues):
                return ValidationStepResult(issues=issues, continue_pipeline=False)

            llm_image_tags = KubernetesDeployer.collect_llm_image_tags(state.docker_images)
            if llm_image_tags:
                context.logger.info("LLM-generated image tags to patch: %s", llm_image_tags)

            applied_resources: List[AppliedResource] = []
            for manifest_path in state.manifests:
                apply_result = deployer.apply_manifest(
                    manifest_path=manifest_path,
                    namespace=namespace,
                    repo_name=state.repo_name,
                    run_context=context.run_context,
                    llm_image_tags=llm_image_tags,
                )
                issues.extend(
                    runtime_issues_to_validation(apply_result.issues, default_subject=manifest_path)
                )
                applied_resources.extend(apply_result.resources)

            if applied_resources:
                readiness_issues = deployer.wait_for_resources_ready(
                    namespace=namespace,
                    resources=applied_resources,
                )
                issues.extend(
                    runtime_issues_to_validation(readiness_issues, default_subject=namespace)
                )

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

        return ValidationStepResult(issues=issues, continue_pipeline=not has_error_issues(issues))
