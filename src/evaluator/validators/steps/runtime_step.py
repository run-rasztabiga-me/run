from __future__ import annotations

from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import has_error_issues
from ..utils import runtime_issues_to_validation
from src.runtime import IngressRuntimeChecker, KubernetesDeployer


class RuntimeValidationStep:
    """Validate runtime availability via ingress endpoint checks."""

    name = "runtime"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        if not state.test_endpoint:
            # No test endpoint means we can't validate runtime, so it's a failure
            return ValidationStepResult(runtime_success=False)

        namespace = context.run_context.k8s_namespace
        issues = []

        if state.applied_resources:
            deployer = KubernetesDeployer(
                workspace=context.workspace,
                command_runner=context.command_runner,
                config=context.config,
                logger=context.logger,
            )
            readiness_issues = deployer.wait_for_resources_ready(
                namespace=namespace,
                resources=state.applied_resources,
            )
            readiness_results = runtime_issues_to_validation(readiness_issues, default_subject=namespace)
            issues.extend(readiness_results)
            if has_error_issues(readiness_results):
                return ValidationStepResult(
                    issues=issues,
                    runtime_success=False,
                )

        checker = IngressRuntimeChecker(
            command_runner=context.command_runner,
            config=context.config,
            logger=context.logger,
        )

        result = checker.check(namespace, state.test_endpoint)
        issues.extend(runtime_issues_to_validation(result.issues, default_subject="runtime"))
        runtime_success = result.success

        return ValidationStepResult(
            issues=issues,
            runtime_success=runtime_success,
        )
