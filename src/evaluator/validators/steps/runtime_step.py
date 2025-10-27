from __future__ import annotations

from ..pipeline import ValidationContext, ValidationState, ValidationStepResult
from ...core.models import ValidationIssue, ValidationSeverity
from ..utils import runtime_issues_to_validation
from src.runtime import IngressRuntimeChecker


class RuntimeValidationStep:
    """Validate runtime availability via ingress endpoint checks."""

    name = "runtime"

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        if not state.test_endpoint:
            # No test endpoint means we can't validate runtime, so it's a failure
            return ValidationStepResult(runtime_success=False)

        namespace = context.run_context.k8s_namespace
        checker = IngressRuntimeChecker(
            command_runner=context.command_runner,
            config=context.config,
            logger=context.logger,
        )

        result = checker.check(namespace, state.test_endpoint)
        issues = runtime_issues_to_validation(result.issues, default_subject="runtime")
        runtime_success = result.success

        return ValidationStepResult(
            issues=issues,
            runtime_success=runtime_success,
        )
