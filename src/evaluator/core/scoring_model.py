"""
Validator Issue Aggregation Model

This module implements a reproducible scoring system that aggregates validator errors,
warnings, and severities into comprehensive scores reflecting the full issue surface.

Design Principles:
1. Phase-based scoring: Different validation phases have different weights
2. Severity-based penalties: Issues have different impacts based on severity
3. Reproducible: Deterministic weights ensure consistent scoring across runs
4. Transparent: Per-phase breakdowns enable debugging and analysis
5. Weighted aggregation: Critical phases contribute more to overall score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..core.models import ValidationIssue, ValidationSeverity


class ValidationPhase(Enum):
    """Validation phases in order of execution."""
    DOCKER_SYNTAX = "docker_syntax"
    DOCKER_LINTERS = "docker_linters"
    DOCKER_BUILD = "docker_build"
    K8S_SYNTAX = "k8s_syntax"
    K8S_LINTERS = "k8s_linters"
    KUBERNETES_APPLY = "kubernetes_apply"
    RUNTIME = "runtime"


@dataclass
class PhaseScore:
    """Score for a single validation phase."""
    phase: ValidationPhase
    base_score: float = 100.0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    error_penalty: float = 0.0
    warning_penalty: float = 0.0
    info_penalty: float = 0.0
    final_score: float = 100.0
    issues: List[ValidationIssue] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"PhaseScore({self.phase.value}: {self.final_score:.1f}, "
            f"errors={self.error_count}, warnings={self.warning_count}, info={self.info_count})"
        )


@dataclass
class ComponentScore:
    """Score for a component (Docker or Kubernetes)."""
    component_name: str
    phase_scores: List[PhaseScore] = field(default_factory=list)
    weighted_score: float = 0.0
    total_errors: int = 0
    total_warnings: int = 0
    total_info: int = 0

    def __repr__(self) -> str:
        return (
            f"ComponentScore({self.component_name}: {self.weighted_score:.1f}, "
            f"phases={len(self.phase_scores)}, "
            f"errors={self.total_errors}, warnings={self.total_warnings})"
        )


@dataclass
class AggregatedScore:
    """Complete aggregated score with all breakdowns."""
    overall_score: float
    docker_component: Optional[ComponentScore] = None
    k8s_component: Optional[ComponentScore] = None
    runtime_score: Optional[PhaseScore] = None
    total_errors: int = 0
    total_warnings: int = 0
    total_info: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "overall_score": round(self.overall_score, 2),
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "total_info": self.total_info,
        }

        if self.docker_component:
            result["docker"] = {
                "weighted_score": round(self.docker_component.weighted_score, 2),
                "total_errors": self.docker_component.total_errors,
                "total_warnings": self.docker_component.total_warnings,
                "total_info": self.docker_component.total_info,
                "phases": {
                    ps.phase.value: {
                        "score": round(ps.final_score, 2),
                        "errors": ps.error_count,
                        "warnings": ps.warning_count,
                        "info": ps.info_count,
                    }
                    for ps in self.docker_component.phase_scores
                },
            }

        if self.k8s_component:
            result["kubernetes"] = {
                "weighted_score": round(self.k8s_component.weighted_score, 2),
                "total_errors": self.k8s_component.total_errors,
                "total_warnings": self.k8s_component.total_warnings,
                "total_info": self.k8s_component.total_info,
                "phases": {
                    ps.phase.value: {
                        "score": round(ps.final_score, 2),
                        "errors": ps.error_count,
                        "warnings": ps.warning_count,
                        "info": ps.info_count,
                    }
                    for ps in self.k8s_component.phase_scores
                },
            }

        if self.runtime_score:
            result["runtime"] = {
                "score": round(self.runtime_score.final_score, 2),
                "errors": self.runtime_score.error_count,
                "warnings": self.runtime_score.warning_count,
                "info": self.runtime_score.info_count,
            }

        return result


class ScoringConfig:
    """
    Configuration for the scoring model with deterministic weights.

    All weights are constants to ensure reproducibility across runs.
    """

    # Severity penalties per issue (how much each issue type reduces the score)
    SEVERITY_PENALTIES = {
        ValidationSeverity.ERROR: 15.0,    # Errors are critical
        ValidationSeverity.WARNING: 10.0,   # Warnings carry meaningful risk
        ValidationSeverity.INFO: 0.0,      # Info issues do not affect score (informational only)
    }

    # Phase importance weights (how much each phase contributes to component score)
    # Syntax errors are most critical, build/apply errors are very important, linter issues less so
    PHASE_WEIGHTS = {
        # Docker phases
        ValidationPhase.DOCKER_SYNTAX: 0.40,    # Syntax must be correct
        ValidationPhase.DOCKER_BUILD: 0.40,     # Build must succeed
        ValidationPhase.DOCKER_LINTERS: 0.20,   # Linter best practices

        # Kubernetes phases
        ValidationPhase.K8S_SYNTAX: 0.35,       # Syntax must be correct
        ValidationPhase.KUBERNETES_APPLY: 0.40, # Deployment must succeed
        ValidationPhase.K8S_LINTERS: 0.25,      # Linter best practices
    }

    # Component weights for overall score
    COMPONENT_WEIGHTS = {
        "docker": 0.35,      # Docker configuration
        "kubernetes": 0.40,  # Kubernetes configuration (slightly more important)
        "runtime": 0.25,     # Runtime health (critical but depends on prior phases)
    }

    # Base score for each phase (starts at 100)
    BASE_PHASE_SCORE = 100.0

    # Minimum score floor (can't go below 0)
    MIN_SCORE = 0.0

    # Runtime scoring: binary with partial credit for warnings
    RUNTIME_SUCCESS_SCORE = 100.0
    RUNTIME_FAILURE_SCORE = 0.0
    RUNTIME_WARNING_PENALTY = 10.0  # Each warning reduces score


class IssueAggregationModel:
    """
    Aggregates validation issues into reproducible scores with phase breakdowns.

    This model:
    1. Groups issues by validation phase
    2. Calculates phase scores based on severity-weighted penalties
    3. Combines phase scores into component scores using phase weights
    4. Aggregates component scores into overall score using component weights
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def calculate_scores(
        self,
        step_issues: Dict[str, List[ValidationIssue]],
        runtime_success: Optional[bool] = None,
    ) -> AggregatedScore:
        """
        Calculate comprehensive scores from validation issues.

        Args:
            step_issues: Dictionary mapping step names to their validation issues
            runtime_success: Whether runtime validation succeeded (None if not run)

        Returns:
            AggregatedScore with full breakdown
        """
        # Calculate phase scores
        docker_phases = self._calculate_docker_scores(step_issues)
        k8s_phases = self._calculate_k8s_scores(step_issues)
        runtime_phase = self._calculate_runtime_score(step_issues, runtime_success)

        # Aggregate into component scores
        docker_component = self._aggregate_component_score("Docker", docker_phases, is_docker=True)
        k8s_component = self._aggregate_component_score("Kubernetes", k8s_phases, is_docker=False)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            docker_component,
            k8s_component,
            runtime_phase,
        )

        # Count total issues
        total_errors = 0
        total_warnings = 0
        total_info = 0

        for component in [docker_component, k8s_component]:
            if component:
                total_errors += component.total_errors
                total_warnings += component.total_warnings
                total_info += component.total_info

        if runtime_phase:
            total_errors += runtime_phase.error_count
            total_warnings += runtime_phase.warning_count
            total_info += runtime_phase.info_count

        return AggregatedScore(
            overall_score=overall_score,
            docker_component=docker_component,
            k8s_component=k8s_component,
            runtime_score=runtime_phase,
            total_errors=total_errors,
            total_warnings=total_warnings,
            total_info=total_info,
        )

    def _calculate_phase_score(
        self,
        phase: ValidationPhase,
        issues: List[ValidationIssue],
    ) -> PhaseScore:
        """Calculate score for a single validation phase."""
        score = PhaseScore(phase=phase, issues=issues)

        # Count issues by severity
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                score.error_count += 1
            elif issue.severity == ValidationSeverity.WARNING:
                score.warning_count += 1
            elif issue.severity == ValidationSeverity.INFO:
                score.info_count += 1

        # Calculate penalties
        score.error_penalty = score.error_count * self.config.SEVERITY_PENALTIES[ValidationSeverity.ERROR]
        score.warning_penalty = score.warning_count * self.config.SEVERITY_PENALTIES[ValidationSeverity.WARNING]
        score.info_penalty = score.info_count * self.config.SEVERITY_PENALTIES[ValidationSeverity.INFO]

        # Calculate final score
        total_penalty = score.error_penalty + score.warning_penalty + score.info_penalty
        score.final_score = max(
            self.config.MIN_SCORE,
            self.config.BASE_PHASE_SCORE - total_penalty
        )

        return score

    def _calculate_docker_scores(
        self,
        step_issues: Dict[str, List[ValidationIssue]],
    ) -> List[PhaseScore]:
        """Calculate scores for all Docker validation phases."""
        phases = []

        phase_mapping = {
            "docker_syntax": ValidationPhase.DOCKER_SYNTAX,
            "docker_linters": ValidationPhase.DOCKER_LINTERS,
            "docker_build": ValidationPhase.DOCKER_BUILD,
        }

        for step_name, phase_enum in phase_mapping.items():
            if step_name in step_issues:
                phase_score = self._calculate_phase_score(phase_enum, step_issues[step_name])
                phases.append(phase_score)

        return phases

    def _calculate_k8s_scores(
        self,
        step_issues: Dict[str, List[ValidationIssue]],
    ) -> List[PhaseScore]:
        """Calculate scores for all Kubernetes validation phases."""
        phases = []

        phase_mapping = {
            "k8s_syntax": ValidationPhase.K8S_SYNTAX,
            "k8s_linters": ValidationPhase.K8S_LINTERS,
            "kubernetes_apply": ValidationPhase.KUBERNETES_APPLY,
        }

        for step_name, phase_enum in phase_mapping.items():
            if step_name in step_issues:
                phase_score = self._calculate_phase_score(phase_enum, step_issues[step_name])
                phases.append(phase_score)

        return phases

    def _calculate_runtime_score(
        self,
        step_issues: Dict[str, List[ValidationIssue]],
        runtime_success: Optional[bool],
    ) -> Optional[PhaseScore]:
        """Calculate runtime validation score."""
        if runtime_success is None:
            return None

        issues = step_issues.get("runtime", [])
        score = PhaseScore(phase=ValidationPhase.RUNTIME, issues=issues)

        # Count issues
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                score.error_count += 1
            elif issue.severity == ValidationSeverity.WARNING:
                score.warning_count += 1
            elif issue.severity == ValidationSeverity.INFO:
                score.info_count += 1

        # Runtime is binary but warnings reduce score
        if runtime_success:
            score.final_score = self.config.RUNTIME_SUCCESS_SCORE
            # Apply warning penalty
            warning_penalty = score.warning_count * self.config.RUNTIME_WARNING_PENALTY
            score.final_score = max(self.config.MIN_SCORE, score.final_score - warning_penalty)
        else:
            score.final_score = self.config.RUNTIME_FAILURE_SCORE

        return score

    def _aggregate_component_score(
        self,
        component_name: str,
        phase_scores: List[PhaseScore],
        is_docker: bool,
    ) -> Optional[ComponentScore]:
        """Aggregate phase scores into a component score using weighted average."""
        if not phase_scores:
            return None

        component = ComponentScore(component_name=component_name, phase_scores=phase_scores)

        # Calculate weighted score
        total_weight = 0.0
        weighted_sum = 0.0

        for phase_score in phase_scores:
            weight = self.config.PHASE_WEIGHTS.get(phase_score.phase, 0.0)
            weighted_sum += phase_score.final_score * weight
            total_weight += weight

            # Accumulate issue counts
            component.total_errors += phase_score.error_count
            component.total_warnings += phase_score.warning_count
            component.total_info += phase_score.info_count

        # Normalize by total weight (in case not all phases were run)
        if total_weight > 0:
            component.weighted_score = weighted_sum / total_weight
        else:
            # Fallback to simple average if no weights defined
            component.weighted_score = sum(ps.final_score for ps in phase_scores) / len(phase_scores)

        return component

    def _calculate_overall_score(
        self,
        docker_component: Optional[ComponentScore],
        k8s_component: Optional[ComponentScore],
        runtime_phase: Optional[PhaseScore],
    ) -> float:
        """
        Calculate overall score from component scores using weighted average.

        Only components that were actually executed contribute to the score.
        Weights are normalized based on which components ran.
        """
        components = []

        if docker_component:
            components.append(("docker", docker_component.weighted_score))

        if k8s_component:
            components.append(("kubernetes", k8s_component.weighted_score))

        if runtime_phase:
            components.append(("runtime", runtime_phase.final_score))

        if not components:
            return 0.0

        # Calculate weighted score, normalizing weights
        total_weight = sum(self.config.COMPONENT_WEIGHTS[name] for name, _ in components)
        weighted_sum = sum(
            score * self.config.COMPONENT_WEIGHTS[name]
            for name, score in components
        )

        return weighted_sum / total_weight if total_weight > 0 else 0.0
