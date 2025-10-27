from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from ...common.models import DockerImageInfo, DockerBuildMetrics


class EvaluationStatus(Enum):
    """Evaluation status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class GenerationResult:
    """Result of configuration generation."""
    repo_url: str
    repo_name: str
    success: bool
    docker_images: List[DockerImageInfo] = field(default_factory=list)
    k8s_manifests: List[str] = field(default_factory=list)
    test_endpoint: Optional[str] = None
    generation_time: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    run_context: Optional['RunContext'] = None  # Run context with workspace info

    @property
    def dockerfiles(self) -> List[str]:
        """Backward compatibility - returns list of dockerfile paths."""
        return [img.dockerfile_path for img in self.docker_images]


@dataclass
class ValidationIssue:
    """A validation issue found during evaluation."""
    file_path: str
    line_number: Optional[int]
    severity: ValidationSeverity
    message: str
    rule_id: Optional[str] = None


@dataclass
class ExecutionMetrics:
    """Metrics collected during agent execution."""
    total_time: float = 0.0
    tool_calls_count: int = 0
    tool_calls_breakdown: Dict[str, int] = field(default_factory=dict)
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error_count: int = 0
    # LangSmith trace identifier
    run_id: Optional[str] = None
    # Docker build metrics
    docker_build_metrics: List['DockerBuildMetrics'] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Quality metrics for generated configurations."""
    dockerfile_score: Optional[float] = None
    k8s_manifests_score: Optional[float] = None
    overall_score: Optional[float] = None
    validation_issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    repo_url: str
    repo_name: str
    evaluation_id: str
    status: EvaluationStatus
    generation_result: Optional[GenerationResult] = None
    execution_metrics: Optional[ExecutionMetrics] = None
    quality_metrics: Optional[QualityMetrics] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_evaluation_time: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    experiment_name: Optional[str] = None
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    repetition_index: Optional[int] = None
    prompt_id: Optional[str] = None
    prompt_override: Optional[str] = None
    build_success: Optional[bool] = None
    runtime_success: Optional[bool] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Calculate evaluation duration."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def add_note(self, note: str) -> None:
        """Add a note to the evaluation."""
        self.notes.append(f"{datetime.now().isoformat()}: {note}")

    def mark_completed(self) -> None:
        """Mark evaluation as completed."""
        self.status = EvaluationStatus.COMPLETED
        self.end_time = datetime.now()
        self.total_evaluation_time = self.duration

    def mark_failed(self, error_message: str) -> None:
        """Mark evaluation as failed."""
        self.status = EvaluationStatus.FAILED
        self.end_time = datetime.now()
        self.total_evaluation_time = self.duration
        self.add_note(f"Evaluation failed: {error_message}")
