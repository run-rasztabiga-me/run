from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


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
    generated_files: List[str] = field(default_factory=list)
    dockerfile_path: Optional[str] = None
    k8s_manifests_path: Optional[str] = None
    generation_time: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationIssue:
    """A validation issue found during evaluation."""
    file_path: str
    line_number: Optional[int]
    severity: ValidationSeverity
    message: str
    rule_id: Optional[str] = None
    category: Optional[str] = None


@dataclass
class ExecutionMetrics:
    """Metrics collected during agent execution."""
    total_time: float
    tool_calls_count: int
    tool_calls_breakdown: Dict[str, int] = field(default_factory=dict)
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error_count: int = 0
    retry_count: int = 0
    # Thesis-specific metrics
    run_number: Optional[int] = None
    success_detected_via_done: bool = False


@dataclass
class QualityMetrics:
    """Quality metrics for generated configurations."""
    dockerfile_score: Optional[float] = None
    k8s_manifests_score: Optional[float] = None
    overall_score: Optional[float] = None
    best_practices_violations: int = 0
    security_issues: int = 0
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
    # Multiple run support
    run_results: List['EvaluationReport'] = field(default_factory=list)

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