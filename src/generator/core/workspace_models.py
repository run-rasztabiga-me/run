"""Models for repository workspace and typed file operation results."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List


@dataclass
class RunContext:
    """
    Immutable context identifying a specific experiment run.

    Provides run-scoped resource naming to enable parallel execution
    without conflicts (namespaces, workspaces, image tags, etc.).

    PARALLEL EXECUTION GUARANTEE:
    - Each run_id is a unique UUID generated per ConfigurationGenerator.generate() call
    - All resource names derived from run_id are guaranteed unique
    - Thread-safe: immutable dataclass with no shared state
    - Zero conflicts between parallel runs of the same or different repositories
    """
    run_id: str  # UUID string - guaranteed unique per run
    repo_name: str
    timestamp: datetime

    @property
    def workspace_dir(self) -> Path:
        """
        Unique workspace directory for this run.
        Format: ./tmp/{repo_name}-{run_id}

        PARALLEL SAFETY: Different runs get different directories even for same repo
        Example: ./tmp/my-app-550e8400... and ./tmp/my-app-660f9511...
        """
        return Path(f"./tmp/{self.repo_name}-{self.run_id}")

    @property
    def k8s_namespace(self) -> str:
        """
        Unique Kubernetes namespace for this run.
        Format: {repo-name}-{run_id}
        Sanitized for K8s naming requirements (lowercase, hyphens).

        PARALLEL SAFETY: Prevents namespace collisions in K8s cluster
        Example: my-app-550e8400-e29b-41d4-a716-446655440000
        """
        sanitized = self.repo_name.lower().replace('_', '-').replace('.', '-')
        return f"{sanitized}-{self.run_id}"

    def get_image_tag(self, image_name: str) -> str:
        """
        Generate unique Docker image tag for this run.
        Format: {image_name}:{run_id}

        PARALLEL SAFETY: Prevents image tag overwrites in registry
        Example: backend:550e8400-e29b-41d4-a716-446655440000

        Args:
            image_name: Base image name (e.g., 'frontend', 'backend')

        Returns:
            Full image tag with run-specific suffix
        """
        return f"{image_name}:{self.run_id}"


@dataclass
class CloneResult:
    """Result of repository clone operation."""
    success: bool
    repo_path: Optional[Path] = None
    repo_name: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FileReadResult:
    """Result of file read operation."""
    success: bool
    content: Optional[str] = None
    path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class FileWriteResult:
    """Result of file write operation."""
    success: bool
    action: Optional[str] = None  # "created" | "modified"
    path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class DirectoryItem:
    """Single item in a directory listing."""
    name: str
    is_directory: bool
    size_bytes: Optional[int] = None


@dataclass
class DirectoryListResult:
    """Result of directory listing operation."""
    success: bool
    items: List[DirectoryItem] = field(default_factory=list)
    path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class TreeResult:
    """Result of directory tree generation."""
    success: bool
    tree_structure: Optional[str] = None
    error: Optional[str] = None
