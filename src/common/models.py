"""Shared data models used across generator, evaluator, and runtime toolkits."""
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


class DockerImageInfo(BaseModel):
    """Information about a Docker image to build."""

    dockerfile_path: str = Field(description="Path to Dockerfile relative to repository root")
    image_tag: str = Field(description="Tag for the Docker image (e.g., 'frontend', 'backend', 'api')")
    build_context: str = Field(default=".", description="Build context path relative to repository root")


@dataclass
class DockerBuildMetrics:
    """Metrics collected during Docker image build."""

    image_tag: str
    build_time: float  # seconds
    image_size_mb: float  # megabytes
    layers_count: int  # number of layers
    image_name: Optional[str] = None  # Optional for registry reference
