"""Shared data models."""
from pydantic import BaseModel, Field


class DockerImageInfo(BaseModel):
    """Information about a Docker image to build."""
    dockerfile_path: str = Field(description="Path to Dockerfile relative to repository root")
    image_tag: str = Field(description="Tag for the Docker image (e.g., 'frontend', 'backend', 'api')")
    build_context: str = Field(default=".", description="Build context path relative to repository root")
