import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class GeneratorConfig(BaseModel):
    """Configuration settings for the configuration generator."""

    # Docker settings
    docker_registry: str = Field(default_factory=lambda: os.environ.get('REGISTRY_URL', '192.168.0.124:32000'))
    default_image_tag: str = "latest"

    # Kubernetes settings
    domain_suffix: str = "rasztabiga.me"
    default_replicas: int = 1

    # LLM settings
    model_name: str = Field(default="gpt-5-nano")
    model_provider: str = Field(default="openai")
    temperature: float = Field(default=0.7)
    seed: Optional[int] = Field(default=42)

    # Agent settings
    recursion_limit: int = 100

    # Directory settings
    tmp_dir_base: str = "./tmp"

    # Timeout settings
    docker_start_timeout: int = 5
    k8s_ingress_timeout: int = 5

    def get_full_image_name(self, repo_name: str, image_tag: str, version: str = None) -> str:
        """
        Generate full image name with registry and tag.

        Args:
            repo_name: Name of the repository
            image_tag: Tag identifying the image role (e.g., 'frontend', 'backend')
            version: Version tag (defaults to default_image_tag if not provided)

        Returns:
            Full image name in format: {registry}/{repo_name}-{image_tag}:{version}
        """
        version = version or self.default_image_tag
        return f"{self.docker_registry}/{repo_name}-{image_tag}:{version}"

    def get_ingress_host(self, repo_name: str) -> str:
        """Generate ingress hostname."""
        return f"{repo_name}.{self.domain_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.dict()


class ToolSchemas:
    """Pydantic schemas for tool inputs."""

    class CloneRepoInput(BaseModel):
        repo_url: str = Field(description="URL of the repository to clone")

    class PrepareRepoTreeInput(BaseModel):
        pass

    class GetFileContentInput(BaseModel):
        file_path: str = Field(description="Path to the file relative to the repository root")

    class WriteFileInput(BaseModel):
        file_path: str = Field(description="Path to the file relative to the repository root")
        content: str = Field(description="Content to write to the file")

    class ListDirectoryInput(BaseModel):
        dir_path: str = Field(description="Path to the directory relative to the repository root")
