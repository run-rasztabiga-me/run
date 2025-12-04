import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class GeneratorConfig(BaseModel):
    """Configuration settings for the configuration generator."""

    # Docker settings
    docker_registry: str = Field(default_factory=lambda: os.environ.get('REGISTRY_URL', '192.168.0.124:32000'))
    default_image_tag: str = "latest"

    # Kubernetes settings
    k8s_cluster_ip: str = Field(default_factory=lambda: os.environ.get('K8S_CLUSTER_IP', '192.168.0.124'))
    domain_suffix: str = "rasztabiga.me"
    default_replicas: int = 1

    # LLM settings
    model_name: str = Field(default="gpt-5-nano")
    model_provider: str = Field(default="openai")
    temperature: float = Field(default=None)
    seed: Optional[int] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None, description="Override system prompt for configuration agent.")
    prompt_version: Optional[str] = Field(default=None, description="Identifier for the prompt variant used.")

    # LLM judge settings
    enable_llm_judge: bool = Field(default=False, description="Enable LLM-as-a-judge validation steps during evaluator runs.")
    llm_judge_model_name: Optional[str] = Field(default=None, description="Override model name for LLM-as-a-judge evaluations.")
    llm_judge_model_provider: Optional[str] = Field(default=None, description="Override provider for LLM-as-a-judge evaluations.")
    llm_judge_temperature: float = Field(default=0.0, description="Sampling temperature for LLM-as-a-judge calls (defaults to 0.0 for determinism).")

    # Agent settings
    recursion_limit: int = 200

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
            image_tag: Tag identifying the image role (e.g., 'frontend', 'backend', or 'frontend:latest')
            version: Version tag (defaults to default_image_tag if not provided)

        Returns:
            Full image name in format: {registry}/{repo_name}-{image_tag}:{version}
        """
        # Strip any existing version tag from image_tag if present
        # (LLM sometimes generates 'backend:latest' instead of just 'backend')
        if ':' in image_tag:
            image_tag_base, existing_version = image_tag.split(':', 1)
            # Use the existing version if no version was explicitly provided
            if version is None:
                version = existing_version
            image_tag = image_tag_base

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

    class SearchFilesInput(BaseModel):
        pattern: str = Field(description="Text pattern or regex to search for in file contents")
        file_pattern: Optional[str] = Field(default=None, description="Optional glob pattern to filter files (e.g., '*.py', '*.yaml')")
        case_sensitive: bool = Field(default=False, description="Whether the search should be case-sensitive")

    class Base64EncodeInput(BaseModel):
        content: str = Field(description="Plain text content to encode to base64")

    class Base64DecodeInput(BaseModel):
        encoded: str = Field(description="Base64-encoded string to decode to plain text")

    class FindFilesInput(BaseModel):
        pattern: str = Field(description="Filename pattern to search for (supports glob patterns like '*.py', 'Dockerfile*', 'package.json')")
        max_results: int = Field(default=50, description="Maximum number of results to return")

    class PatchFileInput(BaseModel):
        file_path: str = Field(description="Path to the file to patch, relative to the repository root")
        patch: str = Field(description="Unified diff patch to apply (standard diff format with @@ line numbers)")

    class ThinkInput(BaseModel):
        thoughts: str = Field(description="Your internal thoughts, observations, analysis, or planning notes about the repository and task at hand")
