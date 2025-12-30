import logging
import time
import uuid
from pathlib import Path
from typing import List, Tuple, Optional
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ..core.config import GeneratorConfig
from ..core.workspace import RepositoryWorkspace
from ..tools.repository_tools import RepositoryTools
from ...common.models import DockerImageInfo
from ..utils.llm import build_llm_kwargs


class ConfigurationOutput(BaseModel):
    """Model for configuration generation output."""
    docker_images: List[DockerImageInfo] = Field(description="List of Docker images to build with their metadata")
    kubernetes_files: List[str] = Field(description="List of generated Kubernetes manifest paths")
    test_endpoints: List[str] = Field(description="List of relative paths to test endpoints that should return 2xx status (e.g., ['/', '/health', '/api/health'])")

    @property
    def dockerfiles(self) -> List[str]:
        """Backward compatibility - returns list of dockerfile paths."""
        return [img.dockerfile_path for img in self.docker_images]


class ConfigurationAgent:
    """Agent for generating Docker and Kubernetes configurations."""

    def __init__(self, config: GeneratorConfig = None, workspace: RepositoryWorkspace = None):
        self.config = config or GeneratorConfig()
        self.logger = logging.getLogger(__name__)
        self.workspace = workspace
        self.agent = None
        if workspace:
            self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the LangGraph agent with tools and prompt."""
        if not self.workspace:
            raise ValueError("Workspace must be set before initializing agent")

        # Create tools
        repo_tools = RepositoryTools(self.workspace)
        tools = repo_tools.create_tools()

        # Initialize LLM
        llm_kwargs = build_llm_kwargs(self.config, logger=self.logger)
        llm = init_chat_model(**llm_kwargs)

        # Create agent with structured output
        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=self._get_system_message(),
            response_format=ConfigurationOutput
        )

    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        if self.config.system_prompt:
            try:
                return self.config.system_prompt.format(
                    domain_suffix=self.config.domain_suffix,
                )
            except KeyError:
                return self.config.system_prompt

        # Load from prompts/default.prompt
        default_prompt_path = Path("prompts/default.prompt")
        if not default_prompt_path.exists():
            raise FileNotFoundError(
                f"Default prompt file not found at {default_prompt_path}. "
                "Please create prompts/default.prompt or provide system_prompt in config."
            )

        prompt_text = default_prompt_path.read_text(encoding="utf-8")
        try:
            return prompt_text.format(
                domain_suffix=self.config.domain_suffix,
            )
        except KeyError:
            return prompt_text

    def generate_configurations(self, repo_url: str, run_id: Optional[str] = None) -> Tuple[ConfigurationOutput, List, str]:
        """
        Generate configurations for a given repository URL.

        Args:
            repo_url: URL of the repository to process
            run_id: Optional run ID to use (if not provided, generates a new one)

        Returns:
            Tuple of (structured output with generated files information, messages list, run_id)
        """
        self.logger.info(f"Starting configuration generation for: {repo_url}")
        prompt_version = self.config.prompt_version or "default"
        self.logger.info("Using prompt version: %s", prompt_version)

        # Use provided run_id or generate a new one for LangSmith tracing
        if run_id is None:
            run_id = str(uuid.uuid4())

        # Retry configuration
        max_retries = 5
        base_delay = 30.0  # Start with 30 seconds

        for attempt in range(max_retries):
            try:
                # Run agent to generate files - response_format ensures structured output
                result = self.agent.invoke(
                    {"messages": [{"role": "user", "content": repo_url}]},
                    {
                        "recursion_limit": self.config.recursion_limit,
                        "run_id": run_id,
                    }
                )

                # Extract structured response and return messages with run_id
                return result['structured_response'], result['messages'], run_id

            except Exception as e:
                # Check if it's a rate limit error
                is_rate_limit = (
                    "RateLimitError" in str(type(e).__name__) or
                    "rate_limit" in str(e).lower() or
                    "429" in str(e)
                )

                if is_rate_limit and attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.1f} seconds... Error: {str(e)}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Not a rate limit error or max retries reached
                    self.logger.error(f"Failed to generate configurations: {str(e)}", exc_info=True)
                    raise

    def get_workspace(self) -> RepositoryWorkspace:
        """Get the workspace instance."""
        return self.workspace

    def set_workspace(self, workspace: RepositoryWorkspace) -> None:
        """Set workspace and initialize agent."""
        self.workspace = workspace
        self._initialize_agent()

    def update_config(self, new_config: GeneratorConfig) -> None:
        """Update configuration and reinitialize agent."""
        self.config = new_config
        if self.workspace:
            self._initialize_agent()
