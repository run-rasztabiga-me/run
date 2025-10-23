import logging
import uuid
from typing import List, Tuple, Any, Optional
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ..core.config import GeneratorConfig
from ..core.workspace import RepositoryWorkspace
from ..tools.repository_tools import RepositoryTools
from .prompts import CONFIGURATION_AGENT_SYSTEM_PROMPT
from ...common.models import DockerImageInfo


class ConfigurationOutput(BaseModel):
    """Model for configuration generation output."""
    docker_images: List[DockerImageInfo] = Field(description="List of Docker images to build with their metadata")
    kubernetes_files: List[str] = Field(description="List of generated Kubernetes manifest paths")
    test_endpoint: str = Field(description="Relative path to a test endpoint that should return 2xx status (e.g., '/', '/health', '/api/health')")

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
        llm_kwargs = {
            "model": self.config.model_name,
            "model_provider": self.config.model_provider,
            "temperature": self.config.temperature,
        }

        # Add seed if configured (only for OpenAI models - Anthropic doesn't support it)
        if self.config.seed is not None and self.config.model_provider == "openai":
            llm_kwargs["seed"] = self.config.seed

        llm = init_chat_model(**llm_kwargs)

        # Create agent with structured output
        self.agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=self._get_system_message(),
            response_format=ConfigurationOutput
        )

    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        if self.config.system_prompt:
            try:
                return self.config.system_prompt.format(
                    default_replicas=self.config.default_replicas,
                    domain_suffix=self.config.domain_suffix,
                )
            except KeyError:
                return self.config.system_prompt

        return CONFIGURATION_AGENT_SYSTEM_PROMPT.format(
            default_replicas=self.config.default_replicas,
            domain_suffix=self.config.domain_suffix
        )

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

        try:
            # Run agent to generate files - response_format ensures structured output
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": repo_url}]},
                {
                    "recursion_limit": self.config.recursion_limit,
                    "run_id": run_id,
                    "tags": [] # TODO?
                }
            )

            # Extract structured response and return messages with run_id
            return result['structured_response'], result['messages'], run_id
        except Exception as e:
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
