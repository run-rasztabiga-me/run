import logging
from typing import List, Tuple, Any
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ..core.config import GeneratorConfig
from ..core.repository import RepositoryManager
from ..tools.repository_tools import RepositoryTools
from .prompts import CONFIGURATION_AGENT_SYSTEM_PROMPT


class ConfigurationOutput(BaseModel):
    """Model for configuration generation output."""
    dockerfiles: List[str] = Field(description="List of generated Dockerfile paths")
    kubernetes_files: List[str] = Field(description="List of generated Kubernetes manifest paths")


class ConfigurationAgent:
    """Agent for generating Docker and Kubernetes configurations."""

    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self.logger = logging.getLogger(__name__)
        self.repository_manager = RepositoryManager()
        self.agent = None
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the LangGraph agent with tools and prompt."""
        # Create tools
        repo_tools = RepositoryTools(self.repository_manager)
        tools = repo_tools.create_tools()

        # Initialize LLM
        llm = init_chat_model(
            model=self.config.model_name,
            model_provider=self.config.model_provider,
            temperature=self.config.temperature,
        )

        # Create agent with structured output
        self.agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=self._get_system_message(),
            response_format=ConfigurationOutput
        )

    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return CONFIGURATION_AGENT_SYSTEM_PROMPT.format(
            default_replicas=self.config.default_replicas,
            domain_suffix=self.config.domain_suffix
        )

    def generate_configurations(self, repo_url: str) -> Tuple[ConfigurationOutput, List]:
        """
        Generate configurations for a given repository URL.

        Args:
            repo_url: URL of the repository to process

        Returns:
            Tuple of (structured output with generated files information, messages list)
        """
        self.logger.info(f"Starting configuration generation for: {repo_url}")

        # Run agent to generate files - response_format ensures structured output
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": repo_url}]},
            {"recursion_limit": self.config.recursion_limit}
        )

        # Extract structured response and return messages
        return result['structured_response'], result['messages']

    def get_repository_manager(self) -> RepositoryManager:
        """Get the repository manager instance."""
        return self.repository_manager

    def update_config(self, new_config: GeneratorConfig) -> None:
        """Update configuration and reinitialize agent."""
        self.config = new_config
        self._initialize_agent()
