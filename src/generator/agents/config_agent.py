import logging
from typing import List
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool

from ..core.config import GeneratorConfig
from ..core.repository import RepositoryManager
from ..tools.repository_tools import RepositoryTools
from .prompts import CONFIGURATION_AGENT_SYSTEM_PROMPT


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
            # use_responses_api=True, # TODO experimental, it messes up the output (different format)
        )

        # Create agent
        self.agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=self._get_system_message()
        )

    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return CONFIGURATION_AGENT_SYSTEM_PROMPT.format(
            default_replicas=self.config.default_replicas,
            domain_suffix=self.config.domain_suffix
        )

    def generate_configurations(self, repo_url: str) -> str:
        """
        Generate configurations for a given repository URL.

        Args:
            repo_url: URL of the repository to process

        Returns:
            Complete agent output as string for parsing
        """
        self.logger.info(f"Starting configuration generation for: {repo_url}")

        agent_output = []

        # Stream agent responses and collect output
        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": repo_url}]},
            {"recursion_limit": self.config.recursion_limit},
            stream_mode="updates"
        ):
            self.logger.debug(f"Agent chunk: {chunk}")
            agent_output.append(str(chunk))

        return "\n".join(agent_output)

    def get_repository_manager(self) -> RepositoryManager:
        """Get the repository manager instance."""
        return self.repository_manager

    def update_config(self, new_config: GeneratorConfig) -> None:
        """Update configuration and reinitialize agent."""
        self.config = new_config
        self._initialize_agent()
