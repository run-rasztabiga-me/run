import logging
from typing import Optional, Tuple, List
from dotenv import load_dotenv

from .config import GeneratorConfig
from ..agents.config_agent import ConfigurationAgent, ConfigurationOutput


class ConfigurationGenerator:
    """Main class for generating Docker and Kubernetes configurations."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        # Load environment variables
        load_dotenv()

        # Setup logging
        self._setup_logging()

        self.logger = logging.getLogger(__name__)
        self.config = config or GeneratorConfig()
        self.agent = ConfigurationAgent(self.config)

    def _setup_logging(self) -> None:
        """Configure logging settings."""
        # Don't reconfigure logging - use the configuration from main entry point
        # This avoids conflicts with evaluator.py's logging setup
        pass

    def generate(self, repo_url: str) -> Tuple[ConfigurationOutput, List, str]:
        """
        Generate configurations for the given repository URL.

        Args:
            repo_url: URL of the Git repository to process

        Returns:
            Tuple of (structured output with generated file paths, messages list, run_id)
        """
        self.logger.info(f"Starting configuration generation for repository: {repo_url}")
        return self.agent.generate_configurations(repo_url)

    def get_repository_manager(self):
        """Get the repository manager from the agent."""
        return self.agent.get_repository_manager()

    def update_config(self, new_config: GeneratorConfig) -> None:
        """Update the configuration."""
        self.config = new_config
        self.agent.update_config(new_config)

    def get_config(self) -> GeneratorConfig:
        """Get the current configuration."""
        return self.config