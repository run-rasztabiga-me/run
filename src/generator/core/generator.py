import logging
import uuid
from datetime import datetime, UTC
from typing import Optional, Tuple, List
from dotenv import load_dotenv

from .config import GeneratorConfig
from .workspace import RepositoryWorkspace
from .workspace_models import RunContext
from ..agents.config_agent import ConfigurationAgent, ConfigurationOutput
from ...utils.repository_utils import extract_repo_name


class ConfigurationGenerator:
    """Main class for generating Docker and Kubernetes configurations."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        # Load environment variables
        load_dotenv()

        # Setup logging
        self._setup_logging()

        self.logger = logging.getLogger(__name__)
        self.config = config or GeneratorConfig()
        # Agent will be created per-generation with workspace
        self.agent = None

    def _setup_logging(self) -> None:
        """Configure logging settings."""
        # Don't reconfigure logging - use the configuration from main entry point
        # This avoids conflicts with evaluator.py's logging setup
        pass

    def generate(self, repo_url: str) -> Tuple[ConfigurationOutput, List, str, RunContext, RepositoryWorkspace]:
        """
        Generate configurations for the given repository URL.

        Args:
            repo_url: URL of the Git repository to process

        Returns:
            Tuple of (structured output, messages list, run_id, run_context, workspace)
        """
        self.logger.info(f"Starting configuration generation for repository: {repo_url}")

        # Generate unique run ID and create run context
        run_id = str(uuid.uuid4())
        repo_name = extract_repo_name(repo_url)
        run_context = RunContext(
            run_id=run_id,
            repo_name=repo_name,
            timestamp=datetime.now(UTC)
        )

        self.logger.info(f"Run ID: {run_id}, Workspace: {run_context.workspace_dir}")

        # Create workspace for this run
        workspace = RepositoryWorkspace(run_context)

        # Create agent with workspace
        agent = ConfigurationAgent(self.config, workspace)

        # Generate configurations using the same run_id for LangSmith tracing
        config_output, messages, langsmith_run_id = agent.generate_configurations(repo_url, run_id=run_id)

        return config_output, messages, run_id, run_context, workspace

    def update_config(self, new_config: GeneratorConfig) -> None:
        """Update the configuration."""
        self.config = new_config

    def get_config(self) -> GeneratorConfig:
        """Get the current configuration."""
        return self.config