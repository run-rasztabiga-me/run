import logging
import os
import uuid
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ..core.config import GeneratorConfig
from ..core.workspace import RepositoryWorkspace
from ..tools.repository_tools import RepositoryTools
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

    # LangChain's init_chat_model known providers
    KNOWN_PROVIDERS = {
        "openai", "anthropic", "azure_openai", "google_genai", "google_vertexai",
        "bedrock", "bedrock_converse", "cohere", "fireworks", "together",
        "mistralai", "huggingface", "groq", "deepseek"
    }

    def _prepare_llm_kwargs(self) -> Dict[str, Any]:
        """
        Prepare LLM kwargs for init_chat_model, automatically routing unknown providers through OpenRouter.

        If the provider is not in the known providers list, it will be routed through OpenRouter
        using the OpenAI-compatible API with a custom base URL.

        Model name format: provider="meta-llama", name="llama-4-scout" -> "meta-llama/llama-4-scout"
        """
        provider = self.config.model_provider
        model_name = self.config.model_name

        # Check if provider is unknown and should be routed through OpenRouter
        if provider not in self.KNOWN_PROVIDERS:
            self.logger.info(
                f"Provider '{provider}' not in known providers list. Routing through OpenRouter."
            )

            # Format model name as provider/model for OpenRouter
            openrouter_model = f"{provider}/{model_name}"

            # Get OpenRouter API key from environment variable
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    f"Unknown provider '{provider}' requires OpenRouter. "
                    "Please set OPENROUTER_API_KEY environment variable."
                )

            llm_kwargs = {
                "model": openrouter_model,
                "model_provider": "openai",  # Use OpenAI provider for OpenRouter
                "temperature": self.config.temperature,
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": api_key,
            }

            self.logger.info(f"Using OpenRouter with model: {openrouter_model}")
        else:
            # Use provider as-is for known providers
            llm_kwargs = {
                "model": model_name,
                "model_provider": provider,
                "temperature": self.config.temperature,
            }

            # Add seed if configured (only for OpenAI models - Anthropic doesn't support it)
            if self.config.seed is not None and provider == "openai":
                llm_kwargs["seed"] = self.config.seed

        return llm_kwargs

    def _initialize_agent(self) -> None:
        """Initialize the LangGraph agent with tools and prompt."""
        if not self.workspace:
            raise ValueError("Workspace must be set before initializing agent")

        # Create tools
        repo_tools = RepositoryTools(self.workspace)
        tools = repo_tools.create_tools()

        # Initialize LLM
        llm_kwargs = self._prepare_llm_kwargs()
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
                    default_replicas=self.config.default_replicas,
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
                default_replicas=self.config.default_replicas,
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
