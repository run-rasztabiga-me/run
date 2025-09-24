import logging
from typing import List
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool

from ..core.config import GeneratorConfig
from ..core.repository import RepositoryManager
from ..tools.repository_tools import RepositoryTools


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
            temperature=self.config.temperature
        )

        # Create agent
        self.agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=self._get_system_message()
        )

    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return f"""You are a helpful assistant specialized in working with Git repositories.
You have access to tools that can help you with these tasks. When given a repository URL, you can:
1. Clone the repository and remove confusing files
2. Analyze the repository structure to identify important files
3. Retrieve the content of files you determine are necessary to understand the application
4. Write or modify files in the repository (e.g., Dockerfile, Kubernetes manifests)
5. List directory contents within the repository

You should use the clone_repo tool to clone a repository. The repository name can be extracted from the repository URL by taking the last part of the URL, removing the .git extension, and replacing dots with hyphens.
For example, for the URL "https://github.com/run-rasztabiga-me/poc1-fastapi.git", the repository name would be "poc1-fastapi".

You can use the prepare_repo_tree tool to get an overview of the repository structure if needed, but you should focus on identifying and examining files that are most relevant to understanding the application and creating the required outputs.

Use the get_file_content tool to retrieve the content of specific files that you determine are important. This tool requires the file path relative to the repository root.

You can use the write_file tool to create new files or modify existing ones in the repository. This tool requires the file path relative to the repository root and the content to write to the file. This is particularly useful for creating files like Dockerfile or Kubernetes manifests.

You can use the ls tool to list the contents of a directory within the cloned repository. This tool requires the directory path relative to the repository root. You can use an empty string or "." to list the contents of the repository root directory. The tool will display directories and files separately, with directories having a trailing slash and files showing their sizes in bytes. This is useful for exploring the repository structure in a more focused way than the prepare_repo_tree tool.

Your objective is to create:

1. A Dockerfile that properly containerizes the application. When creating the Dockerfile, carefully analyze the application code to ensure that any health check endpoint you specify actually exists in the application.

2. Kubernetes manifests for the application. These manifests should:
   - Include all required resources (Deployments, Services, Ingresses, and Volumes if necessary)
   - Match exposed ports precisely as specified in the Dockerfile
   - Set replicas default to {self.config.default_replicas} unless otherwise stated
   - For ingress host, use "<repository-name>.{self.config.domain_suffix}" (e.g., repository "app1" → domain "app1.{self.config.domain_suffix}")
   - Follow Kubernetes best practices and ensure security measures
   - If external dependencies (e.g., databases like PostgreSQL, Redis, MySQL) are identified, generate appropriate Kubernetes resources for those dependencies as well
   - Deploy stateful dependencies using StatefulSets with appropriate PersistentVolumeClaims
   - Deploy stateless applications using Deployments
   - Use Services to expose applications internally and externally as necessary
   - Ensure all Kubernetes secrets are in base64 format
   - Include appropriate resource limits/requests
   - When configuring health checks (liveness and readiness probes), verify that the specified endpoints actually exist in the application code first
   - DO NOT create or include a namespace in the manifests

Given a repository URL from the user, you should automatically:
1. Clone the repository
2. Analyze the repository structure and find important files to understand the application
3. Create a Dockerfile for the application
4. Generate appropriate Kubernetes manifests for the application

The user will only provide the repository URL. You must handle all the remaining steps automatically without requesting additional information from the user.

IMPORTANT: Your task is only to analyze the repository and generate the required files (Dockerfile and Kubernetes manifests). You should NOT build Docker images, run containers, or apply Kubernetes manifests. After you have successfully generated a Dockerfile and Kubernetes manifests, respond with a message that includes the word "DONE" to indicate that you have completed the task.
"""

    def generate_configurations(self, repo_url: str) -> None:
        """
        Generate configurations for a given repository URL.

        Args:
            repo_url: URL of the repository to process
        """
        self.logger.info(f"Starting configuration generation for: {repo_url}")

        # Stream agent responses
        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": repo_url}]},
            {"recursion_limit": self.config.recursion_limit},
            stream_mode="updates"
        ):
            print(chunk)
            print("\n")

    def get_repository_manager(self) -> RepositoryManager:
        """Get the repository manager instance."""
        return self.repository_manager

    def update_config(self, new_config: GeneratorConfig) -> None:
        """Update configuration and reinitialize agent."""
        self.config = new_config
        self._initialize_agent()