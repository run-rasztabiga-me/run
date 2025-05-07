import asyncio
import logging
import os
import shutil
from typing import Dict, Generator, List, Optional, Tuple

from dotenv import load_dotenv
from git import Repo
from pydantic import BaseModel, Field

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import BaseTool
from autogen_ext.models.openai import OpenAIChatCompletionClient, _model_info

# Setup
load_dotenv()
logger = logging.getLogger(__name__)

# Constants
CONFUSING_FILES = {".git", ".github", ".gitignore", ".gitmodules", ".gitattributes", 
                  ".gitlab-ci.yml", ".travis.yml", "LICENSE", "README.md", "CHANGELOG.md",
                  "__pycache__", ".pytest_cache", ".coverage", "htmlcov", ".idea", ".vscode"}

# Define a model client
model_info = _model_info.get_info("gpt-4o")
model_client = OpenAIChatCompletionClient(
    model="openai/gpt-4o",
    model_info=model_info,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Define tool input/output models
class CloneRepoInput(BaseModel):
    repo_url: str = Field(..., description="URL of the repository to clone")

class PrepareRepoTreeInput(BaseModel):
    repo_name: str = Field(..., description="Name of the repository to prepare tree for")

# Define tools
class CloneRepoTool(BaseTool[CloneRepoInput, str]):
    name: str = "clone_repo"
    description: str = "Clone repository and recursively remove confusing files"
    args_schema: type[BaseModel] = CloneRepoInput

    def __init__(self):
        super().__init__(
            args_type=CloneRepoInput,
            return_type=str,
            name=self.name,
            description=self.description
        )

    async def run(self, args: CloneRepoInput, cancellation_token=None) -> str:
        """Clone repository and recursively remove confusing files."""
        repo_name = args.repo_url.split("/")[-1].replace(".git", "").replace(".", "-")
        tmp_dir = f"./tmp/{repo_name}"

        logger.info("Preparing working directory...")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        logger.info("Cloning repository...")
        Repo.clone_from(args.repo_url, tmp_dir)

        # Recursively search and remove confusing files
        for root, dirs, files in os.walk(tmp_dir, topdown=True):
            # Check files in current directory
            for file in files:
                if file in CONFUSING_FILES:
                    file_path = os.path.join(root, file)
                    logger.info(f"Removing confusing file: {file_path}")
                    os.remove(file_path)

            # Check directories in current directory
            for dir_name in list(dirs):  # Create a copy of dirs to modify during iteration
                if dir_name in CONFUSING_FILES:
                    dir_path = os.path.join(root, dir_name)
                    logger.info(f"Removing confusing directory: {dir_path}")
                    shutil.rmtree(dir_path, ignore_errors=True)
                    dirs.remove(dir_name)  # Remove from dirs to prevent further traversal

        return f"Repository {args.repo_url} cloned successfully to {tmp_dir}"


class PrepareRepoTreeTool(BaseTool[PrepareRepoTreeInput, str]):
    name: str = "prepare_repo_tree"
    description: str = "Prepare repository tree structure as a string"
    args_schema: type[BaseModel] = PrepareRepoTreeInput

    def __init__(self):
        super().__init__(
            args_type=PrepareRepoTreeInput,
            return_type=str,
            name=self.name,
            description=self.description
        )

    async def run(self, args: PrepareRepoTreeInput, cancellation_token=None) -> str:
        """Prepare repository tree structure as a string."""
        tmp_dir = f"./tmp/{args.repo_name}"

        if not os.path.isdir(tmp_dir):
            return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

        logger.info("Preparing tree...")
        dir_tree = self._tree(tmp_dir)
        return self._tree_to_str(dir_tree, trim_dir=tmp_dir)

    def _tree(self, some_dir: str) -> Generator[
        Tuple[str, List[str], List[Tuple[str, int]]], None, None]:
        """
        Generate tree structure of directory.

        Args:
            some_dir: Path to the directory

        Yields:
            Tuple of (root, dirs, files_with_sizes)
        """
        assert os.path.isdir(some_dir)

        for root, dirs, files in os.walk(some_dir):
            # Convert files to list of tuples with file sizes
            files_with_sizes = [(file, os.path.getsize(os.path.join(root, file))) for file in files]
            yield root, dirs, files_with_sizes

    def _tree_to_str(self, tree_gen: Generator[Tuple[str, List[str], List[Tuple[str, int]]], None, None],
                    trim_dir: Optional[str] = None) -> str:
        """
        Convert tree generator to string.

        Args:
            tree_gen: Tree generator
            trim_dir: Directory to trim from paths

        Returns:
            str: Tree as string
        """
        tree_str = ""
        for root, _, files_with_sizes in tree_gen:
            if trim_dir:
                root = root.replace(trim_dir, "")
            for file, size in files_with_sizes:
                tree_str += f"{root}/{file} - {size} bytes\n"
        return tree_str


# Define an AssistantAgent with the model, tools, system message, and reflection enabled.
agent = AssistantAgent(
    name="repo_management_agent",
    model_client=model_client,
    tools=[
        CloneRepoTool(),
        PrepareRepoTreeTool()
    ],
    system_message="""You are a helpful assistant specialized in working with Git repositories.
You have access to tools that can help you with these tasks. When given a repository URL, you can:
1. Clone the repository and remove confusing files
2. Prepare a tree structure of the repository that is easily readable by LLMs

You should use the clone_repo tool to clone a repository, and then use the prepare_repo_tree tool to generate a tree structure of the repository.
The prepare_repo_tree tool requires the repository name, which can be extracted from the repository URL by taking the last part of the URL, removing the .git extension, and replacing dots with hyphens.
For example, for the URL "https://github.com/run-rasztabiga-me/poc1-fastapi.git", the repository name would be "poc1-fastapi".
""",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)

# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(agent.run_stream(task="I want to clone this repository: https://github.com/run-rasztabiga-me/poc1-fastapi.git and then show me its directory structure"))
    # Close the connection to the model client.
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
