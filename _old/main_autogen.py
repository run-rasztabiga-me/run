import asyncio
import logging
import os
import shutil
import time
from typing import Dict, Generator, List, Optional, Tuple

import docker
from dotenv import load_dotenv
from git import Repo
from pydantic import BaseModel, Field

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.conditions import TextMessageTermination
from typing import Any, Dict, List, Optional, Set, Callable, Union
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import BaseTool
from autogen_agentchat.messages import TextMessage, BaseChatMessage, BaseAgentEvent
from autogen_ext.models.openai import OpenAIChatCompletionClient, _model_info

# Setup
load_dotenv()
logger = logging.getLogger(__name__)
docker_client = docker.from_env()

# Constants
CONFUSING_FILES = {".git", ".github", ".gitignore", ".gitmodules", ".gitattributes", 
                  ".gitlab-ci.yml", ".travis.yml", "LICENSE", "README.md", "CHANGELOG.md",
                  "__pycache__", ".pytest_cache", ".coverage", "htmlcov", ".idea", ".vscode"}
DOCKER_START_TIMEOUT = 5

# Define a model client
model_info = _model_info.get_info("gpt-4o")
model_client = OpenAIChatCompletionClient(
    model="openai/gpt-4o",
    model_info=model_info,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# TODO:
# 1. Nie trzeba przekazywac repo_name ciagle
# 2. Moze dodac memory?
# 3. Dodac rzeczy z prompta (prompts.py) np technology-specific guidelines (moze jako task? rozpoznajacy technologie i podajacy jakies docsy)
# 4. Dodac generowanie manifestow k8s
# 5. Dodac apply na klaster
# 6. Dodac wybor modeli (trzeba uderzac bezposrednio do API danego modelu)

# Define tool input/output models
class CloneRepoInput(BaseModel):
    repo_url: str = Field(..., description="URL of the repository to clone")

class PrepareRepoTreeInput(BaseModel):
    repo_name: str = Field(..., description="Name of the repository to prepare tree for")

class GetFileContentInput(BaseModel):
    repo_name: str = Field(..., description="Name of the repository")
    file_path: str = Field(..., description="Path to the file relative to the repository root")

class WriteFileInput(BaseModel):
    repo_name: str = Field(..., description="Name of the repository")
    file_path: str = Field(..., description="Path to the file relative to the repository root")
    content: str = Field(..., description="Content to write to the file")

class BuildDockerImageInput(BaseModel):
    repo_name: str = Field(..., description="Name of the repository")
    image_tag: str = Field(..., description="Tag for the Docker image")

class RunDockerContainerInput(BaseModel):
    repo_name: str = Field(..., description="Name of the repository")
    image_tag: str = Field(..., description="Tag for the Docker image")

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


class GetFileContentTool(BaseTool[GetFileContentInput, str]):
    name: str = "get_file_content"
    description: str = "Retrieve the content of a specific file from the repository"
    args_schema: type[BaseModel] = GetFileContentInput

    def __init__(self):
        super().__init__(
            args_type=GetFileContentInput,
            return_type=str,
            name=self.name,
            description=self.description
        )

    async def run(self, args: GetFileContentInput, cancellation_token=None) -> str:
        """Retrieve the content of a specific file from the repository."""
        tmp_dir = f"./tmp/{args.repo_name}"
        file_path = os.path.join(tmp_dir, args.file_path)

        if not os.path.isdir(tmp_dir):
            return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

        if not os.path.isfile(file_path):
            return f"Error: File {args.file_path} does not exist in the repository."

        try:
            logger.info(f"Reading file content: {file_path}")
            with open(file_path, "r") as f:
                content = f.read()
            return f"Content of {args.file_path}:\n\n{content}"
        except Exception as e:
            return f"Error reading file {args.file_path}: {str(e)}"


class WriteFileTool(BaseTool[WriteFileInput, str]):
    name: str = "write_file"
    description: str = "Write content to a file in the repository, creating it if it doesn't exist or modifying it if it does"
    args_schema: type[BaseModel] = WriteFileInput

    def __init__(self):
        super().__init__(
            args_type=WriteFileInput,
            return_type=str,
            name=self.name,
            description=self.description
        )

    async def run(self, args: WriteFileInput, cancellation_token=None) -> str:
        """Write content to a file in the repository."""
        tmp_dir = f"./tmp/{args.repo_name}"
        file_path = os.path.join(tmp_dir, args.file_path)

        if not os.path.isdir(tmp_dir):
            return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            action = "Modified" if os.path.exists(file_path) else "Created"
            logger.info(f"Writing file: {file_path}")
            with open(file_path, "w") as f:
                f.write(args.content)
            return f"{action} file {args.file_path} successfully."
        except Exception as e:
            return f"Error writing to file {args.file_path}: {str(e)}"


class BuildDockerImageTool(BaseTool[BuildDockerImageInput, str]):
    name: str = "build_docker_image"
    description: str = "Build and push a Docker image from the repository"
    args_schema: type[BaseModel] = BuildDockerImageInput

    def __init__(self):
        super().__init__(
            args_type=BuildDockerImageInput,
            return_type=str,
            name=self.name,
            description=self.description
        )

    async def run(self, args: BuildDockerImageInput, cancellation_token=None) -> str:
        """Build and push a Docker image from the repository."""
        tmp_dir = f"./tmp/{args.repo_name}"

        if not os.path.isdir(tmp_dir):
            return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

        dockerfile_path = os.path.join(tmp_dir, "Dockerfile")
        if not os.path.isfile(dockerfile_path):
            return f"Error: Dockerfile does not exist in the repository. Please create a Dockerfile first."

        try:
            logger.info(f"Building Docker image with tag {args.image_tag}...")
            image, build_logs = docker_client.images.build(path=tmp_dir, tag=args.image_tag, forcerm=True, pull=False)

            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    log_line = log['stream'].strip()
                    if log_line:
                        logger.info(log_line)

            logger.info(f"Pushing Docker image {args.image_tag}...")
            docker_client.images.push(args.image_tag)

            return f"Successfully built and pushed Docker image {args.image_tag}"
        except docker.errors.BuildError as e:
            return f"Error building Docker image: {str(e)}"
        except docker.errors.APIError as e:
            return f"Error with Docker API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


class RunDockerContainerTool(BaseTool[RunDockerContainerInput, str]):
    name: str = "run_docker_container"
    description: str = "Run a Docker container from a built image"
    args_schema: type[BaseModel] = RunDockerContainerInput

    def __init__(self):
        super().__init__(
            args_type=RunDockerContainerInput,
            return_type=str,
            name=self.name,
            description=self.description
        )

    def _get_exposed_ports(self, dockerfile_content: str) -> List[str]:
        """Extract exposed ports from Dockerfile content."""
        exposed_ports = []
        for line in dockerfile_content.split("\n"):
            if "EXPOSE" in line:
                exposed_ports = line.split(" ")[1:]
        return exposed_ports

    async def run(self, args: RunDockerContainerInput, cancellation_token=None) -> str:
        """Run a Docker container from a built image."""
        tmp_dir = f"./tmp/{args.repo_name}"

        if not os.path.isdir(tmp_dir):
            return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

        dockerfile_path = os.path.join(tmp_dir, "Dockerfile")
        if not os.path.isfile(dockerfile_path):
            return f"Error: Dockerfile does not exist in the repository. Please create a Dockerfile first."

        try:
            # Read Dockerfile to extract exposed ports
            with open(dockerfile_path, "r") as f:
                dockerfile_content = f.read()

            exposed_ports = self._get_exposed_ports(dockerfile_content)
            if not exposed_ports:
                return f"Error: No exposed ports found in Dockerfile. Please make sure the Dockerfile contains an EXPOSE instruction."

            # Create port mapping
            ports = {f"{port}/tcp": None for port in exposed_ports}

            logger.info(f"Running Docker container from image {args.image_tag}...")
            container = docker_client.containers.run(args.image_tag, detach=True, ports=ports)

            # Wait for container to start
            time.sleep(DOCKER_START_TIMEOUT)
            container.reload()

            # Get container information
            container_info = {
                "id": container.id,
                "name": container.name,
                "status": container.status,
                "ports": container.ports
            }

            return f"Successfully started Docker container from image {args.image_tag}. Container information: {container_info}"
        except docker.errors.ImageNotFound as e:
            return f"Error: Docker image {args.image_tag} not found. Please build the image first: {str(e)}"
        except docker.errors.APIError as e:
            return f"Error with Docker API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


# Define an AssistantAgent with the model, tools, system message, and reflection enabled.
agent = AssistantAgent(
    name="repo_management_agent",
    model_client=model_client,
    tools=[
        CloneRepoTool(),
        PrepareRepoTreeTool(),
        GetFileContentTool(),
        WriteFileTool(),
        BuildDockerImageTool(),
        RunDockerContainerTool()
    ],
    system_message="""You are a helpful assistant specialized in working with Git repositories.
You have access to tools that can help you with these tasks. When given a repository URL, you can:
1. Clone the repository and remove confusing files
2. Analyze the repository structure to identify important files
3. Retrieve the content of files you determine are necessary to understand the application
4. Write or modify files in the repository (e.g., Dockerfile, Kubernetes manifests)
5. Build and push Docker images based on the Dockerfile
6. Run Docker containers from built images to test if they work

You should use the clone_repo tool to clone a repository. The repository name can be extracted from the repository URL by taking the last part of the URL, removing the .git extension, and replacing dots with hyphens.
For example, for the URL "https://github.com/run-rasztabiga-me/poc1-fastapi.git", the repository name would be "poc1-fastapi".

You can use the prepare_repo_tree tool to get an overview of the repository structure if needed, but you should focus on identifying and examining files that are most relevant to understanding the application and creating the required outputs.

Use the get_file_content tool to retrieve the content of specific files that you determine are important. This tool requires the repository name and the file path relative to the repository root.

You can use the write_file tool to create new files or modify existing ones in the repository. This tool requires the repository name, the file path relative to the repository root, and the content to write to the file. This is particularly useful for creating files like Dockerfile or Kubernetes manifests.

After creating a Dockerfile, you can use the build_docker_image tool to build and push a Docker image based on that Dockerfile. This tool requires the repository name and the image tag. The image tag should follow the format "registry/repository-name:tag" (e.g., "localhost:5000/poc1-fastapi:latest").

After building a Docker image, you can use the run_docker_container tool to run a container from that image and test if it works. This tool requires the repository name and the image tag. It will automatically extract the exposed ports from the Dockerfile and map them to random ports on the host.

IMPORTANT: You must continue the conversation until you have successfully generated a Dockerfile for the application, built a Docker image based on that Dockerfile, AND run a Docker container from that image to test if it works. After you have completed all these steps, respond with a message that includes the word "DONE" to indicate that you have completed the task.
""",
    reflect_on_tool_use=True,
    model_client_stream=False,  # Enable streaming tokens for better console output
)

# Define a custom termination condition that checks if the Dockerfile has been generated and Docker image has been built
class DockerBuildTermination(TerminationCondition):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.dockerfile_generated = False
        self.docker_image_built = False
        self.docker_container_run = False

    @property
    def terminated(self) -> bool:
        return self.dockerfile_generated and self.docker_image_built and self.docker_container_run

    async def reset(self) -> None:
        self.dockerfile_generated = False
        self.docker_image_built = False
        self.docker_container_run = False

    async def __call__(self, messages: List[Union[Dict[str, Any], BaseChatMessage, BaseAgentEvent]]) -> Any:
        # If already terminated, don't check again
        if self.terminated:
            from autogen_agentchat.base import TerminatedException
            raise TerminatedException("Termination condition has already been reached")

        # Check if any message indicates that a Dockerfile has been created
        for message in messages:
            # Handle different types of message objects
            if isinstance(message, TextMessage):
                # For TextMessage objects, access properties directly
                if message.source == self.agent_name and isinstance(message.content, str):
                    content = message.content
                    # Check if the message is a tool result indicating a Dockerfile was created
                    if "Created file Dockerfile successfully" in content or "Modified file Dockerfile successfully" in content:
                        self.dockerfile_generated = True
                    # Check if the message indicates a Docker image was built
                    if "Successfully built and pushed Docker image" in content:
                        self.docker_image_built = True
                    # Check if the message indicates a Docker container was run
                    if "Successfully started Docker container" in content:
                        self.docker_container_run = True
                    # Check if the message contains the word "DONE" to indicate task completion
                    if "DONE" in content:
                        self.dockerfile_generated = True
                        self.docker_image_built = True
                        self.docker_container_run = True
            elif isinstance(message, dict):
                # For dictionary-like messages, use get() method
                if message.get("name") == self.agent_name and isinstance(message.get("content"), str):
                    content = message.get("content", "")
                    # Check if the message is a tool result indicating a Dockerfile was created
                    if "Created file Dockerfile successfully" in content or "Modified file Dockerfile successfully" in content:
                        self.dockerfile_generated = True
                    # Check if the message indicates a Docker image was built
                    if "Successfully built and pushed Docker image" in content:
                        self.docker_image_built = True
                    # Check if the message indicates a Docker container was run
                    if "Successfully started Docker container" in content:
                        self.docker_container_run = True
                    # Check if the message contains the word "DONE" to indicate task completion
                    if "DONE" in content:
                        self.dockerfile_generated = True
                        self.docker_image_built = True
                        self.docker_container_run = True
                    # Also check for tool calls that might have created a Dockerfile, built a Docker image, or run a Docker container
                    tool_calls = message.get("tool_calls", [])
                    for tool_call in tool_calls:
                        if tool_call.get("name") == "write_file":
                            args = tool_call.get("args", {})
                            if args.get("file_path") == "Dockerfile":
                                self.dockerfile_generated = True
                        elif tool_call.get("name") == "build_docker_image":
                            self.docker_image_built = True
                        elif tool_call.get("name") == "run_docker_container":
                            self.docker_container_run = True

        if self.dockerfile_generated and self.docker_image_built and self.docker_container_run:
            from autogen_agentchat.messages import StopMessage
            return StopMessage(content="Dockerfile has been generated, Docker image has been built, and Docker container has been run successfully", source="DockerBuildTermination")
        return None

# Add termination condition and create a team
termination_condition = DockerBuildTermination("repo_management_agent")

# Create a team with the agent and the termination condition
team = RoundRobinGroupChat(
    [agent],
    termination_condition=termination_condition,
)

# Run the team with a task and print the messages to the console
async def main() -> None:
    # Use the team's run_stream method to get better console output
    async for message in team.run_stream(
        task="I want to clone this repository: https://github.com/run-rasztabiga-me/poc1-fastapi.git. Analyze the repository and find what files you think are necessary to understand the application. Then create a Dockerfile for this application, build a Docker image with the tag 'localhost:5000/poc1-fastapi:latest', and run a Docker container from that image to see if it works."
    ):  # type: ignore
        print(type(message).__name__, message)

    # Close the connection to the model client.
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
