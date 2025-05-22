import time
from dotenv import load_dotenv
import os
import shutil
import logging
import docker
from git import Repo
from typing import List, Tuple, Generator
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Setup
load_dotenv()
logger = logging.getLogger(__name__)
docker_client = docker.from_env()

# Constants
CONFUSING_FILES = {".git", ".github", ".gitignore", ".gitmodules",
                   ".gitattributes", ".gitlab-ci.yml", ".travis.yml", "LICENSE",
                   "README.md", "CHANGELOG.md", "__pycache__", ".pytest_cache",
                   ".coverage", "htmlcov", ".idea", ".vscode"}
DOCKER_START_TIMEOUT = 5


# TODO
# 1. Dodac schema i wiele argumentow do tooli bo sobie nie radzi
# 2. Nie umie odpalic obrazu dockera bo mu przeszkadzaja porty
# 3. Przepisac na graph z react agent

# Define schemas for tools
class CloneRepoInputSchema(BaseModel):
  repo_url: str = Field(description="URL of the repository to clone")


@tool("clone_repo", args_schema=CloneRepoInputSchema)
def clone_repo(repo_url: str) -> str:
  """Clone repository and recursively remove confusing files."""
  repo_name = repo_url.split("/")[-1].replace(".git", "").replace(".", "-")
  tmp_dir = f"./tmp/{repo_name}"

  logger.info("Preparing working directory...")
  shutil.rmtree(tmp_dir, ignore_errors=True)
  os.makedirs(tmp_dir, exist_ok=True)

  logger.info("Cloning repository...")
  Repo.clone_from(repo_url, tmp_dir)

  for root, dirs, files in os.walk(tmp_dir, topdown=True):
    for file in files:
      if file in CONFUSING_FILES:
        file_path = os.path.join(root, file)
        logger.info(f"Removing confusing file: {file_path}")
        os.remove(file_path)

    for dir_name in list(dirs):
      if dir_name in CONFUSING_FILES:
        dir_path = os.path.join(root, dir_name)
        logger.info(f"Removing confusing directory: {dir_path}")
        shutil.rmtree(dir_path, ignore_errors=True)
        dirs.remove(dir_name)

  return f"Repository {repo_url} cloned successfully to {tmp_dir}"


class PrepareRepoTreeInputSchema(BaseModel):
  repo_name: str = Field(
      description="Name of the repository to prepare the tree for")


@tool("prepare_repo_tree", args_schema=PrepareRepoTreeInputSchema)
def prepare_repo_tree(repo_name: str) -> str:
  """Prepare repository tree structure as a string."""
  tmp_dir = f"./tmp/{repo_name}"

  if not os.path.isdir(tmp_dir):
    return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

  logger.info("Preparing tree...")
  dir_tree = _tree(tmp_dir)
  return _tree_to_str(dir_tree, trim_dir=tmp_dir)


def _tree(some_dir: str) -> Generator[
  Tuple[str, List[str], List[Tuple[str, int]]], None, None]:
  assert os.path.isdir(some_dir)

  for root, dirs, files in os.walk(some_dir):
    files_with_sizes = [(file, os.path.getsize(os.path.join(root, file))) for
                        file in files]
    yield root, dirs, files_with_sizes


def _tree_to_str(tree_gen: Generator[Tuple[str, List[str], List[Tuple[str, int]]], None, None],trim_dir: str = None) -> str:
  tree_str = ""
  for root, _, files_with_sizes in tree_gen:
    if trim_dir:
      root = root.replace(trim_dir, "")
    for file, size in files_with_sizes:
      tree_str += f"{root}/{file} - {size} bytes\n"
  return tree_str


class GetFileContentInputSchema(BaseModel):
  repo_name: str = Field(description="Name of the repository")
  file_path: str = Field(
      description="Path to the file relative to the repository root")


@tool("get_file_content", args_schema=GetFileContentInputSchema)
def get_file_content(repo_name: str, file_path: str) -> str:
  """Retrieve the content of a specific file from the repository."""
  tmp_dir = f"./tmp/{repo_name}"
  file_path_full = os.path.join(tmp_dir, file_path)

  if not os.path.isdir(tmp_dir):
    return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

  if not os.path.isfile(file_path_full):
    return f"Error: File {file_path} does not exist in the repository."

  try:
    logger.info(f"Reading file content: {file_path_full}")
    with open(file_path_full, "r") as f:
      content = f.read()
    return content
  except Exception as e:
    return f"Error reading file {file_path}: {str(e)}"


class WriteFileInputSchema(BaseModel):
  repo_name: str = Field(description="Name of the repository")
  file_path: str = Field(
      description="Path to the file relative to the repository root")
  content: str = Field(description="Content to write to the file")


@tool("write_file", args_schema=WriteFileInputSchema)
def write_file(repo_name: str, file_path: str, content: str) -> str:
  """Write content to a file in the repository."""
  tmp_dir = f"./tmp/{repo_name}"
  file_path_full = os.path.join(tmp_dir, file_path)

  if not os.path.isdir(tmp_dir):
    return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

  os.makedirs(os.path.dirname(file_path_full), exist_ok=True)

  try:
    action = "Modified" if os.path.exists(file_path_full) else "Created"
    logger.info(f"Writing file: {file_path_full}")
    with open(file_path_full, "w") as f:
      f.write(content)
    return f"{action} file {file_path} successfully."
  except Exception as e:
    return f"Error writing to file {file_path}: {str(e)}"


class BuildDockerImageInputSchema(BaseModel):
  repo_name: str = Field(description="Name of the repository")
  image_tag: str = Field(description="Tag for the Docker image")


@tool("build_docker_image", args_schema=BuildDockerImageInputSchema)
def build_docker_image(repo_name: str, image_tag: str) -> str:
  """Build and push a Docker image from the repository."""
  tmp_dir = f"./tmp/{repo_name}"

  if not os.path.isdir(tmp_dir):
    return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

  dockerfile_path = os.path.join(tmp_dir, "Dockerfile")
  if not os.path.isfile(dockerfile_path):
    return f"Error: Dockerfile does not exist in the repository. Please create a Dockerfile first."

  try:
    logger.info(f"Building Docker image with tag {image_tag}...")
    image, build_logs = docker_client.images.build(path=tmp_dir, tag=image_tag,
                                                   forcerm=True, pull=False)

    for log in build_logs:
      if 'stream' in log:
        log_line = log['stream'].strip()
        if log_line:
          logger.info(log_line)

    logger.info(f"Pushing Docker image {image_tag}...")
    docker_client.images.push(image_tag)

    return f"Successfully built and pushed Docker image {image_tag}"
  except docker.errors.BuildError as e:
    return f"Error building Docker image: {str(e)}"
  except docker.errors.APIError as e:
    return f"Error with Docker API: {str(e)}"
  except Exception as e:
    return f"Unexpected error: {str(e)}"


class RunDockerContainerInputSchema(BaseModel):
  repo_name: str = Field(description="Name of the repository")
  image_tag: str = Field(description="Tag for the Docker image")


def _get_exposed_ports(dockerfile_content: str) -> List[str]:
  """Extract exposed ports from Dockerfile content."""
  exposed_ports = []
  for line in dockerfile_content.split("\n"):
    if line.strip().startswith("EXPOSE"):
      ports = line.split()[1:]  # Split and take everything after 'EXPOSE'
      exposed_ports.extend(ports)
  return exposed_ports


@tool("run_docker_container", args_schema=RunDockerContainerInputSchema)
def run_docker_container(repo_name: str, image_tag: str) -> str:
  """Run a Docker container from a built image."""
  tmp_dir = f"./tmp/{repo_name}"

  if not os.path.isdir(tmp_dir):
    return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

  dockerfile_path = os.path.join(tmp_dir, "Dockerfile")
  if not os.path.isfile(dockerfile_path):
    return f"Error: Dockerfile does not exist in the repository. Please create a Dockerfile first."

  try:
    with open(dockerfile_path, "r") as f:
      dockerfile_content = f.read()

    exposed_ports = _get_exposed_ports(dockerfile_content)
    if not exposed_ports:
      return f"Error: No exposed ports found in Dockerfile. Please make sure the Dockerfile contains an EXPOSE instruction."

    # Create port mapping
    ports = {port: None for port in exposed_ports}

    logger.info(f"Running Docker container from image {image_tag}...")
    container = docker_client.containers.run(image_tag, detach=True,
                                             ports=ports)

    time.sleep(DOCKER_START_TIMEOUT)
    container.reload()

    container_info = {
      "id": container.id,
      "name": container.name,
      "status": container.status,
      "ports": container.ports
    }

    return f"Successfully started Docker container from image {image_tag}. Container information: {container_info}"
  except docker.errors.ImageNotFound as e:
    return f"Error: Docker image {image_tag} not found. Please build the image first: {str(e)}"
  except docker.errors.APIError as e:
    return f"Error with Docker API: {str(e)}"
  except Exception as e:
    return f"Unexpected error: {str(e)}"


# Initialize LangGraph agent

tools = [clone_repo, prepare_repo_tree, get_file_content, write_file,
         build_docker_image, run_docker_container]

system_message = """You are a helpful assistant specialized in working with Git repositories.
You have access to tools that can help you with these tasks. When given a repository URL, you can:
1. Clone the repository and remove confusing files
2. Analyze the repository structure to identify important files
3. Retrieve the content of files you determine are necessary to understand the application
4. Write or modify files in the repository (e.g., Dockerfile, Kubernetes manifests)
5. Build and push Docker images based on the Dockerfile
6. Run Docker containers from built images to test if they work
7. Generate Kubernetes manifests for the application

You should use the clone_repo tool to clone a repository. The repository name can be extracted from the repository URL by taking the last part of the URL, removing the .git extension, and replacing dots with hyphens.
For example, for the URL \"https://github.com/run-rasztabiga-me/poc1-fastapi.git\", the repository name would be \"poc1-fastapi\".

You can use the prepare_repo_tree tool to get an overview of the repository structure if needed, but you should focus on identifying and examining files that are most relevant to understanding the application and creating the required outputs.

Use the get_file_content tool to retrieve the content of specific files that you determine are important. This tool requires the repository name and the file path relative to the repository root.

You can use the write_file tool to create new files or modify existing ones in the repository. This tool requires the repository name, the file path relative to the repository root, and the content to write to the file. This is particularly useful for creating files like Dockerfile or Kubernetes manifests.

After creating a Dockerfile, you can use the build_docker_image tool to build and push a Docker image based on that Dockerfile. This tool requires the repository name and the image tag. The image tag should follow the format \"registry/repository-name:tag\" (e.g., \"localhost:5000/poc1-fastapi:latest\").

After building a Docker image, you can use the run_docker_container tool to run a container from that image and test if it works. This tool requires the repository name and the image tag. It will automatically extract the exposed ports from the Dockerfile and map them to random ports on the host.

After successfully running a Docker container, generate Kubernetes manifests for the application. These manifests should:
- Include all required resources (Deployments, Services, Ingresses, and Volumes if necessary)
- Match exposed ports precisely as specified in the Dockerfile
- Set replicas default to 1 unless otherwise stated
- For ingress host, use "<repository-name>.rasztabiga.me" (e.g., repository "app1" → domain "app1.rasztabiga.me")
- Follow Kubernetes best practices and ensure security measures
- If external dependencies (e.g., databases like PostgreSQL, Redis, MySQL) are identified, generate appropriate Kubernetes resources for those dependencies as well
- Deploy stateful dependencies using StatefulSets with appropriate PersistentVolumeClaims
- Deploy stateless applications using Deployments
- Use Services to expose applications internally and externally as necessary
- Ensure all Kubernetes secrets are in base64 format
- Include appropriate resource limits/requests and health checks (liveness and readiness probes)

Use the write_file tool to save these Kubernetes manifests in the repository.

IMPORTANT: You must continue the conversation until you have successfully generated a Dockerfile for the application, built a Docker image based on that Dockerfile, run a Docker container from that image to test if it works, AND generated appropriate Kubernetes manifests for the application. After you have completed all these steps, respond with a message that includes the word \"DONE\" to indicate that you have completed the task.
"""

llm = init_chat_model(
    model="gpt-4o",
    # temperature=0
)

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message
)

if __name__ == "__main__":
  task = "I want to clone this repository: https://github.com/run-rasztabiga-me/poc1-fastapi.git. Analyze the repository and find what files you think are necessary to understand the application. Then create a Dockerfile for this application, build a Docker image with the tag 'localhost:5000/poc1-fastapi:latest', run a Docker container from that image to see if it works, and generate Kubernetes manifests for the application."
  for chunk in agent.stream({"messages": [{"role": "user", "content": task}]},stream_mode="updates"):
    print(chunk)
    print("\n")
