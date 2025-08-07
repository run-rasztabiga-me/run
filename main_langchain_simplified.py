import os
import shutil
import logging
from git import Repo
from typing import List, Tuple, Generator
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv

# Setup
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[] # Remove default handler to avoid duplicates
)
logger = logging.getLogger(__name__)

# Constants
CONFUSING_FILES = {".git", ".github", ".gitignore", ".gitmodules",
                   ".gitattributes", ".gitlab-ci.yml", ".travis.yml", "LICENSE",
                   "README.md", "CHANGELOG.md", "__pycache__", ".pytest_cache",
                   ".coverage", "htmlcov", ".idea", ".vscode", "docker-compose.yml", "Dockerfile"}

# Global state
REPO_NAME = None  # Will be set when the repository is cloned

# Define schemas for tools
class CloneRepoInputSchema(BaseModel):
  repo_url: str = Field(description="URL of the repository to clone")


@tool("clone_repo", args_schema=CloneRepoInputSchema)
def clone_repo(repo_url: str) -> str:
  """Clone repository and recursively remove confusing files."""
  global REPO_NAME
  repo_name = repo_url.split("/")[-1].replace(".git", "").replace(".", "-")
  REPO_NAME = repo_name
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
  pass


@tool("prepare_repo_tree", args_schema=PrepareRepoTreeInputSchema)
def prepare_repo_tree() -> str:
  """Prepare repository tree structure as a string."""
  global REPO_NAME
  if REPO_NAME is None:
    return "Error: No repository has been cloned yet. Please clone a repository first."
  
  tmp_dir = f"./tmp/{REPO_NAME}"

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
  file_path: str = Field(
      description="Path to the file relative to the repository root")


@tool("get_file_content", args_schema=GetFileContentInputSchema)
def get_file_content(file_path: str) -> str:
  """Retrieve the content of a specific file from the repository."""
  global REPO_NAME
  if REPO_NAME is None:
    return "Error: No repository has been cloned yet. Please clone a repository first."
    
  tmp_dir = f"./tmp/{REPO_NAME}"
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
  file_path: str = Field(
      description="Path to the file relative to the repository root")
  content: str = Field(description="Content to write to the file")


@tool("write_file", args_schema=WriteFileInputSchema)
def write_file(file_path: str, content: str) -> str:
  """Write content to a file in the repository."""
  global REPO_NAME
  if REPO_NAME is None:
    return "Error: No repository has been cloned yet. Please clone a repository first."
    
  tmp_dir = f"./tmp/{REPO_NAME}"
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


class ListDirectoryInputSchema(BaseModel):
  dir_path: str = Field(description="Path to the directory relative to the repository root")


@tool("ls", args_schema=ListDirectoryInputSchema)
def list_directory(dir_path: str) -> str:
  """
  List files in a directory within the repository.
  
  Args:
      dir_path: Path to the directory relative to the repository root
      
  Returns:
      str: List of files and directories in the specified path
  """
  global REPO_NAME
  if REPO_NAME is None:
    return "Error: No repository has been cloned yet. Please clone a repository first."
    
  tmp_dir = f"./tmp/{REPO_NAME}"
  
  if not os.path.isdir(tmp_dir):
    return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."
  
  # Handle empty or '.' paths to refer to repository root
  if not dir_path or dir_path == '.':
    dir_path_full = tmp_dir
  else:
    dir_path_full = os.path.join(tmp_dir, dir_path)
  
  if not os.path.exists(dir_path_full):
    return f"Error: Directory {dir_path} does not exist in the repository."
  
  if not os.path.isdir(dir_path_full):
    return f"Error: {dir_path} is not a directory."
  
  try:
    items = os.listdir(dir_path_full)
    
    # Create a formatted output
    result = f"Contents of {dir_path or './'} directory:\n"
    
    # Separate directories and files
    directories = []
    files = []
    
    for item in items:
      item_path = os.path.join(dir_path_full, item)
      if os.path.isdir(item_path):
        directories.append(f"{item}/")
      else:
        file_size = os.path.getsize(item_path)
        files.append(f"{item} - {file_size} bytes")
    
    # Display directories first, then files
    if directories:
      result += "\nDirectories:\n"
      result += "\n".join(sorted(directories))
      
    if files:
      result += "\n\nFiles:\n"
      result += "\n".join(sorted(files))
    
    if not items:
      result += "(empty directory)"
      
    return result
    
  except Exception as e:
    return f"Error listing directory {dir_path}: {str(e)}"


# Initialize LangGraph agent

tools = [clone_repo, prepare_repo_tree, get_file_content, write_file, list_directory]

system_message = """You are a helpful assistant specialized in working with Git repositories.
You have access to tools that can help you with these tasks. When given a repository URL, you can:
1. Clone the repository and remove confusing files
2. Analyze the repository structure to identify important files
3. Retrieve the content of files you determine are necessary to understand the application
4. Write or modify files in the repository (e.g., Dockerfile, Kubernetes manifests)
5. List directory contents within the repository

You should use the clone_repo tool to clone a repository. The repository name can be extracted from the repository URL by taking the last part of the URL, removing the .git extension, and replacing dots with hyphens.
For example, for the URL \"https://github.com/run-rasztabiga-me/poc1-fastapi.git\", the repository name would be \"poc1-fastapi\".

You can use the prepare_repo_tree tool to get an overview of the repository structure if needed, but you should focus on identifying and examining files that are most relevant to understanding the application and creating the required outputs.

Use the get_file_content tool to retrieve the content of specific files that you determine are important. This tool requires the file path relative to the repository root.

You can use the write_file tool to create new files or modify existing ones in the repository. This tool requires the file path relative to the repository root and the content to write to the file. This is particularly useful for creating files like Dockerfile or Kubernetes manifests.

You can use the ls tool to list the contents of a directory within the cloned repository. This tool requires the directory path relative to the repository root. You can use an empty string or "." to list the contents of the repository root directory. The tool will display directories and files separately, with directories having a trailing slash and files showing their sizes in bytes. This is useful for exploring the repository structure in a more focused way than the prepare_repo_tree tool.

Your objective is to create:

1. A Dockerfile that properly containerizes the application. When creating the Dockerfile, carefully analyze the application code to ensure that any health check endpoint you specify actually exists in the application.

2. Kubernetes manifests for the application. These manifests should:
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

llm = init_chat_model(
    model="gpt-5-mini",
    model_provider="openai",
    temperature=1
)

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message
)

if __name__ == "__main__":
  # Configure a console handler to ensure logs are visible
  console_handler = logging.StreamHandler()
  console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
  
  # Add the handler to the root logger
  logging.getLogger().addHandler(console_handler)
  
  # Reset global state
  REPO_NAME = None
  
  # Example repository URLs to process
  repo_url = "https://github.com/run-rasztabiga-me/poc1-fastapi.git"
  
  logger.info("Starting agent with task: " + repo_url)
  
  # Stream agent responses
  for chunk in agent.stream({"messages": [{"role": "user", "content": repo_url}]},{"recursion_limit": 50},stream_mode="updates"):
    print(chunk)
    print("\n")
