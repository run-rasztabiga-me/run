import time
import subprocess
from dotenv import load_dotenv
import os
import shutil
import logging
import docker
from git import Repo
from typing import List, Tuple, Generator, Optional
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException

# Setup
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[] # Remove default handler to avoid duplicates
)
logger = logging.getLogger(__name__)
docker_client = docker.from_env()

# Constants
CONFUSING_FILES = {".git", ".github", ".gitignore", ".gitmodules",
                   ".gitattributes", ".gitlab-ci.yml", ".travis.yml", "LICENSE",
                   "README.md", "CHANGELOG.md", "__pycache__", ".pytest_cache",
                   ".coverage", "htmlcov", ".idea", ".vscode", "docker-compose.yml",}
DOCKER_START_TIMEOUT = 5
K8S_INGRESS_TIMEOUT = 5
DOCKER_REGISTRY = os.environ.get('DOCKER_REGISTRY', 'localhost:5001')

# Global state
REPO_NAME = None  # Will be set when the repository is cloned


# TODO
# 1. Dodac schema i wiele argumentow do tooli bo sobie nie radzi
# 2. Nie umie odpalic obrazu dockera bo mu przeszkadzaja porty
# 3. Przepisac na graph z react agent
# 4. Przerobic zeby wypychal obrazy do rejestru na homelabie
# 5. Przerobic zeby kubectl aplikowal na homelabie
# 6. Jak uderzyc do k8s zeby zweryfikowac czy dziala?
# 7. Dodac curl tool zeby sprawdzic czy dziala
# 8. Dodac "ls" tool do listowania plikow w danym katalogu

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


class BuildDockerImageInputSchema(BaseModel):
  image_tag: str = Field(description="Tag for the Docker image")


@tool("build_docker_image", args_schema=BuildDockerImageInputSchema)
def build_docker_image(image_tag: str) -> str:
  """Build and push a Docker image from the repository."""
  global REPO_NAME
  if REPO_NAME is None:
    return "Error: No repository has been cloned yet. Please clone a repository first."
    
  tmp_dir = f"./tmp/{REPO_NAME}"

  if not os.path.isdir(tmp_dir):
    return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."

  dockerfile_path = os.path.join(tmp_dir, "Dockerfile")
  if not os.path.isfile(dockerfile_path):
    return f"Error: Dockerfile does not exist in the repository. Please create a Dockerfile first."

  # Add registry address if not already in the tag
  if DOCKER_REGISTRY not in image_tag:
    image_tag = f"{DOCKER_REGISTRY}/{image_tag}"
    logger.info(f"Added registry address to image tag: {image_tag}")

  try:
    # Detect current architecture
    current_arch = subprocess.check_output(["uname", "-m"]).decode().strip().lower()
    logger.info(f"Detected architecture: {current_arch}")
    
    # Use platform-specific build options
    build_args = {}
    platform = None
    
    # Check if we're on ARM architecture (Apple Silicon)
    # if current_arch in ["arm64", "aarch64"]:
    #   logger.info("ARM64 architecture detected, using cross-building for AMD64")
    #   platform = "linux/amd64"
    
    logger.info(f"Building Docker image with tag {image_tag}...")
    
    # Use buildx if cross-platform building is needed

    if platform:
      # Create Docker buildx command
      buildx_cmd = [
          "docker", "buildx", "build", 
          "--platform", platform, 
          "-t", image_tag, 
          "--push",
          tmp_dir
      ]
      
      logger.info(f"Executing cross-platform build: {' '.join(buildx_cmd)}")
      subprocess.check_call(buildx_cmd)
      
      # Since we pushed directly with buildx, we don't need to push again
      logger.info(f"Docker image {image_tag} built and pushed using buildx")
      return f"Successfully built and pushed multi-platform Docker image {image_tag} for {platform}"
    else:
      # Use regular docker build for same architecture
      image, build_logs = docker_client.images.build(
          path=tmp_dir, 
          tag=image_tag,
          forcerm=True, 
          pull=False,
          buildargs=build_args
      )

      for log in build_logs:
        if 'stream' in log:
          log_line = log['stream'].strip()
          if log_line:
            logger.info(log_line)

      logger.info(f"Pushing Docker image {image_tag}...")
      docker_client.images.push(image_tag)

      return f"Successfully built and pushed Docker image {image_tag}"
  except subprocess.CalledProcessError as e:
    return f"Error during cross-platform Docker build: {str(e)}"
  except docker.errors.BuildError as e:
    return f"Error building Docker image: {str(e)}"
  except docker.errors.APIError as e:
    return f"Error with Docker API: {str(e)}"
  except Exception as e:
    return f"Unexpected error: {str(e)}"


class RunDockerContainerInputSchema(BaseModel):
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
def run_docker_container(image_tag: str) -> str:
  """Run a Docker container from a built image."""
  global REPO_NAME
  if REPO_NAME is None:
    return "Error: No repository has been cloned yet. Please clone a repository first."
    
  tmp_dir = f"./tmp/{REPO_NAME}"

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


def normalize_namespace(name: str) -> str:
  """
  Normalize namespace name to comply with Kubernetes naming rules.
  Namespace must be a lowercase RFC 1123 label:
  - must consist of lower case alphanumeric characters or '-'
  - must start and end with an alphanumeric character
  - regex: [a-z0-9]([-a-z0-9]*[a-z0-9])?
  
  Args:
      name: Name to normalize
      
  Returns:
      str: Normalized name
  """
  # Replace any non-alphanumeric characters with '-'
  normalized = ''.join(c if c.isalnum() else '-' for c in name.lower())
  
  # Remove leading/trailing hyphens
  normalized = normalized.strip('-')
  
  # Replace multiple consecutive hyphens with a single one
  normalized = '-'.join(filter(None, normalized.split('-')))
  
  # Ensure the name starts and ends with an alphanumeric character
  if not normalized[0].isalnum():
    normalized = 'a' + normalized
  if not normalized[-1].isalnum():
    normalized = normalized + 'a'
      
  return normalized


def ensure_etc_hosts_entry(hostname: str) -> None:
  """
  Ensures that a hostname has an entry in the /etc/hosts file.
  Maps the hostname to the local IP address (127.0.0.1).
  
  Args:
      hostname: The hostname to add to /etc/hosts
  """
  logger.info(f"Ensuring /etc/hosts entry for {hostname}...")
  
  # Check if the hostname is already in /etc/hosts
  try:
    with open('/etc/hosts', 'r') as file:
      hosts_content = file.read()
      
    # If hostname already exists in hosts file, no need to modify
    if hostname in hosts_content:
      logger.info(f"Hostname {hostname} already exists in /etc/hosts")
      return
      
    # Prepare the new entry to add (map to 127.0.0.1)
    new_entry = f"\n127.0.0.1\t{hostname}"
    
    # Since modifying /etc/hosts requires sudo, use subprocess
    add_hosts_cmd = ['sudo', 'sh', '-c', f'echo "{new_entry}" >> /etc/hosts']
    logger.info(f"Adding {hostname} to /etc/hosts with command: {' '.join(add_hosts_cmd)}")
    
    result = subprocess.run(add_hosts_cmd, capture_output=True, text=True)
    if result.returncode == 0:
      logger.info(f"Successfully added {hostname} to /etc/hosts")
    else:
      logger.warning(f"Failed to add {hostname} to /etc/hosts: {result.stderr}")
      
  except Exception as e:
    logger.warning(f"Error ensuring /etc/hosts entry: {str(e)}")
    # Continue execution even if this fails, as it's not critical


class ApplyK8sManifestInputSchema(BaseModel):
  manifest_path: str = Field(description="Path to the Kubernetes manifest file or directory relative to the repository root")


@tool("apply_k8s_manifest", args_schema=ApplyK8sManifestInputSchema)
def apply_k8s_manifest(manifest_path: str) -> str:
  """
  Apply Kubernetes manifest to cluster. This can handle either a single manifest file or a directory containing multiple manifest files.
  
  Args:
      manifest_path: Path to the Kubernetes manifest file or directory relative to the repository root
      
  Returns:
      str: Ingress URL if successful, error message otherwise
  """
  global REPO_NAME
  if REPO_NAME is None:
    return "Error: No repository has been cloned yet. Please clone a repository first."
    
  tmp_dir = f"./tmp/{REPO_NAME}"
  
  if not os.path.isdir(tmp_dir):
    return f"Error: Repository directory {tmp_dir} does not exist. Please clone the repository first."
  
  manifest_path_full = os.path.join(tmp_dir, manifest_path)
  if not os.path.exists(manifest_path_full):
    return f"Error: Manifest path {manifest_path} does not exist in the repository."
  
  logger.info(f"Applying Kubernetes manifest from {manifest_path_full}...")
  
  try:
    # Load Kubernetes configuration
    kubeconfig = os.getenv("KUBECONFIG")
    config.load_kube_config(config_file=kubeconfig if kubeconfig else None)
    
    namespace = normalize_namespace(REPO_NAME)
    api = client.CoreV1Api()
    networking_api = client.NetworkingV1Api()

    # Delete namespace if exists
    try:
      api.delete_namespace(name=namespace)
      logger.info(f"Namespace '{namespace}' deletion initiated.")
    except ApiException as e:
      if e.status == 404:
        logger.info(f"Namespace '{namespace}' not found, skipping deletion.")
      else:
        return f"Error deleting namespace: {str(e)}"

    # Wait for namespace to be deleted
    wait_count = 0
    while wait_count < 10:  # Maximum 10 retries
      try:
        api.read_namespace(name=namespace)
        logger.info("Waiting for namespace to delete...")
        time.sleep(2)
        wait_count += 1
      except ApiException as e:
        if e.status == 404:
          logger.info("Namespace deleted.")
          break
        return f"Error checking namespace: {str(e)}"
      
    if wait_count >= 10:
      return f"Timed out waiting for namespace '{namespace}' to be deleted."

    # Create namespace
    namespace_body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
    api.create_namespace(namespace_body)
    logger.info(f"Namespace '{namespace}' created.")

    # Apply Kubernetes manifests
    manifest_files = []
    
    # Handle both single file and directory cases
    if os.path.isfile(manifest_path_full):
      manifest_files.append(manifest_path_full)
      logger.info(f"Found single manifest file: {manifest_path_full}")
    elif os.path.isdir(manifest_path_full):
      # Get all yaml/yml files in the directory
      for file in os.listdir(manifest_path_full):
        if file.endswith('.yaml') or file.endswith('.yml'):
          manifest_files.append(os.path.join(manifest_path_full, file))
      logger.info(f"Found {len(manifest_files)} manifest files in directory {manifest_path_full}")
    
    if not manifest_files:
      return f"Error: No Kubernetes manifest files found in {manifest_path}"
    
    # Apply each manifest file
    for manifest_file in manifest_files:
      logger.info(f"Applying manifest from {manifest_file}...")
      utils.create_from_yaml(client.ApiClient(), manifest_file, namespace=namespace)
    
    logger.info("All Kubernetes resources applied successfully.")

    # Get URL from ingress
    try:
      time.sleep(K8S_INGRESS_TIMEOUT)
      ingresses = networking_api.list_namespaced_ingress(namespace)
      if not ingresses.items:
        logger.info("No ingress found")
        return f"Kubernetes manifests applied successfully, but no ingress was found."
      
      ingress = ingresses.items[0]
      host = ingress.spec.rules[0].host
      logger.info(f"Ingress URL: http://{host}")
      
      # Ensure /etc/hosts entry for the host
      ensure_etc_hosts_entry(host)
      
      return f"Kubernetes manifests applied successfully. Ingress URL: http://{host}"
    except ApiException as e:
      logger.info(f"No ingress found or error reading ingress: {e}")
      return f"Kubernetes manifests applied successfully, but could not retrieve ingress information: {str(e)}"
      
  except Exception as e:
    return f"Error applying Kubernetes manifests: {str(e)}"


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

tools = [clone_repo, prepare_repo_tree, get_file_content, write_file,
         build_docker_image, run_docker_container, apply_k8s_manifest, list_directory]

system_message = """You are a helpful assistant specialized in working with Git repositories.
You have access to tools that can help you with these tasks. When given a repository URL, you can:
1. Clone the repository and remove confusing files
2. Analyze the repository structure to identify important files
3. Retrieve the content of files you determine are necessary to understand the application
4. Write or modify files in the repository (e.g., Dockerfile, Kubernetes manifests)
5. Build and push Docker images based on the Dockerfile
6. Generate Kubernetes manifests for the application
7. List directory contents within the repository

You should use the clone_repo tool to clone a repository. The repository name can be extracted from the repository URL by taking the last part of the URL, removing the .git extension, and replacing dots with hyphens.
For example, for the URL \"https://github.com/run-rasztabiga-me/poc1-fastapi.git\", the repository name would be \"poc1-fastapi\".

You can use the prepare_repo_tree tool to get an overview of the repository structure if needed, but you should focus on identifying and examining files that are most relevant to understanding the application and creating the required outputs.

Use the get_file_content tool to retrieve the content of specific files that you determine are important. This tool requires the file path relative to the repository root.

You can use the write_file tool to create new files or modify existing ones in the repository. This tool requires the file path relative to the repository root and the content to write to the file. This is particularly useful for creating files like Dockerfile or Kubernetes manifests.

When creating a Dockerfile, carefully analyze the application code to ensure that any health check endpoint you specify actually exists in the application. For example, if you use HEALTHCHECK instruction, the endpoint should be implemented in the application code.

After creating a Dockerfile, you can use the build_docker_image tool to build and push a Docker image based on that Dockerfile. This tool requires the image tag. The image tag should follow the format \"localhost:5001/repository-name:tag\" (e.g., \"localhost:5001/poc1-fastapi:latest\").

After building a Docker image, generate Kubernetes manifests for the application. These manifests should:
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
- DO NOT create or include a namespace in the manifests - namespaces will be created automatically by the apply_k8s_manifest tool

Use the write_file tool to save these Kubernetes manifests in the repository.

After generating the Kubernetes manifests, you can use the apply_k8s_manifest tool to apply them to a Kubernetes cluster. This tool requires the path to the Kubernetes manifest file or directory relative to the repository root. You can provide either a single manifest file or a directory containing multiple YAML files. It will create a namespace for the application based on the repository name, delete any existing namespace with the same name, and apply all the manifests. If any manifest includes an Ingress resource, the tool will return the URL for accessing the application.

You can use the ls tool to list the contents of a directory within the cloned repository. This tool requires the directory path relative to the repository root. You can use an empty string or "." to list the contents of the repository root directory. The tool will display directories and files separately, with directories having a trailing slash and files showing their sizes in bytes. This is useful for exploring the repository structure in a more focused way than the prepare_repo_tree tool.

Given a repository URL from the user, you should automatically:
1. Clone the repository
2. Analyze the repository structure and find important files to understand the application
3. Create a Dockerfile for the application
4. Build a Docker image with the tag 'localhost:5001/[repository-name]:latest'
5. Generate appropriate Kubernetes manifests for the application
6. Apply those manifests to a Kubernetes cluster

The user will only provide the repository URL. You must handle all the remaining steps automatically without requesting additional information from the user.

IMPORTANT: You must continue the conversation until you have successfully generated a Dockerfile for the application, built a Docker image based on that Dockerfile, generated appropriate Kubernetes manifests for the application, AND applied those manifests to a Kubernetes cluster. After you have completed all these steps, respond with a message that includes the word \"DONE\" to indicate that you have completed the task.
"""

llm = init_chat_model(
    model="gpt-4.1-mini",
    temperature=0
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
  
  # Set log level for specific loggers to reduce noise
  logging.getLogger('urllib3').setLevel(logging.WARNING)
  logging.getLogger('docker').setLevel(logging.WARNING)
  
  # Reset global state
  REPO_NAME = None
  
  # Process the task
  
  # POC 1
  # repo_url = "https://github.com/run-rasztabiga-me/poc1-fastapi.git"                # passed
  # repo_url = "https://github.com/enriquecatala/fastapi-helloworld.git"              # passed
  # repo_url = "https://github.com/Azure-Samples/azd-simple-fastapi-appservice.git"   # passed
  # repo_url = "https://github.com/carvalhochris/fastapi-htmx-hello.git"              # passed
  # repo_url = "https://github.com/Sivasuthan9/fastapi-docker-optimized.git"          # passed
  # repo_url = "https://github.com/renceInbox/fastapi-todo.git"                       # passed

  # POC 2
  # repo_url = "https://github.com/run-rasztabiga-me/poc2-fastapi.git"                # passed
  # repo_url = "https://github.com/beerjoa/fastapi-postgresql-boilerplate.git"        # passed

  # POC X
  # repo_url = "https://github.com/igorbenav/FastAPI-boilerplate.git"                 # pending 

  logger.info("Starting agent with task: " + repo_url)
  
  # Stream agent responses
  for chunk in agent.stream({"messages": [{"role": "user", "content": repo_url}]},{"recursion_limit": 100},stream_mode="updates"):
    print(chunk)
    print("\n")
