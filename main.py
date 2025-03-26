import json
import logging
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Generator

import docker
import requests
from dotenv import load_dotenv
from git import Repo
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException

from models import get_model

# Constants
CONFUSING_FILES = ["Dockerfile", "k8s.yaml", "docker-compose.yml", "docker-compose.yaml", "docker_compose", "k8s", ".jar", ".dockerignore", ".git", ".idea", ".vscode", ".fleet"]
DEFAULT_RETRIES = 15
DEFAULT_DELAY = 15
DOCKER_START_TIMEOUT = 5
K8S_INGRESS_TIMEOUT = 5

# Setup
load_dotenv()
logger = logging.getLogger(__name__)
docker_client = docker.from_env()

# model = get_model("openai/gpt-4o")
# model = get_model("openai/o3-mini-high")
model = get_model("google/gemini-2.0-flash-001")                  # działa naprawde bardzo dobrze, prawie zawsze sukces na poc2
# model = get_model("google/gemini-2.5-pro-exp-03-25:free")
# model = get_model("google/gemini-2.0-flash-thinking-exp:free")
# model = get_model("deepseek/deepseek-r1:free")
# model = get_model("meta-llama/llama-3.3-70b-instruct")
# model = get_model("anthropic/claude-3.5-haiku")
# model = get_model("anthropic/claude-3.7-sonnet")
# model = get_model("anthropic/claude-3.7-sonnet:thinking")
# model = get_model("deepseek/deepseek-chat")
# model = get_model("qwen/qwen2.5-vl-32b-instruct:free")
print(model)


def prepare_working_directory(tmp_dir: str) -> None:
    """
    Prepare working directory by clearing and creating it.
    
    Args:
        tmp_dir: Path to the temporary directory
    """
    logger.info("Preparing working directory...")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)


def clone_repo(repo_url: str, tmp_dir: str) -> None:
    """
    Clone repository and recursively remove confusing files.

    Args:
        repo_url: URL of the repository to clone
        tmp_dir: Path to the temporary directory

    Raises:
        git.exc.GitCommandError: If repository cloning fails
    """
    logger.info("Cloning repository...")
    Repo.clone_from(repo_url, tmp_dir)

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


def prepare_repo_tree_as_string(tmp_dir: str) -> str:
    """
    Prepare repository tree as string.
    
    Args:
        tmp_dir: Path to the temporary directory
        
    Returns:
        str: Repository tree as string
    """
    logger.info("Preparing tree...")
    dir_tree = tree(tmp_dir)
    return tree_to_str(dir_tree, trim_dir=tmp_dir)


def tree(some_dir: str) -> Generator[
    Tuple[str, List[str], List[Tuple[str, int]]], None, None]:
    """
    Generate tree structure of directory.

    Args:
        some_dir: Path to the directory

    Yields:
        Tuple of (root, dirs, files_with_sizes)
    """
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)

    for root, dirs, files in os.walk(some_dir):
        # Convert files to list of tuples with file sizes
        files_with_sizes = [(file, os.path.getsize(os.path.join(root, file))) for file in files]
        yield root, dirs, files_with_sizes


def tree_to_str(tree_gen: Generator[Tuple[str, List[str], List[Tuple[str, int]]], None, None],
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


def get_important_files(tree_str: str) -> List[str]:
    """
    Get important files from repository tree.
    
    Args:
        tree_str: Repository tree as string
        
    Returns:
        List of important files
        
    Raises:
        json.JSONDecodeError: If model response is not valid JSON
    """
    logger.info("Finding important files...")
    prompt = {"tree": tree_str}
    response = model.ask_model("get_important_files", json.dumps(prompt))
    files = json.loads(response)["files"]
    return files


def get_files_content(files: List[str], tmp_dir: str) -> Dict[str, str]:
    """
    Get content of files.
    
    Args:
        files: List of files to read
        tmp_dir: Path to the temporary directory
        
    Returns:
        Dictionary mapping file paths to their contents
        
    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    """
    logger.info("Preparing files content...")
    files_content = {}
    for file in files:
        with open(tmp_dir + "/" + file, "r") as f:
            files_content[file] = f.read()
    return files_content


def get_dockerfile(tree_str: str, files_content: Dict[str, str]) -> str:
    """
    Generate Dockerfile.
    
    Args:
        tree_str: Repository tree as string
        files_content: Dictionary mapping file paths to their contents
        
    Returns:
        str: Generated Dockerfile content
        
    Raises:
        json.JSONDecodeError: If model response is not valid JSON
    """
    logger.info("Generating Dockerfile...")
    prompt = {
        "tree": tree_str,
        "files_content": files_content
    }
    response = model.ask_model("get_dockerfile", json.dumps(prompt))
    return json.loads(response)["dockerfile"]


def write_dockerfile(tmp_dir: str, content: str) -> None:
    """
    Write Dockerfile to temporary directory.
    
    Args:
        tmp_dir: Path to the temporary directory
        content: Dockerfile content
    """
    logger.info("Writing Dockerfile...")
    with open(os.path.join(tmp_dir, "Dockerfile"), "w") as f:
        f.write(content)


def get_exposed_ports(dockerfile: str) -> List[str]:
    """
    Get exposed ports from Dockerfile.
    
    Args:
        dockerfile: Dockerfile content
        
    Returns:
        List of exposed ports
    """
    exposed_ports = []
    for line in dockerfile.split("\n"):
        if "EXPOSE" in line:
            exposed_ports = line.split(" ")[1:]
    return exposed_ports


def build_docker_image(tmp_dir: str, image_tag: str) -> docker.models.images.Image:
    """
    Build and push Docker image.
    
    Args:
        tmp_dir: Path to the temporary directory
        image_tag: Tag for the Docker image
        
    Returns:
        Built Docker image
        
    Raises:
        docker.errors.BuildError: If image build fails
        docker.errors.APIError: If image push fails
    """
    logger.info("Building Docker image...")
    image, _ = docker_client.images.build(path=tmp_dir, tag=image_tag, forcerm=True, pull=False)
    docker_client.images.push(image_tag)
    return image


def get_k8s_config(tree_str: str, files_content: Dict[str, str], dockerfile: str, image_tag: str) -> str:
    """
    Generate Kubernetes configuration.
    
    Args:
        tree_str: Repository tree as string
        files_content: Dictionary mapping file paths to their contents
        dockerfile: Dockerfile content
        image_tag: Tag for the Docker image
        
    Returns:
        str: Generated Kubernetes configuration
        
    Raises:
        json.JSONDecodeError: If model response is not valid JSON
    """
    logger.info("Preparing Kubernetes config...")
    prompt = {
        "tree": tree_str,
        "files_content": files_content,
        "dockerfile": dockerfile,
        "image_tag": image_tag
    }
    response = model.ask_model("get_k8s_config", json.dumps(prompt))
    return json.loads(response)["k8s_config"]


def write_k8s_config(tmp_dir: str, k8s_config: str) -> None:
    """
    Write Kubernetes configuration to file.
    
    Args:
        tmp_dir: Path to the temporary directory
        k8s_config: Kubernetes configuration content
    """
    logger.info("Writing k8s...")
    with open(os.path.join(tmp_dir, "k8s.yaml"), "w") as f:
        f.write(k8s_config)


def run_docker_image(image: docker.models.images.Image, exposed_ports: List[str]) -> docker.models.containers.Container:
    """
    Run Docker image.
    
    Args:
        image: Docker image to run
        exposed_ports: List of ports to expose
        
    Returns:
        Running Docker container
        
    Raises:
        docker.errors.APIError: If container cannot be started
    """
    logger.info("Running Docker image...")
    ports = {f"{port}/tcp": None for port in exposed_ports}
    container = docker_client.containers.run(image, detach=True, ports=ports)
    time.sleep(DOCKER_START_TIMEOUT)
    container.reload()
    return container


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


def apply_k8s(repo_name: str, tmp_dir: str) -> Optional[str]:
    """
    Apply Kubernetes configuration.
    
    Args:
        repo_name: Name of the repository
        tmp_dir: Path to the temporary directory
        
    Returns:
        str: Ingress URL if successful, None otherwise
        
    Raises:
        kubernetes.client.rest.ApiException: If Kubernetes operations fail
    """
    logger.info("Applying Kubernetes config...")
    
    # Load Kubernetes configuration
    kubeconfig = os.getenv("KUBECONFIG")
    config.load_kube_config(config_file=kubeconfig if kubeconfig else None)
    
    namespace = normalize_namespace(repo_name)
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
            raise

    # Wait for namespace to be deleted
    while True:
        try:
            api.read_namespace(name=namespace)
            logger.info("Waiting for namespace to delete...")
            time.sleep(2)
        except ApiException as e:
            if e.status == 404:
                logger.info("Namespace deleted.")
                break
            raise

    # Create namespace
    namespace_body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
    api.create_namespace(namespace_body)
    logger.info(f"Namespace '{namespace}' created.")

    # Apply k8s.yaml configuration
    k8s_yaml_path = os.path.join(tmp_dir, "k8s.yaml")
    utils.create_from_yaml(client.ApiClient(), k8s_yaml_path, namespace=namespace)
    logger.info("Kubernetes resources applied successfully.")

    # Get URL from ingress
    try:
        time.sleep(K8S_INGRESS_TIMEOUT)
        ingresses = networking_api.list_namespaced_ingress(namespace)
        if not ingresses.items:
            logger.error("No ingress found")
            return None
        ingress = ingresses.items[0]
        host = ingress.spec.rules[0].host
        logger.info(f"Ingress URL: http://{host}")
        return f"http://{host}"
    except ApiException as e:
        logger.error(f"Failed to read ingress: {e}")
        return None
    

def ensure_etc_hosts_entry(url: str) -> None:
    """
    Ensure /etc/hosts maps given URL to 127.0.0.1.
    Adds entry using `sudo` if needed.
    """
    logger.info("Ensuring /etc/hosts entry exists for %s", url)
    hostname = url.replace("http://", "").replace("https://", "").split("/")[0]

    try:
        with open("/etc/hosts", "r") as f:
            lines = f.readlines()
    except IOError as e:
        logger.error("Could not read /etc/hosts: %s", str(e))
        raise

    already_present = any(line.strip().endswith(hostname) and line.startswith("127.0.0.1") for line in lines)
    if already_present:
        logger.info("/etc/hosts already contains an entry for %s", hostname)
        return

    entry = f"127.0.0.1\t{hostname}"
    try:
        logger.info("Adding entry to /etc/hosts with sudo...")
        subprocess.run(
            ["sudo", "sh", "-c", f"echo '{entry}' >> /etc/hosts"],
            check=True
        )
        logger.info("Successfully added entry to /etc/hosts: %s", entry)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to add /etc/hosts entry via sudo: %s", e)
        raise


def test_deployment(url: str, retries: int = DEFAULT_RETRIES, delay: int = DEFAULT_DELAY) -> bool:
    """
    Test deployment by making HTTP requests.
    
    Args:
        url: URL to test
        retries: Number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        bool: True if deployment is successful, False otherwise
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            if response.status_code == 200 or response.status_code == 404: # some apps don't have default route
                logger.info(f"Deployment test successful on attempt {attempt}.")
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt} failed: {e}")

        if attempt < retries:
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    logger.error(f"Deployment test failed after {retries} attempts.")
    return False


def do_magic(repo_url: str) -> None:
    """
    Main function to process repository and deploy it.
    
    Args:
        repo_url: URL of the repository to process
        
    Raises:
        Exception: If any step fails
    """
    logger.info("Starting with repo_url: %s", repo_url)

    repo_name = repo_url.split("/")[-1].replace(".git", "").replace(".", "-")
    registry = os.getenv("REGISTRY_URL")
    if not registry:
        raise ValueError("REGISTRY_URL environment variable is not set")

    tmp_dir = f"./tmp/{repo_name}"

    try:
        prepare_working_directory(tmp_dir)
        clone_repo(repo_url, tmp_dir)

        tree_str = prepare_repo_tree_as_string(tmp_dir)
        print(tree_str)
        important_files = get_important_files(tree_str)
        logger.info("Important files identified: %s", important_files)

        files_content = get_files_content(important_files, tmp_dir)
        dockerfile = get_dockerfile(tree_str, files_content)
        write_dockerfile(tmp_dir, dockerfile)

        image_tag = f"{registry}/{repo_name.lower()}:latest"
        image = build_docker_image(tmp_dir, image_tag)

        exposed_ports = get_exposed_ports(dockerfile)
        container = run_docker_image(image, exposed_ports)

        k8s = get_k8s_config(tree_str, files_content, dockerfile, image_tag)
        write_k8s_config(tmp_dir, k8s)

        url = apply_k8s(repo_name, tmp_dir)
        if not url:
            raise RuntimeError("Failed to get ingress URL")

        ensure_etc_hosts_entry(url)

        if not test_deployment(url):
            raise RuntimeError("Deployment test failed")

        logger.info("DONE")
    except Exception as e:
        logger.error("Failed to process repository: %s", str(e))
        raise


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    # TODO chyba trzeba spreparować repozytoria proste w różnych technologiach (np. python, kotlin, js)

    # POC 1
    # repo_url = "https://github.com/run-rasztabiga-me/poc1-fastapi.git"                # passed
    # repo_url = "https://github.com/enriquecatala/fastapi-helloworld.git"              # passed
    # repo_url = "https://github.com/Azure-Samples/azd-simple-fastapi-appservice.git"   # passed
    # repo_url = "https://github.com/carvalhochris/fastapi-htmx-hello.git"              # passed
    # repo_url = "https://github.com/Sivasuthan9/fastapi-docker-optimized.git"          # passed
    # repo_url = "https://github.com/renceInbox/fastapi-todo.git"                       # passed

    # POC 2
    # repo_url = "https://github.com/run-rasztabiga-me/poc2-fastapi.git"                # passed
    repo_url = "https://github.com/beerjoa/fastapi-postgresql-boilerplate.git"          # pending


    # POC X
    # repo_url = "https://github.com/igorbenav/FastAPI-boilerplate.git"

    do_magic(repo_url)


if __name__ == "__main__":
    main()
