import json
import logging
import os
import shutil
import time

import docker
import requests
from dotenv import load_dotenv
from git import Repo
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException

from models import get_model

load_dotenv()

logger = logging.getLogger(__name__)

docker_client = docker.from_env()

model = get_model("openai/gpt-4o")
# model = get_model("openai/o3-mini-high")
# model = get_model("google/gemini-2.0-flash-001")
# model = get_model("deepseek/deepseek-r1:free")
# model = get_model("meta-llama/llama-3.3-70b-instruct")
# model = get_model("anthropic/claude-3.5-haiku")
# model = get_model("anthropic/claude-3.7-sonnet")
# model = get_model("anthropic/claude-3.7-sonnet:thinking")
# model = get_model("deepseek/deepseek-chat")
print(model)


def prepare_working_directory(tmp_dir):
    logger.info("Preparing working directory...")

    # clear dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # create dir
    os.makedirs(tmp_dir, exist_ok=True)


def clone_repo(repo_url, tmp_dir):
    logger.info("Cloning repository...")

    # clone repo
    Repo.clone_from(repo_url, tmp_dir)

    confusing_files = ["Dockerfile", "k8s.yaml", "docker-compose.yml", "docker-compose.yaml", "k8s"]
    for file in confusing_files:
        path = os.path.join(tmp_dir, file)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def prepare_repo_tree_as_string(tmp_dir):
    logger.info("Preparing tree...")

    # get tree ignoring .git
    dir_tree = tree(tmp_dir, level=1, ignore=[".git"])

    # tree to string, ignoring .git
    tree_str = tree_to_str(dir_tree, trim_dir=tmp_dir)

    return tree_str


def tree(some_dir, level, ignore):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
        for i in ignore:
            if i in dirs:
                dirs.remove(i)


def tree_to_str(tree, trim_dir=None):
    tree_str = ""
    for root, dirs, files in tree:
        if trim_dir:
            root = root.replace(trim_dir, "")
        for file in files:
            tree_str += f"{root}/{file}\n"
    return tree_str


def get_important_files(tree_str):
    logger.info("Finding important files...")

    prompt = {
        "tree": tree_str,
    }

    response = model.ask_model(
        "get_important_files",
        json.dumps(prompt)
    )

    files = json.loads(response)["files"]

    # ignore unsupported files
    ignored_files = [".jar", "Dockerfile", "k8s.yaml"]
    files = [file for file in files if not any(ignored in file for ignored in ignored_files)]

    return files


def get_files_content(files, tmp_dir):
    logger.info("Preparing files content...")

    # get files content
    files_content = {}
    for file in files:
        with open(tmp_dir + "/" + file, "r") as f:
            files_content[file] = f.read()

    return files_content


def get_dockerfile(tree_str, files_content):
    logger.info("Generating Dockerfile...")

    prompt = {
        "tree": tree_str,
        "files_content": files_content
    }

    response = model.ask_model(
        "get_dockerfile",
        json.dumps(prompt)
    )

    return json.loads(response)["dockerfile"]


def write_dockerfile(tmp_dir, content):
    logger.info("Writing Dockerfile...")

    # write Dockerfile to tmp
    with open(tmp_dir + "/Dockerfile", "w") as f:
        f.write(content)


def get_exposed_ports(dockerfile):
    exposed_ports = []
    for line in dockerfile.split("\n"):
        if "EXPOSE" in line:
            exposed_ports = line.split(" ")[1:]
    return exposed_ports


def build_docker_image(tmp_dir, image_tag):
    logger.info("Building Docker image...")

    # build docker image
    image, logs = docker_client.images.build(path=tmp_dir, tag=image_tag, forcerm=True, pull=False)

    # push to registry
    docker_client.images.push(image_tag)

    return image


def get_k8s_config(tree_str, files_content, dockerfile, image_tag):
    logger.info("Preparing Kubernetes config...")

    prompt = {
        "tree": tree_str,
        "files_content": files_content,
        "dockerfile": dockerfile,
        "image_tag": image_tag
    }

    response = model.ask_model(
        "get_k8s_config",
        json.dumps(prompt)
    )


    return json.loads(response)["k8s_config"]


def write_k8s_config(tmp_dir, k8s_config):
    logger.info("Writing k8s...")

    # write to file
    with open(tmp_dir + "/k8s.yaml", "w") as f:
        f.write(k8s_config)


def run_docker_image(image, exposed_ports):
    logger.info("Running Docker image...")

    # run docker image, expose ports according to Dockerfile
    ports = {}
    for port in exposed_ports:
        ports[port + '/tcp'] = None

    container = docker_client.containers.run(image, detach=True, ports=ports)

    time.sleep(5)  # wait for container to start

    container.reload()

    return container


def apply_k8s(repo_name, tmp_dir):
    logger.info("Applying Kubernetes config...")

    # Load Kubernetes configuration
    kubeconfig = os.getenv("KUBECONFIG")
    if kubeconfig:
        config.load_kube_config(config_file=kubeconfig)
    else:
        config.load_kube_config()

    namespace = repo_name.lower()

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

    # Wait for namespace to be deleted completely
    import time
    while True:
        try:
            api.read_namespace(name=namespace)
            logger.info("Waiting for namespace to delete...")
            time.sleep(2)
        except ApiException as e:
            if e.status == 404:
                logger.info("Namespace deleted.")
                break
            else:
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
        time.sleep(3)  # Wait briefly to ensure ingress is created
        ingresses = networking_api.list_namespaced_ingress(namespace)
        ingress = ingresses.items[0]
        host = ingress.spec.rules[0].host
        logger.info(f"Ingress URL: http://{host}")
        print(f"Ingress URL: http://{host}")
        return f"http://{host}"
    except ApiException as e:
        logger.error(f"Failed to read ingress: {e}")

    # TODO write url to etc/hosts?


def test_deployment(url, retries=5, delay=5):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            if response.status_code == 200:
                logger.info(f"Deployment test successful on attempt {attempt}.")
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt} failed: {e}")

        if attempt < retries:
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    logger.error(f"Deployment test failed after {retries} attempts.")
    return False


def do_magic(repo_url):
    logger.info("Starting with repo_url: %s", repo_url)

    repo_name = repo_url.split("/")[-1].replace(".git", "").replace(".", "-")
    registry = os.getenv("REGISTRY_URL")

    tmp_dir = f"./tmp/{repo_name}"

    prepare_working_directory(tmp_dir)
    clone_repo(repo_url, tmp_dir)

    tree_str = prepare_repo_tree_as_string(tmp_dir)

    important_files = get_important_files(tree_str)
    files_content = get_files_content(important_files, tmp_dir)

    dockerfile = get_dockerfile(tree_str, files_content)

    write_dockerfile(tmp_dir, dockerfile)

    image_tag = f"{registry}/{repo_name.lower()}:latest"
    image = build_docker_image(tmp_dir, image_tag)

    exposed_ports = get_exposed_ports(dockerfile)

    # container = run_docker_image(image, exposed_ports)

    k8s = get_k8s_config(tree_str, files_content, dockerfile, image_tag)

    write_k8s_config(tmp_dir, k8s)

    url = apply_k8s(repo_name, tmp_dir)

    test_deployment(url)

    logger.info("DONE")


def main():
    logging.basicConfig(level=logging.INFO)

    # TODO chyba trzeba spreparować repozytoria proste w różnych technologiach (np. python, kotlin, js)

    # POC 1
    # repo_url = "https://github.com/run-rasztabiga-me/poc1-fastapi.git" # passed

    # POC 2
    repo_url = "https://github.com/run-rasztabiga-me/poc2-fastapi.git" # pending

    do_magic(repo_url)


if __name__ == "__main__":
    main()
