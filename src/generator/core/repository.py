import os
import shutil
import logging
from git import Repo
from typing import List, Tuple, Generator, Optional
from pathlib import Path

from ...utils.repository_utils import extract_repo_name


class RepositoryManager:
    """Manages repository operations including cloning, file operations, and cleanup."""

    def __init__(self, confusing_files: Optional[set] = None):
        self.logger = logging.getLogger(__name__)
        self.confusing_files = confusing_files or {
            ".git", ".github", ".gitignore", ".gitmodules", ".gitattributes",
            ".gitlab-ci.yml", ".travis.yml", "LICENSE", "README.md", "CHANGELOG.md",
            "__pycache__", ".pytest_cache", ".coverage", "htmlcov", ".idea",
            ".vscode", "docker-compose.yml", "Dockerfile"
        }
        self._repo_name: Optional[str] = None
        self._tmp_dir: Optional[str] = None

    @property
    def repo_name(self) -> Optional[str]:
        """Get the current repository name."""
        return self._repo_name

    @property
    def tmp_dir(self) -> Optional[str]:
        """Get the current temporary directory path."""
        return self._tmp_dir

    def clone_repository(self, repo_url: str, cleanup: bool = True) -> str:
        """
        Clone repository and optionally remove confusing files.

        Args:
            repo_url: URL of the repository to clone
            cleanup: Whether to remove confusing files after cloning

        Returns:
            Success message with cloned directory path
        """
        repo_name = extract_repo_name(repo_url)
        self._repo_name = repo_name
        tmp_dir = f"./tmp/{repo_name}"
        self._tmp_dir = tmp_dir

        self.logger.info("Preparing working directory...")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        self.logger.info(f"Cloning repository {repo_url}...")
        Repo.clone_from(repo_url, tmp_dir)

        if cleanup:
            self._cleanup_confusing_files(tmp_dir)

        return f"Repository {repo_url} cloned successfully to {tmp_dir}"

    def get_file_content(self, file_path: str) -> str:
        """
        Retrieve the content of a specific file from the repository.

        Args:
            file_path: Path to the file relative to the repository root

        Returns:
            File content or error message
        """
        if not self._validate_repository():
            return "Error: No repository has been cloned yet. Please clone a repository first."

        file_path_full = os.path.join(self._tmp_dir, file_path)

        if not os.path.isfile(file_path_full):
            return f"Error: File {file_path} does not exist in the repository."

        try:
            self.logger.info(f"Reading file content: {file_path_full}")
            with open(file_path_full, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

    def write_file(self, file_path: str, content: str) -> str:
        """
        Write content to a file in the repository.

        Args:
            file_path: Path to the file relative to the repository root
            content: Content to write to the file

        Returns:
            Success or error message
        """
        if not self._validate_repository():
            return "Error: No repository has been cloned yet. Please clone a repository first."

        file_path_full = os.path.join(self._tmp_dir, file_path)
        os.makedirs(os.path.dirname(file_path_full), exist_ok=True)

        try:
            action = "Modified" if os.path.exists(file_path_full) else "Created"
            self.logger.info(f"Writing file: {file_path_full}")
            with open(file_path_full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"{action} file {file_path} successfully."
        except Exception as e:
            return f"Error writing to file {file_path}: {str(e)}"

    def list_directory(self, dir_path: str = "") -> str:
        """
        List files in a directory within the repository.

        Args:
            dir_path: Path to the directory relative to the repository root

        Returns:
            Formatted list of files and directories
        """
        if not self._validate_repository():
            return "Error: No repository has been cloned yet. Please clone a repository first."

        if not dir_path or dir_path == '.':
            dir_path_full = self._tmp_dir
        else:
            dir_path_full = os.path.join(self._tmp_dir, dir_path)

        if not os.path.exists(dir_path_full):
            return f"Error: Directory {dir_path} does not exist in the repository."

        if not os.path.isdir(dir_path_full):
            return f"Error: {dir_path} is not a directory."

        try:
            items = os.listdir(dir_path_full)
            result = f"Contents of {dir_path or './'} directory:\n"

            directories = []
            files = []

            for item in items:
                item_path = os.path.join(dir_path_full, item)
                if os.path.isdir(item_path):
                    directories.append(f"{item}/")
                else:
                    file_size = os.path.getsize(item_path)
                    files.append(f"{item} - {file_size} bytes")

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

    def prepare_repo_tree(self) -> str:
        """Prepare repository tree structure as a string."""
        if not self._validate_repository():
            return "Error: No repository has been cloned yet. Please clone a repository first."

        self.logger.info("Preparing tree...")
        dir_tree = self._tree(self._tmp_dir)
        return self._tree_to_str(dir_tree, trim_dir=self._tmp_dir)


    def _validate_repository(self) -> bool:
        """Validate that repository has been cloned and directory exists."""
        return (self._repo_name is not None
                and self._tmp_dir is not None
                and os.path.isdir(self._tmp_dir))

    def _cleanup_confusing_files(self, tmp_dir: str) -> None:
        """Remove confusing files and directories from the repository."""
        for root, dirs, files in os.walk(tmp_dir, topdown=True):
            for file in files:
                if file in self.confusing_files:
                    file_path = os.path.join(root, file)
                    self.logger.info(f"Removing confusing file: {file_path}")
                    os.remove(file_path)

            for dir_name in list(dirs):
                if dir_name in self.confusing_files:
                    dir_path = os.path.join(root, dir_name)
                    self.logger.info(f"Removing confusing directory: {dir_path}")
                    shutil.rmtree(dir_path, ignore_errors=True)
                    dirs.remove(dir_name)

    def _tree(self, some_dir: str) -> Generator[Tuple[str, List[str], List[Tuple[str, int]]], None, None]:
        """Generate tree structure of directory."""
        assert os.path.isdir(some_dir)

        for root, dirs, files in os.walk(some_dir):
            files_with_sizes = [
                (file, os.path.getsize(os.path.join(root, file)))
                for file in files
            ]
            yield root, dirs, files_with_sizes

    def _tree_to_str(self, tree_gen: Generator[Tuple[str, List[str], List[Tuple[str, int]]], None, None],
                     trim_dir: str = None) -> str:
        """Convert tree generator to string representation."""
        tree_str = ""
        for root, _, files_with_sizes in tree_gen:
            if trim_dir:
                root = root.replace(trim_dir, "")
            for file, size in files_with_sizes:
                tree_str += f"{root}/{file} - {size} bytes\n"
        return tree_str