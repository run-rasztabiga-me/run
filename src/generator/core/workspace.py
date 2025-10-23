"""Context-managed repository workspace with run-scoped isolation."""
import os
import shutil
import logging
from git import Repo
from typing import Optional, Generator, List, Tuple
from pathlib import Path

from .workspace_models import (
    RunContext,
    CloneResult,
    FileReadResult,
    FileWriteResult,
    DirectoryListResult,
    DirectoryItem,
    TreeResult,
)
from ...utils.repository_utils import extract_repo_name


class RepositoryWorkspace:
    """
    Context-managed repository workspace with run-scoped isolation.

    Each workspace instance:
    - Allocates a unique directory based on run_id
    - Provides typed file operation APIs
    - Supports future cleanup via context manager (currently disabled)
    - Is completely isolated from other workspaces

    Usage:
        run_context = RunContext(run_id=uuid, repo_name=name, timestamp=now)
        workspace = RepositoryWorkspace(run_context)
        workspace.clone_repository(repo_url)
        content = workspace.read_file("Dockerfile")
    """

    def __init__(self, run_context: RunContext, confusing_files: Optional[set] = None):
        """
        Initialize workspace for a specific run.

        Args:
            run_context: Run context with unique identifiers
            confusing_files: Set of files/directories to remove after cloning
        """
        self.run_context = run_context
        self.logger = logging.getLogger(__name__)
        self.confusing_files = confusing_files or {
            ".git", ".github", ".gitignore", ".gitmodules", ".gitattributes",
            ".gitlab-ci.yml", ".travis.yml", "LICENSE", "README.md", "CHANGELOG.md",
            "__pycache__", ".pytest_cache", ".coverage", "htmlcov", ".idea",
            ".vscode", "docker-compose.yml", "Dockerfile"
        }
        self._workspace_path: Optional[Path] = None
        self._is_cloned = False

    def __enter__(self) -> 'RepositoryWorkspace':
        """Enter context manager - workspace is ready."""
        return self

    # TODO
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager.
        Cleanup currently disabled - workspaces kept for debugging.
        """
        # Cleanup disabled per design decision
        # if self._workspace_path and self._workspace_path.exists():
        #     shutil.rmtree(self._workspace_path, ignore_errors=True)
        pass

    @property
    def workspace_path(self) -> Path:
        """
        Get the workspace directory path.

        If workspace directory exists from a previous clone (e.g., passed through pipeline),
        it will be detected and reused automatically.

        Returns:
            Absolute Path to workspace directory

        Raises:
            ValueError: If workspace directory doesn't exist
        """
        if not self._workspace_path:
            # Check if workspace exists from previous clone
            workspace_dir = self.run_context.workspace_dir.resolve()
            if workspace_dir.exists() and workspace_dir.is_dir():
                self._workspace_path = workspace_dir
                self._is_cloned = True
                self.logger.debug(f"Workspace exists from previous clone: {workspace_dir}")
            else:
                raise ValueError(
                    "Workspace path not initialized. "
                    "Please call clone_repository() first."
                )
        return self._workspace_path

    @property
    def repo_name(self) -> str:
        """Get the repository name from run context."""
        return self.run_context.repo_name

    def get_repo_path(self) -> Path:
        """
        Get the repository directory as an absolute Path object.

        Returns:
            Absolute Path object to the repository directory

        Raises:
            ValueError: If no repository has been cloned yet
        """
        return self.workspace_path

    def get_full_path(self, relative_path: str) -> Path:
        """
        Get the absolute full path for a file relative to the repository root.

        Args:
            relative_path: Path relative to the repository root

        Returns:
            Absolute Path object to the file

        Raises:
            ValueError: If no repository has been cloned yet
        """
        return self.workspace_path / relative_path

    def clone_repository(self, repo_url: str, cleanup: bool = True) -> CloneResult:
        """
        Clone repository into unique workspace and optionally remove confusing files.

        Args:
            repo_url: URL of the repository to clone
            cleanup: Whether to remove confusing files after cloning

        Returns:
            CloneResult with success status and workspace path
        """
        try:
            # Initialize workspace path
            workspace_dir = self.run_context.workspace_dir.resolve()
            self._workspace_path = workspace_dir

            self.logger.debug(f"Preparing workspace directory: {workspace_dir}")

            # Clean and create workspace directory
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir, ignore_errors=True)
            workspace_dir.mkdir(parents=True, exist_ok=True)

            # Clone repository
            self.logger.info(f"Cloning {repo_url} to {workspace_dir}")
            Repo.clone_from(repo_url, str(workspace_dir))

            if cleanup:
                self._cleanup_confusing_files(workspace_dir)

            self._is_cloned = True

            return CloneResult(
                success=True,
                repo_path=workspace_dir,
                repo_name=self.run_context.repo_name,
            )

        except Exception as e:
            self.logger.error(f"Failed to clone repository: {e}")
            return CloneResult(
                success=False,
                error=str(e),
            )

    def read_file(self, file_path: str) -> FileReadResult:
        """
        Read the content of a specific file from the repository.

        Args:
            file_path: Path to the file relative to the repository root

        Returns:
            FileReadResult with content or error
        """
        if not self._is_cloned:
            return FileReadResult(
                success=False,
                error="No repository has been cloned yet. Please clone a repository first.",
            )

        try:
            full_path = self.get_full_path(file_path)

            if not full_path.is_file():
                return FileReadResult(
                    success=False,
                    error=f"File {file_path} does not exist in the repository.",
                )

            self.logger.debug(f"Reading file: {full_path}")
            content = full_path.read_text(encoding="utf-8")

            return FileReadResult(
                success=True,
                content=content,
                path=full_path,
            )

        except Exception as e:
            return FileReadResult(
                success=False,
                error=f"Error reading file {file_path}: {str(e)}",
            )

    def write_file(self, file_path: str, content: str) -> FileWriteResult:
        """
        Write content to a file in the repository.

        Args:
            file_path: Path to the file relative to the repository root
            content: Content to write to the file

        Returns:
            FileWriteResult with success status and action taken
        """
        if not self._is_cloned:
            return FileWriteResult(
                success=False,
                error="No repository has been cloned yet. Please clone a repository first.",
            )

        try:
            full_path = self.get_full_path(file_path)
            action = "modified" if full_path.exists() else "created"

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.debug(f"Writing file: {full_path}")
            full_path.write_text(content, encoding="utf-8")

            return FileWriteResult(
                success=True,
                action=action,
                path=full_path,
            )

        except Exception as e:
            return FileWriteResult(
                success=False,
                error=f"Error writing to file {file_path}: {str(e)}",
            )

    def list_directory(self, dir_path: str = "") -> DirectoryListResult:
        """
        List files in a directory within the repository.

        Args:
            dir_path: Path to the directory relative to the repository root

        Returns:
            DirectoryListResult with list of items or error
        """
        if not self._is_cloned:
            return DirectoryListResult(
                success=False,
                error="No repository has been cloned yet. Please clone a repository first.",
            )

        try:
            if not dir_path or dir_path == '.':
                full_path = self.workspace_path
            else:
                full_path = self.get_full_path(dir_path)

            if not full_path.exists():
                return DirectoryListResult(
                    success=False,
                    error=f"Directory {dir_path} does not exist in the repository.",
                )

            if not full_path.is_dir():
                return DirectoryListResult(
                    success=False,
                    error=f"{dir_path} is not a directory.",
                )

            items = []
            for item in full_path.iterdir():
                if item.is_dir():
                    items.append(DirectoryItem(
                        name=item.name + "/",
                        is_directory=True,
                    ))
                else:
                    items.append(DirectoryItem(
                        name=item.name,
                        is_directory=False,
                        size_bytes=item.stat().st_size,
                    ))

            return DirectoryListResult(
                success=True,
                items=items,
                path=full_path,
            )

        except Exception as e:
            return DirectoryListResult(
                success=False,
                error=f"Error listing directory {dir_path}: {str(e)}",
            )

    def prepare_repo_tree(self) -> TreeResult:
        """
        Prepare repository tree structure as a string.

        Returns:
            TreeResult with tree structure or error
        """
        if not self._is_cloned:
            return TreeResult(
                success=False,
                error="No repository has been cloned yet. Please clone a repository first.",
            )

        try:
            self.logger.debug("Preparing tree...")
            tree_gen = self._tree(self.workspace_path)
            tree_str = self._tree_to_str(tree_gen, trim_dir=str(self.workspace_path))

            return TreeResult(
                success=True,
                tree_structure=tree_str,
            )

        except Exception as e:
            return TreeResult(
                success=False,
                error=f"Error generating tree: {str(e)}",
            )

    def _cleanup_confusing_files(self, workspace_dir: Path) -> None:
        """Remove confusing files and directories from the repository."""
        for root, dirs, files in os.walk(workspace_dir, topdown=True):
            for file in files:
                if file in self.confusing_files:
                    file_path = os.path.join(root, file)
                    self.logger.debug(f"Removing confusing file: {file_path}")
                    os.remove(file_path)

            for dir_name in list(dirs):
                if dir_name in self.confusing_files:
                    dir_path = os.path.join(root, dir_name)
                    self.logger.debug(f"Removing confusing directory: {dir_path}")
                    shutil.rmtree(dir_path, ignore_errors=True)
                    dirs.remove(dir_name)

    def _tree(self, some_dir: Path) -> Generator[Tuple[str, List[str], List[Tuple[str, int]]], None, None]:
        """Generate tree structure of directory."""
        for root, dirs, files in os.walk(some_dir):
            files_with_sizes = [
                (file, os.path.getsize(os.path.join(root, file)))
                for file in files
            ]
            yield root, dirs, files_with_sizes

    def _tree_to_str(
        self,
        tree_gen: Generator[Tuple[str, List[str], List[Tuple[str, int]]], None, None],
        trim_dir: str = None
    ) -> str:
        """Convert tree generator to string representation."""
        tree_str = ""
        for root, _, files_with_sizes in tree_gen:
            if trim_dir:
                root = root.replace(trim_dir, "")
            for file, size in files_with_sizes:
                tree_str += f"{root}/{file} - {size} bytes\n"
        return tree_str
