import logging
from langchain_core.tools import tool
from ..core.repository import RepositoryManager
from ..core.config import ToolSchemas


class RepositoryTools:
    """Factory class for creating repository-related LangChain tools."""

    def __init__(self, repository_manager: RepositoryManager):
        self.repository_manager = repository_manager
        self.logger = logging.getLogger(__name__)

    def create_tools(self) -> list:
        """Create and return all repository tools."""
        return [
            self.create_clone_repo_tool(),
            self.create_prepare_repo_tree_tool(),
            self.create_get_file_content_tool(),
            self.create_write_file_tool(),
            self.create_list_directory_tool(),
        ]

    def create_clone_repo_tool(self):
        """Create clone repository tool."""
        repo_manager = self.repository_manager
        logger = self.logger

        @tool("clone_repo", args_schema=ToolSchemas.CloneRepoInput)
        def clone_repo(repo_url: str) -> str:
            """Clone repository and recursively remove confusing files."""
            logger.info(f"🔧 Tool called: clone_repo(repo_url={repo_url})")
            return repo_manager.clone_repository(repo_url, cleanup=True)

        return clone_repo

    def create_prepare_repo_tree_tool(self):
        """Create prepare repository tree tool."""
        repo_manager = self.repository_manager
        logger = self.logger

        @tool("prepare_repo_tree", args_schema=ToolSchemas.PrepareRepoTreeInput)
        def prepare_repo_tree() -> str:
            """Prepare repository tree structure as a string."""
            logger.info(f"🔧 Tool called: prepare_repo_tree()")
            return repo_manager.prepare_repo_tree()

        return prepare_repo_tree

    def create_get_file_content_tool(self):
        """Create get file content tool."""
        repo_manager = self.repository_manager
        logger = self.logger

        @tool("get_file_content", args_schema=ToolSchemas.GetFileContentInput)
        def get_file_content(file_path: str) -> str:
            """Retrieve the content of a specific file from the repository."""
            # Remove leading slash if present to avoid path resolution issues
            if file_path.startswith('/'):
                file_path = file_path.lstrip('/')
            logger.info(f"🔧 Tool called: get_file_content(file_path={file_path})")
            return repo_manager.get_file_content(file_path)

        return get_file_content

    def create_write_file_tool(self):
        """Create write file tool."""
        repo_manager = self.repository_manager
        logger = self.logger

        @tool("write_file", args_schema=ToolSchemas.WriteFileInput)
        def write_file(file_path: str, content: str) -> str:
            """Write content to a file in the repository."""
            # Remove leading slash if present to avoid path resolution issues
            if file_path.startswith('/'):
                file_path = file_path.lstrip('/')
            logger.info(f"🔧 Tool called: write_file(file_path={file_path}, content_length={len(content)})")
            return repo_manager.write_file(file_path, content)

        return write_file

    def create_list_directory_tool(self):
        """Create list directory tool."""
        repo_manager = self.repository_manager
        logger = self.logger

        @tool("ls", args_schema=ToolSchemas.ListDirectoryInput)
        def list_directory(dir_path: str) -> str:
            """List files in a directory within the repository."""
            logger.info(f"🔧 Tool called: ls(dir_path={dir_path})")
            return repo_manager.list_directory(dir_path)

        return list_directory