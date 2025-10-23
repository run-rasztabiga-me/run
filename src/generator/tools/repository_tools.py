import logging
from langchain_core.tools import tool
from ..core.workspace import RepositoryWorkspace
from ..core.config import ToolSchemas


class RepositoryTools:
    """Factory class for creating repository-related LangChain tools."""

    def __init__(self, workspace: RepositoryWorkspace):
        self.workspace = workspace
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
        workspace = self.workspace
        logger = self.logger

        @tool("clone_repo", args_schema=ToolSchemas.CloneRepoInput)
        def clone_repo(repo_url: str) -> str:
            """Clone repository and recursively remove confusing files."""
            logger.info(f"🔧 Tool called: clone_repo(repo_url={repo_url})")
            result = workspace.clone_repository(repo_url, cleanup=True)

            if result.success:
                return f"Repository {repo_url} cloned successfully to {result.repo_path}"
            else:
                return f"Error: {result.error}"

        return clone_repo

    def create_prepare_repo_tree_tool(self):
        """Create prepare repository tree tool."""
        workspace = self.workspace
        logger = self.logger

        @tool("prepare_repo_tree", args_schema=ToolSchemas.PrepareRepoTreeInput)
        def prepare_repo_tree() -> str:
            """Prepare repository tree structure as a string."""
            logger.info(f"🔧 Tool called: prepare_repo_tree()")
            result = workspace.prepare_repo_tree()

            if result.success:
                return result.tree_structure
            else:
                return f"Error: {result.error}"

        return prepare_repo_tree

    def create_get_file_content_tool(self):
        """Create get file content tool."""
        workspace = self.workspace
        logger = self.logger

        @tool("get_file_content", args_schema=ToolSchemas.GetFileContentInput)
        def get_file_content(file_path: str) -> str:
            """Retrieve the content of a specific file from the repository."""
            # Remove leading slash if present to avoid path resolution issues
            if file_path.startswith('/'):
                file_path = file_path.lstrip('/')
            logger.info(f"🔧 Tool called: get_file_content(file_path={file_path})")
            result = workspace.read_file(file_path)

            if result.success:
                return result.content
            else:
                return f"Error: {result.error}"

        return get_file_content

    def create_write_file_tool(self):
        """Create write file tool."""
        workspace = self.workspace
        logger = self.logger

        @tool("write_file", args_schema=ToolSchemas.WriteFileInput)
        def write_file(file_path: str, content: str) -> str:
            """Write content to a file in the repository."""
            # Remove leading slash if present to avoid path resolution issues
            if file_path.startswith('/'):
                file_path = file_path.lstrip('/')
            logger.info(f"🔧 Tool called: write_file(file_path={file_path}, content_length={len(content)})")
            result = workspace.write_file(file_path, content)

            if result.success:
                return f"{result.action.capitalize()} file {file_path} successfully."
            else:
                return f"Error: {result.error}"

        return write_file

    def create_list_directory_tool(self):
        """Create list directory tool."""
        workspace = self.workspace
        logger = self.logger

        @tool("ls", args_schema=ToolSchemas.ListDirectoryInput)
        def list_directory(dir_path: str) -> str:
            """List files in a directory within the repository."""
            logger.info(f"🔧 Tool called: ls(dir_path={dir_path})")
            result = workspace.list_directory(dir_path)

            if not result.success:
                return f"Error: {result.error}"

            # Format output similar to original
            output = f"Contents of {dir_path or './'} directory:\n"

            directories = [item for item in result.items if item.is_directory]
            files = [item for item in result.items if not item.is_directory]

            if directories:
                output += "\nDirectories:\n"
                output += "\n".join(sorted(item.name for item in directories))

            if files:
                output += "\n\nFiles:\n"
                output += "\n".join(
                    sorted(f"{item.name} - {item.size_bytes} bytes" for item in files)
                )

            if not result.items:
                output += "(empty directory)"

            return output

        return list_directory