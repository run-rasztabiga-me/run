import base64
import json
import logging
import os
import re
import fnmatch
from pathlib import Path
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
            self.create_search_files_tool(),
            self.create_find_files_tool(),
            self.create_base64_encode_tool(),
            self.create_base64_decode_tool(),
            self.create_patch_file_tool(),
            self.create_think_tool(),
        ]

    def create_clone_repo_tool(self):
        """Create clone repository tool."""
        workspace = self.workspace
        logger = self.logger

        @tool("clone_repo", args_schema=ToolSchemas.CloneRepoInput)
        def clone_repo(repo_url: str) -> str:
            """Clone a Git repository from the given URL and clean up unnecessary files (node_modules, .git, build artifacts, etc.)."""
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
            """Get an overview of the repository structure as a tree. Use this to understand the project layout and identify important files."""
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
            """Create a new file or modify an existing file in the repository. Use this to generate Dockerfiles, Kubernetes manifests, and other configuration files."""
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
            """List all files and directories in the specified directory path. Shows directory names and file names with sizes. Use '.' or empty string for repository root."""
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

    def create_search_files_tool(self):
        """Create search files tool."""
        workspace = self.workspace
        logger = self.logger

        @tool("search_files", args_schema=ToolSchemas.SearchFilesInput)
        def search_files(pattern: str, file_pattern: str = None, case_sensitive: bool = False) -> str:
            """Search for a text pattern across files in the repository. Returns file paths and matching lines with context."""
            logger.info(f"🔧 Tool called: search_files(pattern={pattern}, file_pattern={file_pattern}, case_sensitive={case_sensitive})")

            if not workspace._is_cloned:
                return "Error: No repository has been cloned yet. Please clone a repository first."

            try:
                # Compile regex pattern
                regex_flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    compiled_pattern = re.compile(pattern, regex_flags)
                except re.error as e:
                    return f"Error: Invalid regex pattern: {e}"

                matches = []
                total_matches = 0
                search_path = workspace.workspace_path

                # Walk through repository using os.walk for compatibility
                for root, dirs, files in os.walk(search_path):
                    root_path = Path(root)

                    # Filter files by pattern if specified
                    if file_pattern:
                        # Extract just the filename part from patterns like **/*.py
                        if '/' in file_pattern:
                            file_pattern = file_pattern.split('/')[-1]
                        files = [f for f in files if fnmatch.fnmatch(f, file_pattern)]

                    for file in files:
                        file_path = root_path / file

                        # Skip binary files and very large files
                        try:
                            if file_path.stat().st_size > 1_000_000:  # Skip files > 1MB
                                continue

                            relative_path = str(file_path.relative_to(workspace.workspace_path))

                            # Read all lines first to enable context extraction
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()

                            # Search for matches and capture context
                            file_match_count = 0
                            for line_num, line in enumerate(lines, 1):
                                if compiled_pattern.search(line):
                                    # Get context lines (1 before, 1 after)
                                    context_before = lines[line_num - 2].rstrip() if line_num > 1 else None
                                    context_after = lines[line_num].rstrip() if line_num < len(lines) else None

                                    # Format with context
                                    match_text = f"{relative_path}:{line_num}: {line.rstrip()}"
                                    if context_before is not None:
                                        match_text = f"{relative_path}:{line_num - 1}- {context_before}\n" + match_text
                                    if context_after is not None:
                                        match_text += f"\n{relative_path}:{line_num + 1}+ {context_after}"

                                    matches.append(match_text)
                                    total_matches += 1
                                    file_match_count += 1

                                    # Limit matches per file
                                    if file_match_count >= 10:
                                        break

                        except (OSError, UnicodeDecodeError):
                            continue

                    # Limit total matches
                    if total_matches >= 100:
                        break

                if not matches:
                    return f"No matches found for pattern: {pattern}"

                result = f"Found {total_matches} match(es) for pattern '{pattern}':\n\n"
                result += "\n\n".join(matches)

                if total_matches >= 100:
                    result += "\n\n(Showing first 100 matches, there may be more)"

                return result

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return f"Error during search: {e}"

        return search_files

    def create_base64_encode_tool(self):
        """Create base64 encode tool."""
        logger = self.logger

        @tool("base64_encode", args_schema=ToolSchemas.Base64EncodeInput)
        def base64_encode(content: str) -> str:
            """Encode plain text content to base64. Useful for Kubernetes Secrets and other base64-encoded data."""
            logger.info(f"🔧 Tool called: base64_encode(content_length={len(content)})")
            try:
                encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
                return encoded
            except Exception as e:
                logger.error(f"Base64 encoding failed: {e}")
                return f"Error: Failed to encode content: {e}"

        return base64_encode

    def create_base64_decode_tool(self):
        """Create base64 decode tool."""
        logger = self.logger

        @tool("base64_decode", args_schema=ToolSchemas.Base64DecodeInput)
        def base64_decode(encoded: str) -> str:
            """Decode base64-encoded string to plain text. Useful for reading Kubernetes Secrets and other base64-encoded data."""
            logger.info(f"🔧 Tool called: base64_decode(encoded_length={len(encoded)})")
            try:
                decoded = base64.b64decode(encoded).decode('utf-8')
                return decoded
            except Exception as e:
                logger.error(f"Base64 decoding failed: {e}")
                return f"Error: Failed to decode content: {e}"

        return base64_decode

    def create_find_files_tool(self):
        """Create find files tool."""
        workspace = self.workspace
        logger = self.logger

        @tool("find_files", args_schema=ToolSchemas.FindFilesInput)
        def find_files(pattern: str, max_results: int = 50) -> str:
            """Find files by filename pattern. Supports glob patterns like '*.py', 'Dockerfile*', 'package.json'."""
            logger.info(f"🔧 Tool called: find_files(pattern={pattern}, max_results={max_results})")

            if not workspace._is_cloned:
                return "Error: No repository has been cloned yet. Please clone a repository first."

            try:
                matches = []
                search_path = workspace.workspace_path

                # Walk through repository
                for root, dirs, files in os.walk(search_path):
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]

                    root_path = Path(root)
                    for file in files:
                        # Match against filename pattern
                        if fnmatch.fnmatch(file, pattern):
                            file_path = root_path / file
                            relative_path = str(file_path.relative_to(search_path))
                            matches.append(relative_path)

                            # Limit results
                            if len(matches) >= max_results:
                                break

                    if len(matches) >= max_results:
                        break

                if not matches:
                    return f"No files found matching pattern: {pattern}"

                result = f"Found {len(matches)} file(s) matching pattern '{pattern}':\n\n"
                result += "\n".join(matches)

                if len(matches) >= max_results:
                    result += f"\n\n(Showing first {max_results} results, there may be more)"

                return result

            except Exception as e:
                logger.error(f"Find files failed: {e}")
                return f"Error during file search: {e}"

        return find_files

    def create_patch_file_tool(self):
        """Create patch file tool."""
        workspace = self.workspace
        logger = self.logger

        @tool("patch_file", args_schema=ToolSchemas.PatchFileInput)
        def patch_file(file_path: str, patch: str) -> str:
            """Apply a unified diff patch to a file. Useful for making targeted edits without rewriting entire files."""
            # Remove leading slash if present
            if file_path.startswith('/'):
                file_path = file_path.lstrip('/')

            logger.info(f"🔧 Tool called: patch_file(file_path={file_path}, patch_length={len(patch)})")

            if not workspace._is_cloned:
                return "Error: No repository has been cloned yet. Please clone a repository first."

            try:
                # Read the original file
                read_result = workspace.read_file(file_path)
                if not read_result.success:
                    return f"Error: {read_result.error}"

                original_lines = read_result.content.splitlines(keepends=True)

                # Parse the patch
                patch_lines = patch.splitlines(keepends=True)

                # Simple patch application (supports unified diff format)
                # Find @@ hunk headers and apply changes
                result_lines = original_lines.copy()
                i = 0

                while i < len(patch_lines):
                    line = patch_lines[i]

                    # Look for hunk header: @@ -start,count +start,count @@
                    if line.startswith('@@'):
                        import re
                        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                        if not match:
                            return f"Error: Invalid patch format at line {i+1}"

                        old_start = int(match.group(1)) - 1  # Convert to 0-indexed
                        new_start = int(match.group(3)) - 1  # Convert to 0-indexed

                        i += 1

                        # Collect changes from this hunk
                        old_lines_to_remove = []
                        new_lines_to_add = []
                        context_lines = []

                        while i < len(patch_lines) and not patch_lines[i].startswith('@@'):
                            pline = patch_lines[i]

                            if pline.startswith('-'):
                                old_lines_to_remove.append(pline[1:])
                            elif pline.startswith('+'):
                                new_lines_to_add.append(pline[1:])
                            elif pline.startswith(' '):
                                # Context line - track for verification
                                context_lines.append(pline[1:])

                            i += 1

                        # Apply the hunk: remove old lines and insert new ones
                        if old_lines_to_remove:
                            # Find and remove the old lines
                            for old_line in old_lines_to_remove:
                                if old_start < len(result_lines) and result_lines[old_start] == old_line:
                                    result_lines.pop(old_start)
                                else:
                                    # Try to find nearby
                                    found = False
                                    for offset in range(-2, 3):
                                        idx = old_start + offset
                                        if 0 <= idx < len(result_lines) and result_lines[idx] == old_line:
                                            result_lines.pop(idx)
                                            found = True
                                            break
                                    if not found:
                                        return f"Error: Could not find line to remove at position {old_start+1}"

                        # Insert new lines
                        for j, new_line in enumerate(new_lines_to_add):
                            result_lines.insert(new_start + j, new_line)
                    else:
                        i += 1

                # Write the patched content back
                patched_content = ''.join(result_lines)
                write_result = workspace.write_file(file_path, patched_content)

                if write_result.success:
                    return f"Successfully patched file: {file_path}"
                else:
                    return f"Error writing patched file: {write_result.error}"

            except Exception as e:
                logger.error(f"Patch application failed: {e}")
                return f"Error applying patch: {e}"

        return patch_file

    def create_think_tool(self):
        """Create think/reflection tool."""
        logger = self.logger

        @tool("think", args_schema=ToolSchemas.ThinkInput)
        def think(thoughts: str) -> str:
            """
            A space for reflection and planning. Use this to organize your thoughts about:
            - What you've learned from exploring the repository
            - Key observations about the application architecture
            - Your strategy for creating the Dockerfile and Kubernetes manifests
            - Potential issues or edge cases you've identified
            - Next steps in your analysis

            This tool doesn't modify anything - it's purely for your internal reasoning process.
            """
            logger.info(f"🧠 Agent reflection ({len(thoughts)} chars)")
            logger.debug(f"Thinking: {thoughts[:200]}...")  # Log first 200 chars at debug level

            # Return a simple acknowledgment
            return "Thoughts recorded. Continue with your analysis and file generation."

        return think
