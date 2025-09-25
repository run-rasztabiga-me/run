"""Utility functions for repository operations."""


def extract_repo_name(repo_url: str) -> str:
    """Extract repository name from URL.

    Args:
        repo_url: Repository URL (e.g., 'https://github.com/user/repo.git')

    Returns:
        Repository name with .git removed and dots replaced with hyphens
        (e.g., 'repo' or 'repo-name')
    """
    return repo_url.split("/")[-1].replace(".git", "").replace(".", "-")