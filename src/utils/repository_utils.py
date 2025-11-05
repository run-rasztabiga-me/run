"""Utility functions for repository operations."""

import re


def extract_repo_name(repo_url: str) -> str:
    """Extract and normalize repository name from URL.

    Args:
        repo_url: Repository URL (e.g., 'https://github.com/user/repo.git')

    Returns:
        Repository name normalized for use in Docker image names, Kubernetes
        namespaces, and filesystem paths. Ensures lowercase characters, replaces
        dots with hyphens, and collapses invalid characters into single hyphens.
    """
    repo_token = repo_url.rstrip("/").split("/")[-1]
    if repo_token.endswith(".git"):
        repo_token = repo_token[:-4]

    normalized = repo_token.replace(".", "-").lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")

    return normalized or "repo"
