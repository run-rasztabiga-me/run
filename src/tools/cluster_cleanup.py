from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from src.generator.core.config import GeneratorConfig

logger = logging.getLogger(__name__)

DEFAULT_SKIP_NAMESPACES: Set[str] = {
    "container-registry",
    "default",
    "ingress",
    "kube-node-lease",
    "kube-public",
    "kube-system",
}


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """
    Execute a subprocess command and return the completed process.
    Raises CalledProcessError automatically when the command fails.
    """
    logger.debug("Executing command: %s", " ".join(command))
    return subprocess.run(command, check=True, capture_output=True, text=True)


def list_namespaces(kubectl_bin: str) -> List[str]:
    """Return a list of namespaces from the current Kubernetes context."""
    if shutil.which(kubectl_bin) is None:
        raise RuntimeError(f"kubectl binary '{kubectl_bin}' not found in PATH.")

    try:
        result = run_command([kubectl_bin, "get", "namespaces", "-o", "json"])
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown error"
        raise RuntimeError(f"Failed to list namespaces: {stderr}") from exc

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("kubectl returned invalid JSON when listing namespaces.") from exc

    namespaces = [item["metadata"]["name"] for item in data.get("items", [])]
    logger.debug("Discovered namespaces: %s", namespaces)
    return namespaces


def delete_namespaces(
    namespaces: Iterable[str],
    skip: Set[str],
    kubectl_bin: str,
    dry_run: bool = False,
) -> None:
    """Delete namespaces except those explicitly skipped."""
    for namespace in sorted(set(namespaces)):
        if namespace in skip:
            logger.info("Skipping protected namespace: %s", namespace)
            continue

        if dry_run:
            logger.info("[dry-run] Would delete namespace: %s", namespace)
            continue

        logger.info("Deleting namespace: %s", namespace)
        try:
            run_command([kubectl_bin, "delete", "namespace", namespace, "--wait=false"])
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown error"
            logger.warning("Failed to delete namespace '%s': %s", namespace, stderr)


def cleanup_hosts_file(
    cluster_ip: str,
    domain_suffix: str,
    dry_run: bool = False,
) -> bool:
    """
    Remove entries mapped to the cluster IP or managed domain suffix from /etc/hosts.

    Returns:
        True when cleanup is successful or no changes are needed, False otherwise.
    """
    hosts_path = Path("/etc/hosts")
    if not hosts_path.exists():
        logger.warning("/etc/hosts not found; skipping hosts cleanup.")
        return True

    try:
        original_content = hosts_path.read_text()
    except PermissionError:
        logger.error("Insufficient permissions to read /etc/hosts. Try running with sudo.")
        return False

    lines = original_content.splitlines()
    retained_lines: List[str] = []
    removed_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            retained_lines.append(line)
            continue

        tokens = stripped.split()
        ip = tokens[0]
        hosts = tokens[1:]

        matches_ip = bool(cluster_ip) and ip == cluster_ip
        matches_domain = bool(domain_suffix) and any(host.endswith(domain_suffix) for host in hosts)

        if matches_ip or matches_domain:
            removed_lines.append(line)
            continue

        retained_lines.append(line)

    if not removed_lines:
        logger.info("No /etc/hosts entries required cleanup.")
        return True

    logger.info(
        "Identified %d /etc/hosts entries for removal (cluster IP: %s, domain suffix: %s).",
        len(removed_lines),
        cluster_ip,
        domain_suffix,
    )
    for line in removed_lines:
        logger.debug("Removing hosts entry: %s", line.strip())

    if dry_run:
        logger.info("[dry-run] Skipping /etc/hosts modification.")
        return True

    repo_root = Path(__file__).resolve().parents[2]
    backup_dir = repo_root / "tmp"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "hosts.cleanup.backup"

    try:
        backup_path.write_text(original_content)
        logger.info("Created /etc/hosts backup at %s", backup_path)
    except OSError as exc:
        logger.warning("Failed to create backup file at %s: %s", backup_path, exc)

    new_content = "\n".join(retained_lines)
    if retained_lines:
        new_content += "\n"

    try:
        hosts_path.write_text(new_content)
    except PermissionError:
        logger.error("Insufficient permissions to modify /etc/hosts. Re-run with sudo privileges.")
        return False
    except OSError as exc:
        logger.error("Failed to update /etc/hosts: %s", exc)
        return False

    logger.info("Successfully cleaned up /etc/hosts.")
    return True


def configure_logging(verbose: bool) -> None:
    """Configure root logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(message)s",
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Clean Kubernetes namespaces and related hosts entries.")
    parser.add_argument(
        "--kubectl",
        dest="kubectl_bin",
        default="kubectl",
        help="Path to the kubectl binary (default: kubectl).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without executing them.",
    )
    parser.add_argument(
        "--skip-namespace",
        action="append",
        default=[],
        help="Additional namespace to preserve. Can be provided multiple times.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def cleanup_cluster(
    *,
    kubectl_bin: str = "kubectl",
    skip_namespaces: Optional[Iterable[str]] = None,
    dry_run: bool = False,
    generator_config: Optional[GeneratorConfig] = None,
) -> bool:
    """
    Execute the namespace and hosts cleanup routine.

    Args:
        kubectl_bin: Path to the kubectl executable.
        skip_namespaces: Additional namespaces to preserve.
        dry_run: When True, only log intended actions.
        generator_config: Optional config supplying cluster IP and domain settings.

    Returns:
        True when cleanup succeeds, False when /etc/hosts could not be updated.

    Raises:
        RuntimeError: When the namespace listing fails or kubectl is missing.
    """
    skip_set = DEFAULT_SKIP_NAMESPACES.union(set(skip_namespaces or []))

    namespaces = list_namespaces(kubectl_bin)
    delete_namespaces(
        namespaces=namespaces,
        skip=skip_set,
        kubectl_bin=kubectl_bin,
        dry_run=dry_run,
    )

    config = generator_config or GeneratorConfig()
    hosts_cleaned = cleanup_hosts_file(
        cluster_ip=config.k8s_cluster_ip,
        domain_suffix=config.domain_suffix,
        dry_run=dry_run,
    )

    return hosts_cleaned


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the cleanup routine."""
    args = parse_args(argv or sys.argv[1:])
    configure_logging(args.verbose)

    try:
        hosts_cleaned = cleanup_cluster(
            kubectl_bin=args.kubectl_bin,
            skip_namespaces=args.skip_namespace,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    return 0 if hosts_cleaned else 2


if __name__ == "__main__":
    sys.exit(main())
