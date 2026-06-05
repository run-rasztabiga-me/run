#!/usr/bin/env python3
"""Create Git snapshots for repositories used in the paper experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import yaml


DEFAULT_CONFIGS = (
    Path("experiments/h1.yaml"),
    Path("experiments/h2.yaml"),
    Path("experiments/h5.yaml"),
)
DEFAULT_OUTPUT_DIR = Path("repo_dumps/paper_repositories")


@dataclass
class RepoEntry:
    """Repository URL with experiment membership metadata."""

    url: str
    experiments: set[str] = field(default_factory=set)

    @property
    def slug(self) -> str:
        return slug_from_url(self.url)


def normalize_url(url: str) -> str:
    """Normalize a Git URL enough to deduplicate config entries."""
    normalized = url.strip()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.rstrip("/")


def slug_from_url(url: str) -> str:
    """Convert a Git URL to a stable filesystem-safe token."""
    normalized = normalize_url(url)
    normalized = re.sub(r"^https?://", "", normalized)
    normalized = re.sub(r"^git@", "", normalized)
    normalized = normalized.replace(":", "/")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", normalized).strip("_")
    return slug or "repository"


def load_repo_entries(config_paths: Sequence[Path]) -> list[RepoEntry]:
    """Load and deduplicate repositories from experiment suite YAML files."""
    entries: dict[str, RepoEntry] = {}

    for config_path in config_paths:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {config_path}")

        experiments = data.get("experiments") or []
        if not isinstance(experiments, list):
            raise ValueError(f"Expected experiments list in {config_path}")

        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            experiment_name = str(experiment.get("name") or config_path.stem)
            repos = experiment.get("repos") or []
            if not isinstance(repos, list):
                raise ValueError(f"Expected repos list for {experiment_name} in {config_path}")

            for repo_url in repos:
                if not isinstance(repo_url, str) or not repo_url.strip():
                    continue
                key = normalize_url(repo_url)
                entry = entries.setdefault(key, RepoEntry(url=repo_url.strip()))
                entry.experiments.add(experiment_name)

    return sorted(entries.values(), key=lambda item: item.slug)


def run_git(args: Sequence[str], cwd: Path | None = None) -> None:
    """Run a git command and fail with useful context."""
    command = ["git", *args]
    subprocess.run(command, cwd=cwd, check=True)


def dump_repository(repo: RepoEntry, snapshot_dir: Path, update_existing: bool) -> dict[str, str]:
    """Clone or update a mirror and export it as a portable bundle."""
    mirror_dir = snapshot_dir / "mirrors" / f"{repo.slug}.git"
    bundle_path = snapshot_dir / "bundles" / f"{repo.slug}.bundle"
    mirror_dir.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    if mirror_dir.exists():
        if not update_existing:
            raise FileExistsError(
                f"Mirror already exists: {mirror_dir}. Use --update-existing to fetch updates."
            )
        run_git(["-C", str(mirror_dir), "remote", "set-url", "origin", repo.url])
        run_git(["-C", str(mirror_dir), "fetch", "--prune", "--tags", "origin"])
    else:
        run_git(["clone", "--mirror", repo.url, str(mirror_dir)])

    run_git(["-C", str(mirror_dir), "bundle", "create", str(bundle_path), "--all"])

    head_sha = git_output(["-C", str(mirror_dir), "rev-parse", "HEAD"])
    default_branch = git_output(["-C", str(mirror_dir), "symbolic-ref", "--short", "HEAD"], allow_failure=True)

    return {
        "url": repo.url,
        "slug": repo.slug,
        "experiments": ",".join(sorted(repo.experiments)),
        "mirror_path": str(mirror_dir),
        "bundle_path": str(bundle_path),
        "head_sha": head_sha,
        "default_branch": default_branch,
    }


def git_output(args: Sequence[str], allow_failure: bool = False) -> str:
    """Return stdout from a git command."""
    result = subprocess.run(["git", *args], check=not allow_failure, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def write_manifest(snapshot_dir: Path, snapshot_date: str, rows: Iterable[dict[str, str]]) -> None:
    """Write JSON and CSV manifests for the generated dump."""
    row_list = list(rows)
    manifest = {
        "snapshot_date": snapshot_date,
        "repository_count": len(row_list),
        "repositories": row_list,
    }

    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (snapshot_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slug",
                "url",
                "experiments",
                "head_sha",
                "default_branch",
                "bundle_path",
                "mirror_path",
            ],
        )
        writer.writeheader()
        writer.writerows(row_list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump repositories used by experiments/h1.yaml, h2.yaml and h5.yaml."
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        dest="configs",
        help="Experiment config to read. Can be passed more than once.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base directory for snapshots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--snapshot-date",
        default=date.today().isoformat(),
        help="Snapshot directory name/date. Default: today's local date.",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Fetch and refresh mirrors if the snapshot directory already contains them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print repositories that would be dumped.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_paths = args.configs or list(DEFAULT_CONFIGS)
    repos = load_repo_entries(config_paths)
    snapshot_dir = args.output_dir / args.snapshot_date

    if args.dry_run:
        print(f"Would dump {len(repos)} repositories into {snapshot_dir}:")
        for repo in repos:
            experiments = ", ".join(sorted(repo.experiments))
            print(f"- {repo.url} [{experiments}]")
        return 0

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, repo in enumerate(repos, start=1):
        print(f"[{index}/{len(repos)}] Dumping {repo.url}")
        rows.append(dump_repository(repo, snapshot_dir, update_existing=args.update_existing))

    write_manifest(snapshot_dir, args.snapshot_date, rows)
    print(f"Wrote dump manifest to {snapshot_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
