"""Command-line interface for the repository dataset builder."""

from __future__ import annotations

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.evaluator.experiments.dataset_builder import (
    RepositoryDataset,
    RepositoryDatasetBuilder,
    RepositoryFilters,
    default_dataset_path,
    load_model_specs,
    load_prompts,
)


def configure_logging(verbose: bool) -> None:
    """Configure basic logging for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s %(message)s",
    )


def parse_mapping(path: Optional[str]) -> Dict[str, Any]:
    """Load a dictionary from a YAML or JSON file if provided."""
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Override file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    try:
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse overrides from {file_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Override document at {file_path} must be a mapping.")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover GitHub repositories for evaluator experiments.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover",
        help="Discover repositories matching the provided filters and save a dataset JSON.",
    )
    discover.add_argument("--name", required=True, help="Dataset name (used for filename).")
    discover.add_argument("--query", help="Freeform GitHub search query to include.")
    discover.add_argument(
        "--language",
        action="append",
        dest="languages",
        default=[],
        help="Programming language filter (repeatable).",
    )
    discover.add_argument(
        "--topic",
        action="append",
        dest="topics",
        default=[],
        help="GitHub topic filter (repeatable).",
    )
    discover.add_argument("--min-stars", type=int, help="Minimum star count.")
    discover.add_argument("--max-stars", type=int, help="Maximum star count.")
    discover.add_argument(
        "--include-forks",
        action="store_true",
        help="Include forked repositories in the results.",
    )
    discover.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived repositories in the results.",
    )
    discover.add_argument("--limit", type=int, default=50, help="Maximum repositories to fetch.")
    discover.add_argument("--sort", default="stars", help="GitHub sort field (default: stars).")
    discover.add_argument("--order", default="desc", help="Sort order (default: desc).")
    discover.add_argument(
        "--output",
        help="Optional output path for the dataset JSON (defaults to evaluation_reports/datasets/<name>.json).",
    )
    discover.add_argument("--note", help="Optional note stored alongside dataset metadata.")
    discover.add_argument(
        "--selection-seed",
        type=int,
        help="Persist a custom selection seed for downstream sampling (defaults to timestamp if omitted).",
    )

    to_experiment = subparsers.add_parser(
        "to-experiment",
        help="Convert a dataset JSON into an experiment suite YAML.",
    )
    to_experiment.add_argument("--dataset", required=True, help="Path to a dataset JSON file.")
    to_experiment.add_argument("--output", required=True, help="Path to the experiment YAML to write.")
    to_experiment.add_argument("--name", required=True, dest="experiment_name", help="Experiment name.")
    to_experiment.add_argument("--description", help="Optional experiment description.")
    to_experiment.add_argument(
        "--models",
        required=True,
        help="Path to a YAML/JSON document containing a 'models' list.",
    )
    to_experiment.add_argument("--top-n", type=int, help="Take the first N repositories from the dataset.")
    to_experiment.add_argument(
        "--sample-size",
        type=int,
        help="Randomly sample this many repositories (uses dataset seed when present).",
    )
    to_experiment.add_argument("--sample-seed", type=int, help="Override the sampling seed.")
    to_experiment.add_argument("--repetitions", type=int, default=1, help="Experiment repetitions per repo/model.")
    to_experiment.add_argument(
        "--generator-overrides",
        help="Optional YAML/JSON file providing generator override mapping.",
    )
    to_experiment.add_argument(
        "--run-overrides",
        help="Optional YAML/JSON file providing run override mapping.",
    )
    to_experiment.add_argument(
        "--prompts",
        help="Optional YAML/JSON file containing a 'prompts' list.",
    )
    to_experiment.add_argument(
        "--cleanup",
        choices=["never", "per_run", "per_experiment"],
        default="per_experiment",
        help="Cleanup mode for experiment execution (default: per_experiment).",
    )

    return parser


def handle_discover(args: argparse.Namespace) -> int:
    filters = RepositoryFilters(
        query=args.query,
        languages=args.languages,
        topics=args.topics,
        min_stars=args.min_stars,
        max_stars=args.max_stars,
        include_forks=args.include_forks,
        include_archived=args.include_archived,
    )
    builder = RepositoryDatasetBuilder()

    dataset = builder.discover_and_save(
        name=args.name,
        filters=filters,
        limit=args.limit,
        sort=args.sort,
        order=args.order,
        output_path=args.output or default_dataset_path(args.name),
        selection_seed=args.selection_seed,
        note=args.note,
    )

    logging.info(
        "Discovered %d repositories (total count reported by GitHub: %d).",
        len(dataset.repositories),
        dataset.metadata.total_count,
    )
    return 0


def handle_to_experiment(args: argparse.Namespace) -> int:
    dataset = RepositoryDataset.load_json(args.dataset)
    models = load_model_specs(args.models)
    prompts = load_prompts(args.prompts) if args.prompts else None
    generator_overrides = parse_mapping(args.generator_overrides)
    run_overrides = parse_mapping(args.run_overrides)

    dataset.write_experiment_yaml(
        args.output,
        experiment_name=args.experiment_name,
        models=models,
        description=args.description,
        top_n=args.top_n,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        repetitions=args.repetitions,
        generator_overrides=generator_overrides or None,
        run_overrides=run_overrides or None,
        prompts=prompts,
        cleanup=args.cleanup,
    )
    logging.info("Experiment YAML written to %s", args.output)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        if args.command == "discover":
            return handle_discover(args)
        if args.command == "to-experiment":
            return handle_to_experiment(args)
    except KeyboardInterrupt:
        logging.info("Aborted.")
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.error("%s", exc)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
