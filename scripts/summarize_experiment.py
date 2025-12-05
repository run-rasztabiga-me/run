#!/usr/bin/env python3
"""
Aggregate success metrics for a given experiment run.

Expected layout:
evaluation_reports/experiments/<experiment_name>/<run_id>/**/*.json
"""
import argparse
import json
import math
import pathlib
import sys
from typing import Dict, Iterable, Optional, Tuple


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float, str]:
    if total == 0:
        return 0.0, 0.0, "wilson"
    z = 1.96
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    radius = z * math.sqrt((phat * (1 - phat) / total) + (z * z / (4 * total * total))) / denom
    return max(0.0, center - radius), min(1.0, center + radius), "wilson"


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float, str]:
    if total == 0:
        return 0.0, 0.0, "clopper_pearson"
    if not _scipy_available():
        return wilson_interval(successes, total, alpha)
    from scipy.stats import beta

    lower = beta.ppf(alpha / 2, successes, total - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    if successes == 0:
        lower = 0.0
    if successes == total:
        upper = 1.0
    return lower, upper, "clopper_pearson"


def _scipy_available() -> bool:
    try:
        import importlib

        importlib.import_module("scipy")
        return True
    except Exception:
        return False


def load_runs(base: pathlib.Path) -> Iterable[Dict]:
    for path in base.rglob("*.json"):
        if path.name == "status.json":
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if not data.get("repo_name"):
            continue
        yield data


def stage_value(run: Dict, stage: str) -> Optional[bool]:
    if stage == "build":
        return bool(run.get("build_success"))
    if stage == "runtime":
        return bool(run.get("runtime_success"))
    if stage == "apply":
        qb = ((run.get("quality_metrics") or {}).get("scoring_breakdown") or {})
        ka = (qb.get("kubernetes") or {}).get("phases", {}).get("kubernetes_apply") or {}
        if not ka:
            return None
        return ka.get("errors", 1) == 0
    return None


def compute_stats(base: pathlib.Path) -> Dict:
    stage_counts = {"build": 0, "apply": 0, "runtime": 0}
    stage_success = {"build": 0, "apply": 0, "runtime": 0}
    prompt_counts: Dict[str, int] = {}
    prompt_success: Dict[str, int] = {}
    per_repo_counts: Dict[str, Dict[str, int]] = {}

    total_runs = 0
    total_success = 0

    for run in load_runs(base):
        total_runs += 1
        repo = run.get("repo_name")
        prompt = run.get("prompt_id") or "unknown"
        per_repo_counts.setdefault(repo, {"success": 0, "total": 0})
        per_repo_counts[repo]["total"] += 1
        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1

        build = stage_value(run, "build")
        apply_ok = stage_value(run, "apply")
        runtime = stage_value(run, "runtime")

        for key, val in (("build", build), ("apply", apply_ok), ("runtime", runtime)):
            if val is None:
                continue
            stage_counts[key] += 1
            if val:
                stage_success[key] += 1

        final_success = bool(build) and bool(runtime) and (apply_ok is True)
        if final_success:
            total_success += 1
            per_repo_counts[repo]["success"] += 1
            prompt_success[prompt] = prompt_success.get(prompt, 0) + 1

    return {
        "total_runs": total_runs,
        "total_success": total_success,
        "stage_counts": stage_counts,
        "stage_success": stage_success,
        "prompt_counts": prompt_counts,
        "prompt_success": prompt_success,
        "per_repo_counts": per_repo_counts,
    }


def rate_and_ci(successes: int, total: int, alpha: float = 0.05) -> Dict:
    if total == 0:
        return {"rate": None, "ci": None, "method": None}
    rate = successes / total
    lower, upper, method = clopper_pearson(successes, total, alpha)
    return {
        "rate": rate,
        "ci": (lower, upper),
        "method": method,
    }


def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument("run_id", help="Run ID directory, e.g. 20251205_215015")
    parser.add_argument(
        "--experiment",
        default="poc",
        help="Experiment name under evaluation_reports/experiments (default: poc)",
    )
    parser.add_argument(
        "--base-dir",
        default="../evaluation_reports/experiments",
        help="Base directory for experiments (default: evaluation_reports/experiments)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for confidence intervals (default: 0.05 -> 95%% CI)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable summary",
    )
    args = parser.parse_args()

    base = pathlib.Path(args.base_dir) / args.experiment / args.run_id
    if not base.exists():
        print(f"Path not found: {base}", file=sys.stderr)
        return 1

    stats = compute_stats(base)
    overall = rate_and_ci(stats["total_success"], stats["total_runs"], args.alpha)

    if args.json:
        out = {
            "base_path": str(base),
            "total_runs": stats["total_runs"],
            "total_success": stats["total_success"],
            "overall": overall,
            "stages": {},
            "prompts": {},
            "per_repo": {},
        }
        for stage in ("build", "apply", "runtime"):
            out["stages"][stage] = rate_and_ci(
                stats["stage_success"].get(stage, 0),
                stats["stage_counts"].get(stage, 0),
                args.alpha,
            )
        for prompt, total in stats["prompt_counts"].items():
            succ = stats["prompt_success"].get(prompt, 0)
            out["prompts"][prompt] = rate_and_ci(succ, total, args.alpha)
            out["prompts"][prompt]["count"] = total
            out["prompts"][prompt]["successes"] = succ
        for repo, cnt in stats["per_repo_counts"].items():
            out["per_repo"][repo] = rate_and_ci(cnt["success"], cnt["total"], args.alpha)
            out["per_repo"][repo]["count"] = cnt["total"]
            out["per_repo"][repo]["successes"] = cnt["success"]
        print(json.dumps(out, indent=2))
        return 0

    print(f"Experiment path: {base}")
    print(f"Runs: {stats['total_success']}/{stats['total_runs']} = {format_pct(overall['rate'])}")
    if overall["ci"]:
        low, high = overall["ci"]
        print(f"  {int((1-args.alpha)*100)}% CI ({overall['method']}): {format_pct(low)}–{format_pct(high)}")

    print("\nStages:")
    for stage in ("build", "apply", "runtime"):
        succ = stats["stage_success"].get(stage, 0)
        tot = stats["stage_counts"].get(stage, 0)
        stage_ci = rate_and_ci(succ, tot, args.alpha)
        print(f"  {stage}: {succ}/{tot} = {format_pct(stage_ci['rate'])}", end="")
        if stage_ci["ci"]:
            low, high = stage_ci["ci"]
            print(f" (CI: {format_pct(low)}–{format_pct(high)}, {stage_ci['method']})")
        else:
            print()

    print("\nPrompts:")
    for prompt, tot in stats["prompt_counts"].items():
        succ = stats["prompt_success"].get(prompt, 0)
        ci = rate_and_ci(succ, tot, args.alpha)
        line = f"  {prompt}: {succ}/{tot} = {format_pct(ci['rate'])}"
        if ci["ci"]:
            low, high = ci["ci"]
            line += f" (CI: {format_pct(low)}–{format_pct(high)}, {ci['method']})"
        print(line)

    print("\nPer repo:")
    for repo, cnt in stats["per_repo_counts"].items():
        succ = cnt["success"]
        tot = cnt["total"]
        ci = rate_and_ci(succ, tot, args.alpha)
        line = f"  {repo}: {succ}/{tot} = {format_pct(ci['rate'])}"
        if ci["ci"]:
            low, high = ci["ci"]
            line += f" (CI: {format_pct(low)}–{format_pct(high)}, {ci['method']})"
        print(line)

    if not _scipy_available():
        print("\nNote: scipy not available, Wilson intervals used as fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
