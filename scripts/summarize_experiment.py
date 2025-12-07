#!/usr/bin/env python3
"""
Aggregate success metrics for a given experiment run.

Expected layout:
evaluation_reports/experiments/<experiment_name>/<run_id>/**/*.json
"""
import argparse
import json
import pathlib
import sys
from typing import Dict, Iterable, Optional, Tuple


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float, str]:
    if total == 0:
        return 0.0, 0.0, "clopper_pearson"
    from scipy.stats import beta

    lower = beta.ppf(alpha / 2, successes, total - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    if successes == 0:
        lower = 0.0
    if successes == total:
        upper = 1.0
    return lower, upper, "clopper_pearson"


def load_runs(base: pathlib.Path) -> Iterable[Dict]:
    runs = []
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
        runs.append(data)

    # Sort by start_time (oldest first)
    runs.sort(key=lambda r: r.get("start_time", ""))

    for run in runs:
        yield run


def stage_value(run: Dict, stage: str) -> Optional[bool]:
    if stage == "build":
        return bool(run.get("build_success"))
    if stage == "runtime":
        return bool(run.get("runtime_success"))
    if stage == "apply":
        # Check if kubernetes manifests were generated
        gen_result = run.get("generation_result") or {}
        k8s_manifests = gen_result.get("k8s_manifests") or []
        if not k8s_manifests:
            return None  # No K8s manifests, so apply doesn't apply

        # If manifests exist, check if they were successfully validated/applied
        qb = ((run.get("quality_metrics") or {}).get("scoring_breakdown") or {})
        k8s_breakdown = qb.get("kubernetes") or {}

        # Check kubernetes_apply phase first (if it exists)
        phases = k8s_breakdown.get("phases") or {}
        ka = phases.get("kubernetes_apply")
        if ka:
            return ka.get("errors", 1) == 0

        # Fallback: check k8s_syntax validation
        k8s_syntax = phases.get("k8s_syntax")
        if k8s_syntax:
            return k8s_syntax.get("errors", 1) == 0

        # If we have manifests but no phase data, check overall k8s errors
        k8s_errors = k8s_breakdown.get("total_errors", 1)
        return k8s_errors == 0
    return None


def compute_stats(base: pathlib.Path) -> Dict:
    stage_counts = {"build": 0, "apply": 0, "runtime": 0}
    stage_success = {"build": 0, "apply": 0, "runtime": 0}
    # Conditional counts: given previous stage succeeded
    stage_conditional_counts = {"apply": 0, "runtime": 0}  # apply | build, runtime | apply
    stage_conditional_success = {"apply": 0, "runtime": 0}
    prompt_counts: Dict[str, int] = {}
    prompt_success: Dict[str, int] = {}
    per_repo_counts: Dict[str, Dict[str, int]] = {}
    per_model_counts: Dict[str, Dict[str, int]] = {}

    total_runs = 0
    total_success = 0

    for run in load_runs(base):
        total_runs += 1
        repo = run.get("repo_name")
        prompt = run.get("prompt_id") or "unknown"
        model = run.get("model_name") or "unknown"
        per_repo_counts.setdefault(repo, {"success": 0, "total": 0})
        per_repo_counts[repo]["total"] += 1
        per_model_counts.setdefault(model, {"success": 0, "total": 0})
        per_model_counts[model]["total"] += 1
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

        # Conditional statistics: apply given build succeeded
        if build is True and apply_ok is not None:
            stage_conditional_counts["apply"] += 1
            if apply_ok:
                stage_conditional_success["apply"] += 1

        # Conditional statistics: runtime given apply succeeded
        if apply_ok is True and runtime is not None:
            stage_conditional_counts["runtime"] += 1
            if runtime:
                stage_conditional_success["runtime"] += 1

        final_success = bool(build) and bool(runtime) and (apply_ok is True)
        if final_success:
            total_success += 1
            per_repo_counts[repo]["success"] += 1
            per_model_counts[model]["success"] += 1
            prompt_success[prompt] = prompt_success.get(prompt, 0) + 1

    return {
        "total_runs": total_runs,
        "total_success": total_success,
        "stage_counts": stage_counts,
        "stage_success": stage_success,
        "stage_conditional_counts": stage_conditional_counts,
        "stage_conditional_success": stage_conditional_success,
        "prompt_counts": prompt_counts,
        "prompt_success": prompt_success,
        "per_repo_counts": per_repo_counts,
        "per_model_counts": per_model_counts,
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
    parser.add_argument(
        "results_path",
        help="Full path to experiment results directory (e.g., evaluation_reports/experiments/h1/20251205_215015)",
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

    base = pathlib.Path(args.results_path)
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
            "per_model": {},
            "per_repo": {},
        }
        for stage in ("build", "apply", "runtime"):
            stage_data = rate_and_ci(
                stats["stage_success"].get(stage, 0),
                stats["stage_counts"].get(stage, 0),
                args.alpha,
            )
            # Add conditional stats for stages that depend on previous stage
            if stage in stats["stage_conditional_counts"]:
                stage_data["conditional"] = rate_and_ci(
                    stats["stage_conditional_success"].get(stage, 0),
                    stats["stage_conditional_counts"].get(stage, 0),
                    args.alpha,
                )
            out["stages"][stage] = stage_data
        for prompt, total in stats["prompt_counts"].items():
            succ = stats["prompt_success"].get(prompt, 0)
            out["prompts"][prompt] = rate_and_ci(succ, total, args.alpha)
            out["prompts"][prompt]["count"] = total
            out["prompts"][prompt]["successes"] = succ
        for model, cnt in stats["per_model_counts"].items():
            out["per_model"][model] = rate_and_ci(cnt["success"], cnt["total"], args.alpha)
            out["per_model"][model]["count"] = cnt["total"]
            out["per_model"][model]["successes"] = cnt["success"]
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
        print(f"  {int((1-args.alpha)*100)}% CI: {format_pct(low)}–{format_pct(high)}")

    print("\nStages:")
    for stage in ("build", "apply", "runtime"):
        succ = stats["stage_success"].get(stage, 0)
        tot = stats["stage_counts"].get(stage, 0)
        stage_ci = rate_and_ci(succ, tot, args.alpha)
        print(f"  {stage}: {succ}/{tot} = {format_pct(stage_ci['rate'])}", end="")
        if stage_ci["ci"]:
            low, high = stage_ci["ci"]
            print(f" (CI: {format_pct(low)}–{format_pct(high)})", end="")

        # Show conditional rate for stages that depend on previous stage
        if stage in stats["stage_conditional_counts"]:
            cond_succ = stats["stage_conditional_success"].get(stage, 0)
            cond_tot = stats["stage_conditional_counts"].get(stage, 0)
            cond_ci = rate_and_ci(cond_succ, cond_tot, args.alpha)
            prev_stage = "build" if stage == "apply" else "apply"
            print(f"\n    └─ given {prev_stage} OK: {cond_succ}/{cond_tot} = {format_pct(cond_ci['rate'])}", end="")
            if cond_ci["ci"]:
                low, high = cond_ci["ci"]
                print(f" (CI: {format_pct(low)}–{format_pct(high)})")
            else:
                print()
        else:
            print()

    print("\nPrompts:")
    for prompt, tot in stats["prompt_counts"].items():
        succ = stats["prompt_success"].get(prompt, 0)
        ci = rate_and_ci(succ, tot, args.alpha)
        line = f"  {prompt}: {succ}/{tot} = {format_pct(ci['rate'])}"
        if ci["ci"]:
            low, high = ci["ci"]
            line += f" (CI: {format_pct(low)}–{format_pct(high)})"
        print(line)

    print("\nPer model:")
    for model, cnt in stats["per_model_counts"].items():
        succ = cnt["success"]
        tot = cnt["total"]
        ci = rate_and_ci(succ, tot, args.alpha)
        line = f"  {model}: {succ}/{tot} = {format_pct(ci['rate'])}"
        if ci["ci"]:
            low, high = ci["ci"]
            line += f" (CI: {format_pct(low)}–{format_pct(high)})"
        print(line)

    print("\nPer repo:")
    for repo, cnt in stats["per_repo_counts"].items():
        succ = cnt["success"]
        tot = cnt["total"]
        ci = rate_and_ci(succ, tot, args.alpha)
        line = f"  {repo}: {succ}/{tot} = {format_pct(ci['rate'])}"
        if ci["ci"]:
            low, high = ci["ci"]
            line += f" (CI: {format_pct(low)}–{format_pct(high)})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
