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


def wilson_score_interval(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p_hat = successes / total

    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * ((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5) / denominator

    lower = center - margin
    upper = center + margin

    # Clamp to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return lower, upper


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
        return {"rate": None, "ci": None,}
    rate = successes / total
    lower, upper = wilson_score_interval(successes, total, alpha)
    return {
        "rate": rate,
        "ci": (lower, upper)
    }


def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"


def format_pct_latex(x: Optional[float]) -> str:
    """Format percentage for LaTeX with comma as decimal separator."""
    if x is None:
        return "n/a"
    return f"{x*100:.1f}".replace(".", "{,}") + r"\%"


def format_ci_latex(ci: Optional[Tuple[float, float]]) -> str:
    """Format confidence interval for LaTeX."""
    if ci is None:
        return "n/a"
    low, high = ci
    low_str = f"{low*100:.1f}".replace(".", "{,}")
    high_str = f"{high*100:.1f}".replace(".", "{,}")
    return f"{low_str}\\%--{high_str}\\%"


def write_latex_results(output_file: pathlib.Path, stats: Dict, alpha: float) -> None:
    """Write results in LaTeX table format."""
    with open(output_file, "w") as f:
        # Stages table
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"    \centering" + "\n")
        f.write(r"    \begin{tabular}{lccc}" + "\n")
        f.write(r"        \textbf{Etap} & \textbf{Sukcesy} & \textbf{Skuteczność} & \textbf{95\% CI} \\" + "\n")

        # Build stage
        succ = stats["stage_success"].get("build", 0)
        tot = stats["stage_counts"].get("build", 0)
        ci_data = rate_and_ci(succ, tot, alpha)
        f.write(f"        Build & {succ}/{tot} & {format_pct_latex(ci_data['rate'])} & {format_ci_latex(ci_data['ci'])} \\\\\n")

        # Apply stage
        succ = stats["stage_success"].get("apply", 0)
        tot = stats["stage_counts"].get("apply", 0)
        ci_data = rate_and_ci(succ, tot, alpha)
        f.write(f"        K8s apply & {succ}/{tot} & {format_pct_latex(ci_data['rate'])} & {format_ci_latex(ci_data['ci'])} \\\\\n")

        # Conditional apply | build
        if "apply" in stats["stage_conditional_counts"]:
            succ = stats["stage_conditional_success"].get("apply", 0)
            tot = stats["stage_conditional_counts"].get("apply", 0)
            ci_data = rate_and_ci(succ, tot, alpha)
            f.write(f"        apply $|$ build & {succ}/{tot} & {format_pct_latex(ci_data['rate'])} & {format_ci_latex(ci_data['ci'])} \\\\\n")

        # Runtime stage
        succ = stats["stage_success"].get("runtime", 0)
        tot = stats["stage_counts"].get("runtime", 0)
        ci_data = rate_and_ci(succ, tot, alpha)
        f.write(f"        Runtime & {succ}/{tot} & {format_pct_latex(ci_data['rate'])} & {format_ci_latex(ci_data['ci'])} \\\\\n")

        # Conditional runtime | apply
        if "runtime" in stats["stage_conditional_counts"]:
            succ = stats["stage_conditional_success"].get("runtime", 0)
            tot = stats["stage_conditional_counts"].get("runtime", 0)
            ci_data = rate_and_ci(succ, tot, alpha)
            f.write(f"        runtime $|$ apply & {succ}/{tot} & {format_pct_latex(ci_data['rate'])} & {format_ci_latex(ci_data['ci'])} \\\\\n")

        f.write(r"    \end{tabular}" + "\n")
        f.write(r"    \caption{Skuteczność etapów w H1}" + "\n")
        f.write(r"    \label{tab:h1-stages}" + "\n")
        f.write(r"\end{table}" + "\n\n")

        # Per model section
        f.write(r"\subsubsection{Rozbicie per model}" + "\n\n")
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"    \centering" + "\n")
        f.write(r"    \begin{tabular}{lccc}" + "\n")
        f.write(r"        \textbf{Model} & \textbf{Sukcesy} & \textbf{Skuteczność} & \textbf{95\% CI} \\" + "\n")

        for model, cnt in stats["per_model_counts"].items():
            succ = cnt["success"]
            tot = cnt["total"]
            ci_data = rate_and_ci(succ, tot, alpha)
            f.write(f"        {model} & {succ}/{tot} & {format_pct_latex(ci_data['rate'])} & {format_ci_latex(ci_data['ci'])} \\\\\n")

        f.write(r"    \end{tabular}" + "\n")
        f.write(r"    \caption{Skuteczność per model w H1}" + "\n")
        f.write(r"    \label{tab:h1-models}" + "\n")
        f.write(r"\end{table}" + "\n\n")

        # Per repo section
        f.write(r"\subsubsection{Rozbicie per repozytorium}" + "\n\n")
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"    \centering" + "\n")
        f.write(r"    \begin{tabular}{lccc}" + "\n")
        f.write(r"        \textbf{Repozytorium} & \textbf{Sukcesy} & \textbf{Skuteczność} & \textbf{95\% CI} \\" + "\n")

        for repo, cnt in stats["per_repo_counts"].items():
            succ = cnt["success"]
            tot = cnt["total"]
            ci_data = rate_and_ci(succ, tot, alpha)
            # Escape underscores for LaTeX
            repo_latex = repo.replace("_", r"\_")
            f.write(f"        {repo_latex} & {succ}/{tot} & {format_pct_latex(ci_data['rate'])} & {format_ci_latex(ci_data['ci'])} \\\\\n")

        f.write(r"    \end{tabular}" + "\n")
        f.write(r"    \caption{Skuteczność per repozytorium w H1}" + "\n")
        f.write(r"    \label{tab:h1-repos}" + "\n")
        f.write(r"\end{table}" + "\n")


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
    args = parser.parse_args()

    base = pathlib.Path(args.results_path)
    if not base.exists():
        print(f"Path not found: {base}", file=sys.stderr)
        return 1

    stats = compute_stats(base)
    overall = rate_and_ci(stats["total_success"], stats["total_runs"], args.alpha)

    output_file = base / "results.txt"
    with open(output_file, "w") as f:
        print(f"Experiment path: {base}", file=f)
        print(f"Runs: {stats['total_success']}/{stats['total_runs']} = {format_pct(overall['rate'])}", file=f)
        if overall["ci"]:
            low, high = overall["ci"]
            print(f"  {int((1-args.alpha)*100)}% CI: {format_pct(low)}–{format_pct(high)}", file=f)

        print("\nStages:", file=f)
        for stage in ("build", "apply", "runtime"):
            succ = stats["stage_success"].get(stage, 0)
            tot = stats["stage_counts"].get(stage, 0)
            stage_ci = rate_and_ci(succ, tot, args.alpha)
            print(f"  {stage}: {succ}/{tot} = {format_pct(stage_ci['rate'])}", end="", file=f)
            if stage_ci["ci"]:
                low, high = stage_ci["ci"]
                print(f" (CI: {format_pct(low)}–{format_pct(high)})", end="", file=f)

            # Show conditional rate for stages that depend on previous stage
            if stage in stats["stage_conditional_counts"]:
                cond_succ = stats["stage_conditional_success"].get(stage, 0)
                cond_tot = stats["stage_conditional_counts"].get(stage, 0)
                cond_ci = rate_and_ci(cond_succ, cond_tot, args.alpha)
                prev_stage = "build" if stage == "apply" else "apply"
                print(f"\n    └─ given {prev_stage} OK: {cond_succ}/{cond_tot} = {format_pct(cond_ci['rate'])}", end="", file=f)
                if cond_ci["ci"]:
                    low, high = cond_ci["ci"]
                    print(f" (CI: {format_pct(low)}–{format_pct(high)})", file=f)
                else:
                    print(file=f)
            else:
                print(file=f)

        print("\nPrompts:", file=f)
        for prompt, tot in stats["prompt_counts"].items():
            succ = stats["prompt_success"].get(prompt, 0)
            ci = rate_and_ci(succ, tot, args.alpha)
            line = f"  {prompt}: {succ}/{tot} = {format_pct(ci['rate'])}"
            if ci["ci"]:
                low, high = ci["ci"]
                line += f" (CI: {format_pct(low)}–{format_pct(high)})"
            print(line, file=f)

        print("\nPer model:", file=f)
        for model, cnt in stats["per_model_counts"].items():
            succ = cnt["success"]
            tot = cnt["total"]
            ci = rate_and_ci(succ, tot, args.alpha)
            line = f"  {model}: {succ}/{tot} = {format_pct(ci['rate'])}"
            if ci["ci"]:
                low, high = ci["ci"]
                line += f" (CI: {format_pct(low)}–{format_pct(high)})"
            print(line, file=f)

        print("\nPer repo:", file=f)
        for repo, cnt in stats["per_repo_counts"].items():
            succ = cnt["success"]
            tot = cnt["total"]
            ci = rate_and_ci(succ, tot, args.alpha)
            line = f"  {repo}: {succ}/{tot} = {format_pct(ci['rate'])}"
            if ci["ci"]:
                low, high = ci["ci"]
                line += f" (CI: {format_pct(low)}–{format_pct(high)})"
            print(line, file=f)

    print(f"Results saved to {output_file}")

    # Write LaTeX results
    latex_output_file = base / "results.tex"
    write_latex_results(latex_output_file, stats, args.alpha)
    print(f"LaTeX results saved to {latex_output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
