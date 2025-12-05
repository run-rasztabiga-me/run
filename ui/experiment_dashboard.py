"""
Streamlit dashboard to explore experiment outputs.

Run with:
    streamlit run ui/experiment_dashboard.py
"""

from __future__ import annotations

import html
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

st.set_page_config(page_title="Experiment Dashboard", layout="wide")


BASE_EXPERIMENT_DIR = Path("evaluation_reports/experiments").resolve()
EXPERIMENTS_CONFIG_DIR = Path("experiments").resolve()
LOG_WINDOW_STYLE = """
<style>
.log-window {
  max-height: 360px;
  overflow-y: auto;
  background-color: var(--secondary-background-color, rgba(0, 0, 0, 0.04));
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 6px;
  padding: 0.75rem;
  font-family: Menlo, Consolas, 'Liberation Mono', monospace;
  font-size: 0.6rem;
  line-height: 1.4;
}
.log-window pre {
  margin: 0;
  white-space: pre-wrap;
}
</style>
"""


@dataclass
class ExperimentRun:
    experiment_name: str
    timestamp_dir: Path
    summary_csv: Path
    summary_json: Path
    status_json: Optional[Path]


def discover_experiments(base_dir: Path) -> Dict[str, List[ExperimentRun]]:
    experiments: Dict[str, List[ExperimentRun]] = {}
    if not base_dir.exists():
        return experiments

    for experiment_root in sorted(base_dir.iterdir()):
        if not experiment_root.is_dir():
            continue

        runs: List[ExperimentRun] = []
        for timestamp_dir in sorted(experiment_root.iterdir(), reverse=True):
            if not timestamp_dir.is_dir():
                continue
            summary_csv = timestamp_dir / "summary.csv"
            summary_json = timestamp_dir / "summary.json"
            status_json = timestamp_dir / "status.json"
            if summary_csv.exists() or status_json.exists():
                runs.append(
                    ExperimentRun(
                        experiment_name=experiment_root.name,
                        timestamp_dir=timestamp_dir,
                        summary_csv=summary_csv,
                        summary_json=summary_json if summary_json.exists() else None,
                        status_json=status_json if status_json.exists() else None,
                    )
                )

        if runs:
            experiments[experiment_root.name] = runs

    return experiments


def load_summary(run: ExperimentRun) -> pd.DataFrame:
    if not run.summary_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(run.summary_csv)
    for col in ["generation_success", "build_success", "runtime_success"]:
        if col in df.columns:
            df[col] = df[col].replace({
                "True": True,
                "False": False,
                "true": True,
                "false": False,
                "None": None,
                "": None,
            })
    # Ensure consistent column order
    desired_cols = [
        "experiment",
        "timestamp",
        "repo_name",
        "model_name",
        "prompt_id",
        "status",
        "generation_success",
        "build_success",
        "runtime_success",
        "generation_time",
        "overall_score",
        "dockerfile_score",
        "k8s_score",
        "runtime_score",
        "tool_calls",
        "tokens_used",
        "repetition",
        "report_path",
    ]
    existing = [col for col in desired_cols if col in df.columns]
    remaining = [col for col in df.columns if col not in existing]
    return df[existing + remaining]


def load_summary_json(run: ExperimentRun) -> Optional[Dict]:
    if not run.summary_json or not run.summary_json.exists():
        return None
    return json.loads(run.summary_json.read_text(encoding="utf-8"))


def load_status(run: ExperimentRun) -> Optional[Dict]:
    if not run.status_json or not run.status_json.exists():
        return None
    return json.loads(run.status_json.read_text(encoding="utf-8"))


def resolve_workspace_dir(
    repo_name: str,
    run_id: Optional[str],
    report_start_time: Optional[str],
) -> Optional[Path]:
    """Best-effort lookup of the run workspace under tmp/, favoring run_id if present."""
    tmp_root = Path("tmp").resolve()
    if not tmp_root.exists():
        return None

    if repo_name and run_id:
        direct = (tmp_root / f"{repo_name}-{run_id}").resolve()
        if direct.exists():
            return direct

    candidates = [
        path for path in tmp_root.iterdir()
        if path.is_dir() and path.name.startswith(f"{repo_name}-")
    ]
    if not candidates:
        return None

    report_dt: Optional[datetime] = None
    if report_start_time:
        try:
            report_dt = datetime.fromisoformat(report_start_time)
        except ValueError:
            report_dt = None

    if report_dt:
        def score(path: Path) -> float:
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
            except (FileNotFoundError, OSError):
                return float("inf")
            return abs((mtime - report_dt).total_seconds())

        candidates.sort(key=score)
        return candidates[0]

    # Fallback: most recently modified
    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except (FileNotFoundError, OSError):
            return 0.0

    candidates.sort(key=modified, reverse=True)
    return candidates[0]


def resolve_manifest_path(manifest_entry: str, report_dir: Optional[Path], workspace_dir: Optional[Path]) -> Optional[Path]:
    """Resolve manifest entry against known directories."""
    manifest_path = Path(manifest_entry)
    candidates = []
    if manifest_path.is_absolute():
        candidates.append(manifest_path)
    if report_dir is not None:
        candidates.append((report_dir / manifest_path).resolve())
    if workspace_dir is not None:
        candidates.append((workspace_dir / manifest_path).resolve())

    checked = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in checked:
            continue
        checked.add(candidate_str)
        if candidate.exists():
            return candidate
    return None


def render_score_summary(
    df: pd.DataFrame,
    group_col: str,
    label: str,
) -> bool:
    """Render aggregated score summary grouped by the given column.

    Returns True if a summary was rendered.
    """
    if df.empty or "overall_score" not in df.columns:
        return False
    if group_col not in df.columns:
        return False

    distinct_values = df[group_col].dropna().unique()
    if len(distinct_values) <= 1:
        return False

    agg_dict: Dict[str, tuple] = {"runs": (group_col, "count")}

    if "generation_success" in df.columns:
        agg_dict["generation_success_rate"] = ("generation_success", "mean")
    if "overall_score" in df.columns:
        agg_dict["avg_overall"] = ("overall_score", "mean")
    if "generation_time" in df.columns:
        agg_dict["avg_generation_time"] = ("generation_time", "mean")
    if "build_success" in df.columns:
        agg_dict["build_success_rate"] = ("build_success", "mean")
    if "runtime_success" in df.columns:
        agg_dict["runtime_success_rate"] = ("runtime_success", "mean")

    if len(agg_dict) <= 1:
        return False

    grouped = df.groupby(group_col).agg(**agg_dict).reset_index()

    if "generation_success_rate" in grouped.columns:
        grouped["generation_success_rate"] = pd.to_numeric(grouped["generation_success_rate"], errors="coerce")
        grouped["generation_success_rate"] = (grouped["generation_success_rate"] * 100).round(1)
    if "avg_overall" in grouped.columns:
        grouped["avg_overall"] = pd.to_numeric(grouped["avg_overall"], errors="coerce").round(2)
    if "avg_generation_time" in grouped.columns:
        grouped["avg_generation_time"] = pd.to_numeric(grouped["avg_generation_time"], errors="coerce").round(2)
    if "build_success_rate" in grouped.columns:
        grouped["build_success_rate"] = pd.to_numeric(grouped["build_success_rate"], errors="coerce")
        grouped["build_success_rate"] = (grouped["build_success_rate"] * 100).round(1)
    if "runtime_success_rate" in grouped.columns:
        grouped["runtime_success_rate"] = pd.to_numeric(grouped["runtime_success_rate"], errors="coerce")
        grouped["runtime_success_rate"] = (grouped["runtime_success_rate"] * 100).round(1)

    st.subheader(f"Score Summary by {label}")
    st.dataframe(grouped, width="stretch")
    return True


def discover_experiment_configs() -> List[Path]:
    """Discover available experiment config files."""
    if not EXPERIMENTS_CONFIG_DIR.exists():
        return []
    return sorted(EXPERIMENTS_CONFIG_DIR.glob("*.yaml"))


def run_experiment_async(config_path: Path) -> subprocess.Popen:
    """Run experiment in a separate process, logging output to file."""
    try:
        # Load environment variables from .env file
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)

        # Get current environment and merge with loaded vars
        env = os.environ.copy()

        # Parse experiment name from config
        config_content = yaml.safe_load(config_path.read_text())
        exp_name = config_content.get("experiments", [{}])[0].get("name", "unknown")
        safe_name = re.sub(r'[^\w\-]', '_', exp_name)

        # Run the experiment using subprocess, capturing output to PIPE
        # We'll monitor for the directory creation and redirect there
        process = subprocess.Popen(
            ["python", "run_experiments.py", "--config", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            env=env,  # Pass environment variables
        )

        # Start a background thread to monitor experiment directory and redirect logs
        def monitor_and_log():
            # Wait for experiment directory to be created
            max_wait = 10  # seconds
            experiment_dir = None
            for _ in range(max_wait * 2):  # Check every 0.5 seconds
                time.sleep(0.5)
                exp_root = BASE_EXPERIMENT_DIR / safe_name
                if exp_root.exists():
                    # Find the most recent timestamped directory
                    dirs = sorted(exp_root.iterdir(), reverse=True)
                    if dirs and dirs[0].is_dir():
                        experiment_dir = dirs[0]
                        break

            if experiment_dir:
                log_file = experiment_dir / "experiment.log"
                pid_file = experiment_dir / "experiment.pid"

                # Write PID
                pid_file.write_text(str(process.pid), encoding="utf-8")

                # Stream output to log file
                with log_file.open("w", encoding="utf-8") as log_handle:
                    for line in process.stdout:
                        log_handle.write(line)
                        log_handle.flush()
            else:
                # Fallback: just consume the output
                for line in process.stdout:
                    pass

        log_thread = threading.Thread(target=monitor_and_log, daemon=True)
        log_thread.start()

        return process
    except Exception as e:
        st.error(f"Failed to start experiment: {e}")
        return None


def get_running_experiments() -> Dict[str, Dict[str, any]]:
    """Check for running experiments by looking at status.json files."""
    running = {}
    if not BASE_EXPERIMENT_DIR.exists():
        return running

    for experiment_root in BASE_EXPERIMENT_DIR.iterdir():
        if not experiment_root.is_dir():
            continue
        for timestamp_dir in experiment_root.iterdir():
            if not timestamp_dir.is_dir():
                continue
            status_json = timestamp_dir / "status.json"
            if status_json.exists():
                try:
                    status = json.loads(status_json.read_text(encoding="utf-8"))
                    if status.get("state") == "running":
                        log_file = timestamp_dir / "experiment.log"
                        pid_file = timestamp_dir / "experiment.pid"
                        running[f"{experiment_root.name}/{timestamp_dir.name}"] = {
                            "experiment": status.get("experiment"),
                            "started_at": status.get("started_at"),
                            "log_file": log_file if log_file.exists() else None,
                            "pid_file": pid_file if pid_file.exists() else None,
                            "status_file": status_json,
                            "dir": timestamp_dir,
                            "status": status,
                        }
                except Exception:
                    continue
    return running


def stop_experiment(exp_info: Dict) -> bool:
    """Stop a running experiment by killing its process."""
    try:
        pid_file = exp_info.get("pid_file")
        if pid_file and pid_file.exists():
            pid = int(pid_file.read_text().strip())
            # Kill the process
            os.kill(pid, signal.SIGTERM)

            # Update status to failed/stopped
            status_file = exp_info.get("status_file")
            if status_file and status_file.exists():
                status = json.loads(status_file.read_text(encoding="utf-8"))
                status["state"] = "stopped"
                status["stopped_at"] = datetime.now(timezone.utc).isoformat()
                status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")

            # Remove PID file
            pid_file.unlink()
            return True
    except ProcessLookupError:
        # Process already dead
        return True
    except Exception as e:
        st.error(f"Failed to stop experiment: {e}")
        return False
    return False


def main() -> None:
    st.markdown(LOG_WINDOW_STYLE, unsafe_allow_html=True)
    st.title("Experiment Dashboard")
    st.caption("Browse experiment runs generated by the evaluator/runner.")
    local_tz = datetime.now().astimezone().tzinfo or timezone.utc

    # Check for running experiments and show monitor
    running_experiments = get_running_experiments()
    if running_experiments:
        with st.expander(f"🔄 {len(running_experiments)} Running Experiment(s) - Click to View Logs", expanded=True):
            auto_refresh_running = st.checkbox(
                "Auto-refresh running experiments every 5s",
                key="auto_refresh_running_toggle",
                help="Automatically reload this section while experiments are in progress.",
            )
            for exp_key, exp_info in running_experiments.items():
                st.subheader(f"📊 {exp_info['experiment']}")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Status", "Running")
                    if exp_info.get("started_at"):
                        st.caption(f"Started: {exp_info['started_at'][:19]}")
                    status_payload = exp_info.get("status") or {}
                    runs_total = status_payload.get("runs_total")
                    runs_completed = status_payload.get("runs_completed")
                    if isinstance(runs_total, int) and runs_total > 0:
                        completed = runs_completed or 0
                        st.metric("Runs Completed", f"{completed}/{runs_total}")
                        progress_value = min(max(completed / runs_total, 0.0), 1.0)
                        st.progress(progress_value)
                        st.caption(f"{completed}/{runs_total} runs complete")
                    eta_seconds = status_payload.get("eta_seconds")
                    if eta_seconds is not None:
                        eta_minutes = max(float(eta_seconds), 0.0) / 60.0
                        st.metric("ETA (min)", f"{eta_minutes:.1f}")
                    avg_run_duration = status_payload.get("avg_run_duration")
                    if avg_run_duration is not None:
                        st.caption(f"Avg run duration: {avg_run_duration:.1f}s")
                    updated_at = status_payload.get("updated_at")
                    if updated_at:
                        try:
                            updated_dt = datetime.fromisoformat(updated_at)
                            if updated_dt.tzinfo:
                                updated_local = updated_dt.astimezone(local_tz)
                            else:
                                updated_local = updated_dt.replace(tzinfo=timezone.utc).astimezone(local_tz)
                            st.caption(f"Last update: {updated_local.strftime('%Y-%m-%d %H:%M:%S')}")
                        except ValueError:
                            st.caption(f"Last update: {updated_at}")
                    estimated_completion = status_payload.get("estimated_completion")
                    if estimated_completion and eta_seconds:
                        try:
                            eta_dt = datetime.fromisoformat(estimated_completion)
                            if eta_dt.tzinfo:
                                eta_local = eta_dt.astimezone(local_tz)
                            else:
                                eta_local = eta_dt.replace(tzinfo=timezone.utc).astimezone(local_tz)
                            st.caption(f"Projected finish: {eta_local.strftime('%Y-%m-%d %H:%M:%S')}")
                        except ValueError:
                            st.caption(f"Projected finish: {estimated_completion}")

                    # Action buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(f"🔄 Refresh", key=f"refresh_{exp_key}"):
                            st.rerun()
                    with btn_col2:
                        if st.button(f"⛔ Stop", key=f"stop_{exp_key}", type="secondary"):
                            if stop_experiment(exp_info):
                                st.success("Experiment stopped")
                                st.rerun()
                            else:
                                st.error("Failed to stop experiment")

                with col2:
                    if exp_info.get("log_file") and exp_info["log_file"].exists():
                        try:
                            log_content = exp_info["log_file"].read_text(encoding="utf-8")
                            # Show last 100 lines
                            lines = log_content.split("\n")
                            display_lines = lines[-100:] if len(lines) > 100 else lines
                            log_html = html.escape("\n".join(display_lines)).replace("\n", "<br>")
                            st.markdown(
                                f"<div class='log-window'><pre>{log_html}</pre></div>",
                                unsafe_allow_html=True,
                            )

                            if auto_refresh_running:
                                st.caption("Auto-refreshing logs every 5s…")
                        except Exception as e:
                            st.warning(f"Could not read log file: {e}")
                    else:
                        st.info("Log file not yet created. Refresh to check again.")
                st.divider()
            if auto_refresh_running:
                time.sleep(5)
                st.rerun()

    # Sidebar: Experiment runner section
    st.sidebar.header("Run New Experiment")
    experiment_configs = discover_experiment_configs()

    if experiment_configs:
        config_names = [cfg.name for cfg in experiment_configs]
        selected_config = st.sidebar.selectbox(
            "Select Experiment Config",
            config_names,
            help="Choose an experiment configuration to run"
        )

        if selected_config:
            config_path = next(cfg for cfg in experiment_configs if cfg.name == selected_config)

            # Display config info
            with st.sidebar.expander("Config Preview"):
                try:
                    config_content = yaml.safe_load(config_path.read_text())
                    st.json(config_content)
                except Exception:
                    st.text(config_path.read_text()[:500] + "...")

            if st.sidebar.button("🚀 Start Experiment", type="primary"):
                # Start the experiment without pre-creating directories
                # The experiment runner will create its own timestamped directory
                process = run_experiment_async(config_path)
                if process:
                    # Wait a moment for the experiment to start and create status.json
                    time.sleep(1)
                    st.rerun()
    else:
        st.sidebar.info(f"No experiment configs found in {EXPERIMENTS_CONFIG_DIR}")

    st.sidebar.divider()
    st.sidebar.header("Browse Results")

    experiments = discover_experiments(BASE_EXPERIMENT_DIR)
    if not experiments:
        st.warning(
            f"No experiment summaries found under {BASE_EXPERIMENT_DIR}. "
            "Run an experiment using the sidebar or via CLI: `python run_experiments.py --config <file>`"
        )
        return

    experiment_names = sorted(experiments.keys())
    selected_experiment = st.sidebar.selectbox(
        "Experiment", experiment_names, index=0
    )

    runs = experiments[selected_experiment]
    timestamp_labels: List[str] = []
    label_to_dir: Dict[str, str] = {}
    for run in runs:
        raw_name = run.timestamp_dir.name
        try:
            parsed_utc = datetime.strptime(raw_name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            local_dt = parsed_utc.astimezone(local_tz)
            label = local_dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            label = raw_name
        timestamp_labels.append(label)
        label_to_dir[label] = raw_name

    selected_label = st.sidebar.selectbox(
        "Run Timestamp",
        timestamp_labels,
        index=0,
        help="Each timestamp corresponds to one invocation of the experiment runner.",
    )
    selected_run_dir = label_to_dir[selected_label]
    current_run = next(run for run in runs if run.timestamp_dir.name == selected_run_dir)

    st.header(f"{selected_experiment} – {selected_label}")

    status_payload = load_status(current_run)
    if status_payload:
        state = status_payload.get("state")
        if state == "running":
            st.info("Experiment is still running; results will update as new runs complete.")
        elif state == "failed":
            st.error(f"Experiment run failed: {status_payload.get('error', 'unknown error')}")
        elif state == "stopped":
            runs_completed = status_payload.get("runs_completed", 0)
            runs_total = status_payload.get("runs_total", 0)
            st.warning(
                f"⚠️ Experiment was stopped manually. Showing partial results: "
                f"{runs_completed}/{runs_total} runs completed."
            )

    with st.spinner("Loading summary..."):
        summary_df = load_summary(current_run)
    filtered_df = summary_df
    # Overall summary across all runs
    if not summary_df.empty:
        st.subheader("Overall Summary")

        total_runs = len(summary_df)

        # Calculate metrics
        metrics = {}
        metrics["Total Runs"] = total_runs

        if "overall_score" in summary_df.columns:
            avg_overall = pd.to_numeric(summary_df["overall_score"], errors="coerce").mean()
            if not pd.isna(avg_overall):
                metrics["Avg Overall Score"] = f"{avg_overall:.2f}"

        if "generation_success" in summary_df.columns:
            gen_success_rate = pd.to_numeric(summary_df["generation_success"], errors="coerce").mean()
            if not pd.isna(gen_success_rate):
                metrics["Generation Success Rate"] = f"{gen_success_rate * 100:.1f}%"

        if "build_success" in summary_df.columns:
            build_success_rate = pd.to_numeric(summary_df["build_success"], errors="coerce").mean()
            if not pd.isna(build_success_rate):
                metrics["Build Success Rate"] = f"{build_success_rate * 100:.1f}%"

        if "runtime_success" in summary_df.columns:
            runtime_success_rate = pd.to_numeric(summary_df["runtime_success"], errors="coerce").mean()
            if not pd.isna(runtime_success_rate):
                metrics["Runtime Success Rate"] = f"{runtime_success_rate * 100:.1f}%"

        if "generation_time" in summary_df.columns:
            avg_gen_time = pd.to_numeric(summary_df["generation_time"], errors="coerce").mean()
            if not pd.isna(avg_gen_time):
                metrics["Avg Generation Time"] = f"{avg_gen_time:.2f}s"

        if "dockerfile_score" in summary_df.columns:
            avg_dockerfile = pd.to_numeric(summary_df["dockerfile_score"], errors="coerce").mean()
            if not pd.isna(avg_dockerfile):
                metrics["Avg Dockerfile Score"] = f"{avg_dockerfile:.2f}"

        if "k8s_score" in summary_df.columns:
            avg_k8s = pd.to_numeric(summary_df["k8s_score"], errors="coerce").mean()
            if not pd.isna(avg_k8s):
                metrics["Avg K8s Score"] = f"{avg_k8s:.2f}"

        if "runtime_score" in summary_df.columns:
            avg_runtime = pd.to_numeric(summary_df["runtime_score"], errors="coerce").mean()
            if not pd.isna(avg_runtime):
                metrics["Avg Runtime Score"] = f"{avg_runtime:.2f}"

        if "tool_calls" in summary_df.columns:
            avg_tool_calls = pd.to_numeric(summary_df["tool_calls"], errors="coerce").mean()
            if not pd.isna(avg_tool_calls):
                metrics["Avg Tool Calls"] = f"{avg_tool_calls:.1f}"

        if "tokens_used" in summary_df.columns:
            avg_tokens = pd.to_numeric(summary_df["tokens_used"], errors="coerce").mean()
            if not pd.isna(avg_tokens):
                metrics["Avg Tokens Used"] = f"{avg_tokens:.0f}"

        # Display metrics in a table
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.divider()

    st.subheader("Per-run Summary")
    if summary_df.empty:
        st.write("No completed runs recorded yet.")
    else:
        filter_definitions = [
            ("Model", "model_name"),
            ("Repository", "repo_name"),
            ("Prompt", "prompt_id"),
            ("Status", "status"),
        ]
        applied_filters: Dict[str, List[str]] = {}
        with st.expander("Filter runs", expanded=False):
            for label, column in filter_definitions:
                if column not in summary_df.columns:
                    continue
                options = sorted(
                    {str(value) for value in summary_df[column].dropna().unique()}
                )
                if not options:
                    continue
                selection = st.multiselect(
                    f"{label}",
                    options,
                    default=options,
                    key=f"filter_{column}_{current_run.timestamp_dir.name}",
                )
                if selection and set(selection) != set(options):
                    applied_filters[column] = selection
        filtered_df = summary_df.copy()
        for column, values in applied_filters.items():
            filtered_df = filtered_df[filtered_df[column].astype(str).isin(values)]
        st.caption(f"Showing {len(filtered_df)} of {len(summary_df)} runs")
        st.dataframe(filtered_df, width="stretch")

    if not filtered_df.empty:
        for column, label in [
            ("prompt_id", "Prompt"),
            ("model_name", "Model"),
            ("repo_name", "Repository"),
            ("temperature", "Temperature"),
        ]:
            render_score_summary(filtered_df, column, label)

    st.subheader("Run Details")
    if not filtered_df.empty:
        selected_row = st.selectbox(
            "Select a run to inspect",
            filtered_df.index,
            format_func=lambda idx: f"{filtered_df.loc[idx, 'repo_name']} | {filtered_df.loc[idx, 'model_name']} | prompt={filtered_df.loc[idx, 'prompt_id']} | repetition={filtered_df.loc[idx, 'repetition']}",
        )
        row = filtered_df.loc[selected_row].copy()
        for col in ["generation_success", "build_success", "runtime_success"]:
            if col in row and pd.notna(row[col]):
                row[col] = bool(row[col])
        report_rel_path = row.get("report_path")
        report_data: Optional[Dict] = None
        report_bytes: Optional[bytes] = None
        report_full_path: Optional[Path] = None
        if isinstance(report_rel_path, str):
            report_full_path = (current_run.timestamp_dir / report_rel_path).resolve()
            if report_full_path.exists():
                try:
                    report_text = report_full_path.read_text(encoding="utf-8")
                    report_bytes = report_text.encode("utf-8")
                    report_data = json.loads(report_text)
                except Exception as ex:
                    st.warning(f"Unable to parse report JSON: {ex}")
                    try:
                        report_bytes = report_full_path.read_bytes()
                    except Exception:
                        report_bytes = None
            else:
                st.warning("Report file not found on disk. It may have been moved or deleted.")
        if report_bytes and report_full_path:
            st.download_button(
                "Download report JSON",
                data=report_bytes,
                file_name=report_full_path.name,
                mime="application/json",
                key=f"download_report_{selected_row}",
            )
        if report_data:
            summary_tab, metrics_tab, validation_tab, artifacts_tab, notes_tab = st.tabs(
                ["Summary", "Metrics", "Validation", "Artifacts", "Notes"]
            )
            with summary_tab:
                summary_fields = {
                    "Status": report_data.get("status"),
                    "Build success": report_data.get("build_success"),
                    "Runtime success": report_data.get("runtime_success"),
                    "Start time": report_data.get("start_time"),
                    "End time": report_data.get("end_time"),
                    "Total evaluation time": report_data.get("total_evaluation_time"),
                    "Prompt ID": report_data.get("prompt_id"),
                    "Model": report_data.get("model_name"),
                    "Experiment": report_data.get("experiment_name"),
                }
                extra_metadata = report_data.get("extra_metadata") or {}
                col_summary, col_extra = st.columns(2)
                with col_summary:
                    st.json({k: v for k, v in summary_fields.items() if v is not None})
                with col_extra:
                    if extra_metadata:
                        st.json(extra_metadata)
                    else:
                        st.write("No extra metadata.")
            with metrics_tab:
                exec_metrics = report_data.get("execution_metrics") or {}
                quality_metrics = report_data.get("quality_metrics") or {}
                col_exec, col_quality = st.columns(2)
                with col_exec:
                    st.markdown("**Execution Metrics**")
                    if exec_metrics:
                        st.json(exec_metrics)
                    else:
                        st.write("No execution metrics recorded.")
                with col_quality:
                    st.markdown("**Quality Metrics**")
                    if quality_metrics:
                        st.json({k: v for k, v in quality_metrics.items() if k != "validation_issues"})
                    else:
                        st.write("No quality metrics recorded.")
            with validation_tab:
                quality_metrics = report_data.get("quality_metrics") or {}
                validation_issues = quality_metrics.get("validation_issues") or []
                st.json(validation_issues)
            with artifacts_tab:
                generation_result = report_data.get("generation_result") or {}
                docker_images = generation_result.get("docker_images") or []
                k8s_manifests = generation_result.get("k8s_manifests") or []
                workspace_dir_raw = (
                    generation_result.get("workspace_dir")
                    or (report_data.get("extra_metadata") or {}).get("workspace_dir")
                )
                run_id_value = (
                    generation_result.get("run_id")
                    or (report_data.get("extra_metadata") or {}).get("run_id")
                )
                repo_name = str(row.get("repo_name", ""))
                declared_workspace_missing: Optional[Path] = None
                workspace_dir: Optional[Path] = None
                if workspace_dir_raw:
                    candidate = Path(workspace_dir_raw)
                    try:
                        candidate = candidate.resolve()
                    except Exception:
                        pass
                    if candidate.exists():
                        workspace_dir = candidate
                    else:
                        declared_workspace_missing = candidate
                expected_workspace: Optional[Path] = None
                if repo_name and run_id_value:
                    expected_workspace = (Path("tmp") / f"{repo_name}-{run_id_value}").resolve()
                if workspace_dir is None:
                    workspace_dir = resolve_workspace_dir(
                        repo_name=repo_name,
                        run_id=run_id_value,
                        report_start_time=report_data.get("start_time"),
                    )
                if workspace_dir:
                    if expected_workspace and workspace_dir == expected_workspace:
                        pass
                    elif workspace_dir_raw:
                        pass
                else:
                    if declared_workspace_missing:
                        st.warning(
                            f"Workspace directory `{declared_workspace_missing}` not found; "
                            "generated artifacts may have been cleaned up."
                        )
                    elif expected_workspace:
                        st.warning(
                            f"Workspace directory `{expected_workspace}` not found; "
                            "generated artifacts may have been cleaned up."
                        )
                col_docker, col_k8s = st.columns(2)
                docker_options: List[Tuple[str, Dict[str, str]]] = []
                if docker_images:
                    for image_spec in docker_images:
                        docker_path = image_spec.get("dockerfile_path")
                        if docker_path:
                            docker_options.append(
                                (
                                    str(docker_path),
                                    {
                                        "path": docker_path,
                                        "context": image_spec.get("build_context"),
                                        "image_tag": image_spec.get("image_tag"),
                                    },
                                )
                            )
                with col_docker:
                    st.markdown("**Dockerfiles**")
                    if docker_options:
                        docker_labels = [opt[0] for opt in docker_options]
                        selected_docker = st.selectbox(
                            "Select Dockerfile",
                            docker_labels,
                            key=f"docker_select_{selected_row}",
                        )
                        docker_entry = next(
                            (entry for label, entry in docker_options if label == selected_docker),
                            None,
                        )
                        docker_path = docker_entry.get("path") if docker_entry else None
                        manifest_parent = report_full_path.parent if report_full_path else None
                        resolved_docker = resolve_manifest_path(
                            docker_path or "",
                            report_dir=manifest_parent,
                            workspace_dir=workspace_dir,
                        )
                        if resolved_docker and resolved_docker.exists():
                            caption_bits = [docker_path]
                            if docker_entry.get("image_tag"):
                                caption_bits.append(f"tag: {docker_entry['image_tag']}")
                            if docker_entry.get("context"):
                                caption_bits.append(f"context: {docker_entry['context']}")
                            st.caption(" | ".join(filter(None, caption_bits)))
                            try:
                                docker_text = resolved_docker.read_text(encoding="utf-8")
                                st.code(docker_text, language="dockerfile")
                            except Exception as ex:
                                st.warning(f"Unable to read Dockerfile: {ex}")
                        else:
                            st.warning("Dockerfile not found; check tmp/ for cleaned workspaces.")
                    else:
                        st.write("No Dockerfiles generated.")

                k8s_options: List[str] = list(k8s_manifests or [])
                with col_k8s:
                    st.markdown("**Kubernetes Manifests**")
                    if k8s_options:
                        selected_manifest = st.selectbox(
                            "Select K8s manifest",
                            k8s_options,
                            key=f"k8s_select_{selected_row}",
                        )
                        manifest_parent = report_full_path.parent if report_full_path else None
                        manifest_path = resolve_manifest_path(
                            selected_manifest or "",
                            report_dir=manifest_parent,
                            workspace_dir=workspace_dir,
                        )
                        if manifest_path and manifest_path.exists():
                            st.caption(selected_manifest)
                            try:
                                manifest_text = manifest_path.read_text(encoding="utf-8")
                                st.code(manifest_text, language="yaml")
                            except Exception as ex:
                                st.warning(f"Unable to read manifest: {ex}")
                        else:
                            st.warning(
                                "Manifest file not found. Ensure the workspace directory "
                                "is still available under `tmp/`."
                            )
                    else:
                        st.write("No Kubernetes manifests generated.")
            with notes_tab:
                notes = report_data.get("notes") or []
                st.json(notes)

    summary_json = load_summary_json(current_run)
    if summary_json:
        with st.expander("Raw summary.json"):
            st.json(summary_json)

    st.sidebar.info(
        "Click on a run to view the full report JSON under the referenced `report_path`."
    )


if __name__ == "__main__":
    main()
