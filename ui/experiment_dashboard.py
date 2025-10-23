"""
Streamlit dashboard to explore experiment outputs.

Run with:
    streamlit run ui/experiment_dashboard.py
"""

from __future__ import annotations

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
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

st.set_page_config(page_title="Experiment Dashboard", layout="wide")


BASE_EXPERIMENT_DIR = Path("evaluation_reports/experiments").resolve()
EXPERIMENTS_CONFIG_DIR = Path("experiments").resolve()


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
        agg_dict["success_rate"] = ("generation_success", "mean")
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

    if "success_rate" in grouped.columns:
        grouped["success_rate"] = (grouped["success_rate"] * 100).round(1)
    if "avg_overall" in grouped.columns:
        grouped["avg_overall"] = grouped["avg_overall"].round(2)
    if "avg_generation_time" in grouped.columns:
        grouped["avg_generation_time"] = grouped["avg_generation_time"].round(2)
    if "build_success_rate" in grouped.columns:
        grouped["build_success_rate"] = (grouped["build_success_rate"] * 100).round(1)
    if "runtime_success_rate" in grouped.columns:
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
                status["stopped_at"] = datetime.utcnow().isoformat()
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
                            st.code("\n".join(display_lines), language="log")

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
    st.markdown(f"Reports directory: `{current_run.timestamp_dir}`")

    status_payload = load_status(current_run)
    if status_payload:
        state = status_payload.get("state")
        if state == "running":
            st.info("Experiment is still running; results will update as new runs complete.")
        elif state == "failed":
            st.error(f"Experiment run failed: {status_payload.get('error', 'unknown error')}")

    with st.spinner("Loading summary..."):
        summary_df = load_summary(current_run)
    filtered_df = summary_df
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
        st.json(row.to_dict())
        report_rel_path = row.get("report_path")
        if isinstance(report_rel_path, str):
            report_full_path = current_run.timestamp_dir / report_rel_path
            st.caption(f"Report path: `{report_full_path}`")
            if report_full_path.exists():
                try:
                    report_bytes = report_full_path.read_bytes()
                    st.download_button(
                        "Download report JSON",
                        data=report_bytes,
                        file_name=report_full_path.name,
                        mime="application/json",
                        key=f"download_report_{selected_row}",
                    )
                except Exception as ex:
                    st.warning(f"Unable to load report for download: {ex}")
            else:
                st.warning("Report file not found on disk. It may have been moved or deleted.")

    summary_json = load_summary_json(current_run)
    if summary_json:
        with st.expander("Raw summary.json"):
            st.json(summary_json)

    st.sidebar.info(
        "Click on a run to view the full report JSON under the referenced `report_path`."
    )


if __name__ == "__main__":
    main()
