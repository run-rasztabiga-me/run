from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

from ..core.models import EvaluationReport, GenerationResult


@dataclass(frozen=True)
class _GroupKey:
    experiment: str
    repo_url: str
    model_identifier: str
    prompt_label: str


@dataclass
class _GroupInfo:
    repo_name: str
    model_name: str
    model_provider: str
    model_label: str
    prompt_description: Optional[str]


@dataclass
class _RunRecord:
    run_id: Optional[str]
    repetition: Optional[int]
    report_path: Optional[str]
    workspace_dir: Optional[str]
    tool_steps: Optional[int]
    file_labels: List[str]
    snapshot_text: str

    @property
    def repetition_label(self) -> Optional[str]:
        if self.repetition is None:
            return None
        return f"run{self.repetition + 1}"


@dataclass
class _Snapshot:
    text: str
    file_labels: List[str] = field(default_factory=list)


class RunRepeatabilityAnalyzer:
    """Compute run-to-run diff ratios and related repeatability metrics."""

    def __init__(self, diff_threshold: float = 0.2):
        self.logger = logging.getLogger(__name__)
        self.diff_threshold = diff_threshold
        self._groups: Dict[_GroupKey, Dict[str, object]] = {}

    def record_run(
        self,
        *,
        experiment_name: str,
        repo_url: str,
        repo_name: str,
        model_identifier: str,
        model_name: str,
        model_provider: str,
        model_label: str,
        prompt_label: str,
        prompt_description: Optional[str],
        report: EvaluationReport,
        report_path: Optional[str] = None,
    ) -> None:
        """Capture generated files for a run to enable diff analysis."""
        generation_result = report.generation_result
        if not generation_result or not getattr(generation_result, "run_context", None):
            self.logger.debug("Skipping repeatability capture (no generation result/run_context)")
            return

        snapshot = self._build_snapshot(generation_result)
        if snapshot is None:
            self.logger.debug("Skipping repeatability capture (no generated files to compare)")
            return

        key = _GroupKey(
            experiment=experiment_name,
            repo_url=repo_url,
            model_identifier=model_identifier,
            prompt_label=prompt_label,
        )
        group = self._groups.setdefault(
            key,
            {
                "info": _GroupInfo(
                    repo_name=repo_name,
                    model_name=model_name,
                    model_provider=model_provider,
                    model_label=model_label,
                    prompt_description=prompt_description,
                ),
                "records": [],
            },
        )

        workspace_dir = None
        try:
            workspace_dir = str(Path(generation_result.run_context.workspace_dir).resolve())
        except Exception:  # pragma: no cover - defensive
            workspace_dir = str(generation_result.run_context.workspace_dir)

        record = _RunRecord(
            run_id=getattr(generation_result.run_context, "run_id", None),
            repetition=report.repetition_index,
            report_path=report_path,
            workspace_dir=workspace_dir,
            tool_steps=report.execution_metrics.tool_calls_count if report.execution_metrics else None,
            file_labels=snapshot.file_labels,
            snapshot_text=snapshot.text,
        )
        group["records"].append(record)

    def finalize(self, output_path: Path, relative_base: Optional[Path] = None) -> Optional[Dict[str, object]]:
        """Finalize metrics and persist them to JSON."""
        payload = self._build_payload(relative_base)
        if not payload["groups"]:
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info(
            "Repeatability analysis written to %s (%d groups)",
            output_path,
            len(payload["groups"]),
        )
        return {
            "repeatability_report": str(output_path),
            "repeatability_groups": len(payload["groups"]),
        }

    def _build_payload(self, relative_base: Optional[Path]) -> Dict[str, object]:
        groups_payload: List[Dict[str, object]] = []
        for key, data in self._groups.items():
            records: List[_RunRecord] = data["records"]
            if len(records) < 2:
                continue
            info: _GroupInfo = data["info"]
            pair_entries = self._build_pair_entries(records, relative_base)
            diff_values = [entry["diff_ratio"] for entry in pair_entries]
            tool_counts = [record.tool_steps for record in records if record.tool_steps is not None]

            group_payload = {
                "experiment": key.experiment,
                "repo_url": key.repo_url,
                "repo_name": info.repo_name,
                "model_identifier": key.model_identifier,
                "model_name": info.model_name,
                "model_provider": info.model_provider,
                "model_label": info.model_label,
                "prompt_label": key.prompt_label,
                "prompt_description": info.prompt_description,
                "runs_recorded": len(records),
                "pair_count": len(pair_entries),
                "avg_diff_ratio": self._round(mean(diff_values)),
                "max_diff_ratio": self._round(max(diff_values)),
                "std_diff_ratio": self._round(pstdev(diff_values)) if len(diff_values) > 1 else 0.0,
                "threshold": self.diff_threshold,
                "threshold_exceeded_pairs": sum(1 for value in diff_values if value > self.diff_threshold),
                "tool_steps_mean": self._round(mean(tool_counts)) if tool_counts else None,
                "tool_steps_std": self._round(pstdev(tool_counts)) if len(tool_counts) > 1 else None,
                "runs": [
                    self._serialize_run_record(record, relative_base)
                    for record in records
                ],
                "pairs": pair_entries,
            }
            groups_payload.append(group_payload)

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "diff_threshold": self.diff_threshold,
            "groups": groups_payload,
        }

    def _build_pair_entries(
        self,
        records: List[_RunRecord],
        relative_base: Optional[Path],
    ) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                first = records[i]
                second = records[j]
                diff_ratio = self._calculate_diff_ratio(first.snapshot_text, second.snapshot_text)
                entries.append(
                    {
                        "run_ids": [first.run_id, second.run_id],
                        "repetitions": [first.repetition_label, second.repetition_label],
                        "diff_ratio": self._round(diff_ratio),
                        "report_paths": [
                            self._relativize(first.report_path, relative_base),
                            self._relativize(second.report_path, relative_base),
                        ],
                    }
                )
        return entries

    def _serialize_run_record(
        self,
        record: _RunRecord,
        relative_base: Optional[Path],
    ) -> Dict[str, object]:
        return {
            "run_id": record.run_id,
            "repetition": record.repetition_label,
            "report_path": self._relativize(record.report_path, relative_base),
            "workspace_dir": record.workspace_dir,
            "tool_steps": record.tool_steps,
            "files": record.file_labels,
        }

    def _build_snapshot(self, generation_result: GenerationResult) -> Optional[_Snapshot]:
        files_with_content: List[Tuple[str, str]] = []
        workspace_dir_path = Path(generation_result.run_context.workspace_dir).resolve()
        if not workspace_dir_path.exists():
            self.logger.warning("Workspace directory %s missing, skipping snapshot", workspace_dir_path)
            return None

        for image in generation_result.docker_images or []:
            rel_path = image.dockerfile_path
            file_path = workspace_dir_path / rel_path
            content = self._safe_read(file_path)
            if content is not None:
                files_with_content.append((f"Dockerfile::{rel_path}", content))

        for manifest in generation_result.k8s_manifests or []:
            rel_path = str(manifest)
            file_path = workspace_dir_path / rel_path
            content = self._safe_read(file_path)
            if content is not None:
                files_with_content.append((f"K8s::{rel_path}", content))

        if not files_with_content:
            return None

        serialized_lines: List[str] = []
        serialized_labels: List[str] = []
        for label, content in sorted(files_with_content, key=lambda item: item[0]):
            serialized_labels.append(label)
            serialized_lines.append(f"@@FILE:{label}")
            serialized_lines.extend(content.splitlines())

        serialized_text = "\n".join(serialized_lines)
        return _Snapshot(text=serialized_text, file_labels=serialized_labels)

    @staticmethod
    def _calculate_diff_ratio(first: str, second: str) -> float:
        first_lines = first.splitlines()
        second_lines = second.splitlines()
        diff = difflib.ndiff(first_lines, second_lines)
        changed_lines = sum(1 for line in diff if line.startswith("+ ") or line.startswith("- "))
        avg_length = max((len(first_lines) + len(second_lines)) / 2.0, 1.0)
        return changed_lines / avg_length

    @staticmethod
    def _relativize(path_str: Optional[str], base_dir: Optional[Path]) -> Optional[str]:
        if not path_str:
            return None
        if not base_dir:
            return path_str
        try:
            return str(Path(path_str).resolve().relative_to(base_dir.resolve()))
        except Exception:
            return path_str

    @staticmethod
    def _round(value: float, digits: int = 4) -> float:
        return round(value, digits)

    def _safe_read(self, file_path: Path) -> Optional[str]:
        try:
            return file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.logger.warning("Generated file missing at %s", file_path)
            return None
        except UnicodeDecodeError:
            self.logger.warning("Failed to decode %s as UTF-8", file_path)
            return None


__all__ = ["RunRepeatabilityAnalyzer"]
