import json
import csv
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from ..core.models import EvaluationReport, ValidationSeverity


class EvaluationReporter:
    """Generates reports from evaluation results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def save_report(self, report: EvaluationReport, output_dir: str = "./evaluation_reports") -> str:
        """
        Save individual evaluation report to JSON file.

        Args:
            report: Evaluation report to save
            output_dir: Directory to save reports

        Returns:
            Path to saved report file
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.repo_name}_{timestamp}_{report.evaluation_id[:8]}.json"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._report_to_dict(report), f, indent=2, default=str)

            self.logger.info(f"Report saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
            raise

    def export_batch_results(self, reports: List[EvaluationReport], output_dir: str = "./evaluation_reports") -> Dict[str, str]:
        """
        Export batch evaluation results in multiple formats.

        Args:
            reports: List of evaluation reports
            output_dir: Directory to save exports

        Returns:
            Dictionary mapping format to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exported_files = {}

        try:
            # Export as JSON
            json_file = os.path.join(output_dir, f"batch_evaluation_{timestamp}.json")
            self._export_json(reports, json_file)
            exported_files['json'] = json_file

            # Export as CSV summary
            csv_file = os.path.join(output_dir, f"batch_summary_{timestamp}.csv")
            self._export_csv_summary(reports, csv_file)
            exported_files['csv'] = csv_file

            # Export detailed CSV
            detailed_csv = os.path.join(output_dir, f"batch_detailed_{timestamp}.csv")
            self._export_detailed_csv(reports, detailed_csv)
            exported_files['detailed_csv'] = detailed_csv

            # Generate HTML report
            html_file = os.path.join(output_dir, f"batch_report_{timestamp}.html")
            self._export_html_report(reports, html_file)
            exported_files['html'] = html_file

            self.logger.info(f"Batch results exported to {len(exported_files)} formats")

        except Exception as e:
            self.logger.error(f"Failed to export batch results: {str(e)}")
            raise

        return exported_files

    def _report_to_dict(self, report: EvaluationReport) -> Dict[str, Any]:
        """Convert evaluation report to dictionary for serialization."""
        return {
            "repo_url": report.repo_url,
            "repo_name": report.repo_name,
            "evaluation_id": report.evaluation_id,
            "status": report.status.value,
            "start_time": report.start_time.isoformat(),
            "end_time": report.end_time.isoformat() if report.end_time else None,
            "total_evaluation_time": report.total_evaluation_time,
            "generation_result": self._generation_result_to_dict(report.generation_result) if report.generation_result else None,
            "execution_metrics": self._execution_metrics_to_dict(report.execution_metrics) if report.execution_metrics else None,
            "quality_metrics": self._quality_metrics_to_dict(report.quality_metrics) if report.quality_metrics else None,
            "run_results": [self._report_to_dict(run_report) for run_report in report.run_results],
            "notes": report.notes
        }

    def _generation_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert generation result to dictionary."""
        return {
            "repo_url": result.repo_url,
            "repo_name": result.repo_name,
            "success": result.success,
            "generated_files": result.generated_files,
            "dockerfile_path": result.dockerfile_path,
            "k8s_manifests_path": result.k8s_manifests_path,
            "generation_time": result.generation_time,
            "error_message": result.error_message,
            "timestamp": result.timestamp.isoformat()
        }

    def _execution_metrics_to_dict(self, metrics) -> Dict[str, Any]:
        """Convert execution metrics to dictionary."""
        return {
            "total_time": metrics.total_time,
            "tool_calls_count": metrics.tool_calls_count,
            "tool_calls_breakdown": metrics.tool_calls_breakdown,
            "tokens_used": metrics.tokens_used,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "error_count": metrics.error_count,
            "retry_count": metrics.retry_count,
            "run_number": metrics.run_number,
            "success_detected_via_done": metrics.success_detected_via_done
        }

    def _quality_metrics_to_dict(self, metrics) -> Dict[str, Any]:
        """Convert quality metrics to dictionary."""
        return {
            "dockerfile_score": metrics.dockerfile_score,
            "k8s_manifests_score": metrics.k8s_manifests_score,
            "overall_score": metrics.overall_score,
            "best_practices_violations": metrics.best_practices_violations,
            "security_issues": metrics.security_issues,
            "validation_issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "rule_id": issue.rule_id,
                    "category": issue.category
                }
                for issue in metrics.validation_issues
            ]
        }


    def _export_json(self, reports: List[EvaluationReport], filepath: str) -> None:
        """Export reports as JSON."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_reports": len(reports),
            "reports": [self._report_to_dict(report) for report in reports]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _export_csv_summary(self, reports: List[EvaluationReport], filepath: str) -> None:
        """Export summary statistics as CSV."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Repo Name', 'Repo URL', 'Status', 'Generation Success',
                'Generation Time (s)', 'Tool Calls', 'Overall Score',
                'Dockerfile Score', 'K8s Score', 'Validation Issues',
                'Security Issues', 'Best Practice Violations'
            ])

            # Data rows
            for report in reports:
                writer.writerow([
                    report.repo_name,
                    report.repo_url,
                    report.status.value,
                    report.generation_result.success if report.generation_result else False,
                    report.generation_result.generation_time if report.generation_result else 0,
                    report.execution_metrics.tool_calls_count if report.execution_metrics else 0,
                    report.quality_metrics.overall_score if report.quality_metrics else 0,
                    report.quality_metrics.dockerfile_score if report.quality_metrics else 0,
                    report.quality_metrics.k8s_manifests_score if report.quality_metrics else 0,
                    len(report.quality_metrics.validation_issues) if report.quality_metrics else 0,
                    report.quality_metrics.security_issues if report.quality_metrics else 0,
                    report.quality_metrics.best_practices_violations if report.quality_metrics else 0
                ])

    def _export_detailed_csv(self, reports: List[EvaluationReport], filepath: str) -> None:
        """Export detailed validation issues as CSV."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Repo Name', 'File Path', 'Line Number', 'Severity',
                'Message', 'Rule ID', 'Category'
            ])

            # Data rows
            for report in reports:
                if report.quality_metrics and report.quality_metrics.validation_issues:
                    for issue in report.quality_metrics.validation_issues:
                        writer.writerow([
                            report.repo_name,
                            issue.file_path,
                            issue.line_number,
                            issue.severity.value,
                            issue.message,
                            issue.rule_id,
                            issue.category
                        ])

    def _export_html_report(self, reports: List[EvaluationReport], filepath: str) -> None:
        """Export HTML dashboard report."""
        # Calculate summary statistics
        total_repos = len(reports)
        successful_generations = sum(1 for r in reports if r.generation_result and r.generation_result.success)
        avg_generation_time = sum(
            r.generation_result.generation_time for r in reports
            if r.generation_result and r.generation_result.generation_time
        ) / total_repos if total_repos > 0 else 0

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Configuration Generation Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 10px; }}
        .metric h3 {{ margin: 0; color: #007bff; }}
        .metric p {{ margin: 5px 0; font-size: 24px; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
        .status-completed {{ color: #28a745; }}
        .status-failed {{ color: #dc3545; }}
        .severity-error {{ color: #dc3545; }}
        .severity-warning {{ color: #ffc107; }}
        .severity-info {{ color: #17a2b8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Configuration Generation Evaluation Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="summary">
        <div class="metric">
            <h3>Total Repositories</h3>
            <p>{total_repos}</p>
        </div>
        <div class="metric">
            <h3>Successful Generations</h3>
            <p>{successful_generations}</p>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <p>{(successful_generations/total_repos*100):.1f}%</p>
        </div>
        <div class="metric">
            <h3>Avg Generation Time</h3>
            <p>{avg_generation_time:.1f}s</p>
        </div>
    </div>

    <h2>Repository Results</h2>
    <table>
        <thead>
            <tr>
                <th>Repository</th>
                <th>Status</th>
                <th>Generation Time</th>
                <th>Overall Score</th>
                <th>Tool Calls</th>
                <th>Issues</th>
            </tr>
        </thead>
        <tbody>
"""

        for report in reports:
            status_class = f"status-{report.status.value.replace('_', '-')}"
            generation_time = report.generation_result.generation_time if report.generation_result else "N/A"
            overall_score = report.quality_metrics.overall_score if report.quality_metrics else "N/A"
            tool_calls = report.execution_metrics.tool_calls_count if report.execution_metrics else "N/A"
            issues_count = len(report.quality_metrics.validation_issues) if report.quality_metrics else 0

            html_content += f"""
            <tr>
                <td><a href="{report.repo_url}" target="_blank">{report.repo_name}</a></td>
                <td class="{status_class}">{report.status.value}</td>
                <td>{generation_time}</td>
                <td>{overall_score}</td>
                <td>{tool_calls}</td>
                <td>{issues_count}</td>
            </tr>
"""

        html_content += """
        </tbody>
    </table>
</body>
</html>
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def generate_summary_stats(self, reports: List[EvaluationReport]) -> Dict[str, Any]:
        """Generate summary statistics from reports."""
        if not reports:
            return {}

        total_repos = len(reports)
        completed_reports = [r for r in reports if r.status.value == "completed"]

        return {
            "total_repositories": total_repos,
            "completed_evaluations": len(completed_reports),
            "success_rate": len(completed_reports) / total_repos if total_repos > 0 else 0,
            "average_generation_time": sum(
                r.generation_result.generation_time for r in completed_reports
                if r.generation_result and r.generation_result.generation_time
            ) / len(completed_reports) if completed_reports else 0,
            "average_tool_calls": sum(
                r.execution_metrics.tool_calls_count for r in completed_reports
                if r.execution_metrics
            ) / len(completed_reports) if completed_reports else 0,
            "average_overall_score": sum(
                r.quality_metrics.overall_score for r in completed_reports
                if r.quality_metrics and r.quality_metrics.overall_score
            ) / len(completed_reports) if completed_reports else 0
        }