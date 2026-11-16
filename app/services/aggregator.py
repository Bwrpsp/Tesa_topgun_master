import csv
from pathlib import Path
from typing import List, Dict, Any
from app.models.job import Job, JobStatus


class LeaderboardAggregator:
    """
    Aggregates completed jobs into a structured leaderboard and exports final_leaderboard.csv.
    """

    def generate_leaderboard(self, jobs: List[Job], output_csv: Path) -> List[Dict[str, Any]]:
        rows = []
        for job in jobs:
            eval_score = job.evaluation
            hw = job.hardware_report

            row = {
                "team_name": job.team_name,
                "job_id": job.job_id,
                "day": job.day,
                "status": job.status.value,
                "accuracy": eval_score.accuracy if eval_score else 0.0,
                "macro_f1": eval_score.macro_f1 if eval_score else 0.0,
                "mean_processing_s": eval_score.mean_processing_seconds if eval_score else 0.0,
                "mean_total_s": eval_score.mean_total_seconds if eval_score else 0.0,
                "model_load_s": eval_score.model_load_seconds if eval_score else 0.0,
                "total_wall_time_s": eval_score.total_wall_time_s if eval_score else 0.0,
                "composite_score": eval_score.composite_score if eval_score else 0.0,
                "slave_id": job.assigned_slave or "N/A",
                "start_temp": hw.start_temp if hw else None,
                "end_temp": hw.end_temp if hw else None,
                "throttled_flag": hw.throttled_flag if hw else "N/A",
                "error_message": job.error_message or ""
            }
            rows.append(row)

        # Sort: COMPLETED first, then by composite_score DESC, macro_f1 DESC, mean_processing_s ASC
        def sort_key(item):
            is_completed = 1 if item["status"] == JobStatus.COMPLETED.value else 0
            return (is_completed, item["composite_score"], item["macro_f1"], -item["mean_processing_s"])

        rows.sort(key=sort_key, reverse=True)

        # Assign ranks
        rank = 1
        for row in rows:
            if row["status"] == JobStatus.COMPLETED.value:
                row["rank"] = rank
                rank += 1
            else:
                row["rank"] = "-"

        # Ensure directory exists and write CSV
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            fieldnames = [
                "rank", "team_name", "status", "composite_score", "macro_f1", "accuracy",
                "mean_processing_s", "mean_total_s", "model_load_s", "total_wall_time_s",
                "slave_id", "start_temp", "end_temp", "throttled_flag", "job_id", "day", "error_message"
            ]
            with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})

        return rows


aggregator = LeaderboardAggregator()
