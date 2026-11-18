import csv
from pathlib import Path
from typing import List, Dict, Any
from app.models.job import Job, JobStatus


class LeaderboardAggregator:
    """
    Aggregates completed jobs into a structured leaderboard and exports final_leaderboard.csv
    with metrics appropriate for Day 1 (Classification), Day 2 (Masking IoU95), or Day 3 (mAP@95).
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
                # Day 1 metrics
                "correct_pics": f"{eval_score.correct_predictions}/{eval_score.total_images}" if (eval_score and job.day == 1) else "",
                "accuracy": eval_score.accuracy if eval_score else 0.0,
                # Day 2 metrics
                "count": f"{eval_score.count_accuracy:.2f}%" if (eval_score and job.day == 2) else "",
                "iou95": eval_score.iou95 if eval_score else 0,
                "iou90": eval_score.iou90 if eval_score else 0,
                "iou_average": eval_score.iou_average if eval_score else 0.0,
                "latency": eval_score.latency if (eval_score and job.day == 2) else (eval_score.total_wall_time_s if eval_score else 0.0),
                # Day 2 & 3 legacy/detailed metrics
                "tp": eval_score.tp if eval_score else 0,
                "fp": eval_score.fp if eval_score else 0,
                "fn": eval_score.fn if eval_score else 0,
                "precision": eval_score.precision if eval_score else 0.0,
                "recall": eval_score.recall if eval_score else 0.0,
                "f1_score": eval_score.f1_score if (eval_score and job.day in (2, 3)) else (eval_score.macro_f1 if eval_score else 0.0),
                "mean_iou": eval_score.mean_iou if (eval_score and job.day in (2, 3)) else 0.0,
                "map_95": eval_score.map_95 if (eval_score and job.day == 3) else 0.0,
                "class_accuracy": eval_score.class_accuracy if (eval_score and job.day == 3) else 0.0,
                # Latency / timing metrics
                "mean_processing_s": eval_score.mean_processing_seconds if eval_score else 0.0,
                "mean_total_s": eval_score.mean_total_seconds if eval_score else 0.0,
                "model_load_s": eval_score.model_load_seconds if eval_score else 0.0,
                "total_wall_time_s": eval_score.total_wall_time_s if eval_score else 0.0,
                "composite_score": eval_score.composite_score if eval_score else 0.0,
                # Hardware report
                "slave_id": job.assigned_slave or "N/A",
                "start_temp": hw.start_temp if hw else None,
                "end_temp": hw.end_temp if hw else None,
                "throttled_flag": hw.throttled_flag if hw else "N/A",
                "error_message": job.error_message or ""
            }
            rows.append(row)

        sample_day = jobs[0].day if jobs else 1

        # Sort: COMPLETED first, then metrics based on competition day
        def sort_key(item):
            is_completed = 1 if item["status"] == JobStatus.COMPLETED.value else 0
            if sample_day == 2:
                c_score = float(item.get("composite_score") or 0.0)
                iou95 = float(item.get("iou95") or 0.0)
                iou90 = float(item.get("iou90") or 0.0)
                lat = float(item.get("latency") or 999999.0)
                return (is_completed, c_score, iou95, iou90, -lat)
            elif sample_day == 1:
                c_score = float(item.get("composite_score") or 0.0)
                acc = float(item.get("accuracy") or 0.0)
                wall = float(item.get("total_wall_time_s") or 999999.0)
                return (is_completed, c_score, acc, -wall)
            else:
                c_score = float(item.get("composite_score") or 0.0)
                f1 = float(item.get("f1_score") or 0.0)
                proc = float(item.get("mean_processing_s") or 999999.0)
                return (is_completed, c_score, f1, -proc)

        rows.sort(key=sort_key, reverse=True)

        # Assign ranks
        rank = 1
        for row in rows:
            if row["status"] == JobStatus.COMPLETED.value:
                row["rank"] = rank
                rank += 1
            else:
                row["rank"] = "-"

        # Determine fieldnames dynamically based on the day
        if sample_day == 1:
            fieldnames = [
                "rank", "team_name", "status", "composite_score", "correct_pics", "accuracy",
                "mean_processing_s", "mean_total_s", "model_load_s", "total_wall_time_s",
                "slave_id", "start_temp", "end_temp", "throttled_flag", "job_id", "day", "error_message"
            ]
        elif sample_day == 2:
            fieldnames = [
                "rank", "team_name", "status", "composite_score", "count", "iou95", "iou90", "iou_average", "latency",
                "slave_id", "start_temp", "end_temp", "throttled_flag", "job_id", "day", "error_message"
            ]
        else:  # Day 3
            fieldnames = [
                "rank", "team_name", "status", "composite_score", "map_95", "f1_score", "class_accuracy",
                "mean_iou", "tp", "fp", "fn",
                "mean_processing_s", "mean_total_s", "model_load_s", "total_wall_time_s",
                "slave_id", "start_temp", "end_temp", "throttled_flag", "job_id", "day", "error_message"
            ]

        # Write CSV
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})

        return rows


aggregator = LeaderboardAggregator()
