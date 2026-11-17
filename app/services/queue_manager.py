import csv
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from threading import RLock
from app.models.job import Job, JobStatus, JobReturnRequest, EvaluationScore, HardwareReport


class QueueManager:
    """
    Thread-safe, multi-day job queue manager for master orchestrator.
    Uses RLock to prevent re-entrant deadlocks across worker threads.
    """

    def __init__(self, default_timeout_s: int = 300):
        self.default_timeout_s = default_timeout_s
        self._lock = RLock()
        # Storage: day -> {job_id: Job}
        self.jobs: Dict[int, Dict[str, Job]] = {1: {}, 2: {}, 3: {}}

    def _find_job_unlocked(self, job_id: str) -> Optional[Job]:
        """Internal helper to locate a job without acquiring lock again."""
        for day_jobs in self.jobs.values():
            if job_id in day_jobs:
                return day_jobs[job_id]
        return None

    @staticmethod
    def _float_or_default(val: Any, default: float = 0.0) -> float:
        try:
            return float(val) if val not in (None, "", "N/A", "-", "None") else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _int_or_default(val: Any, default: int = 0) -> int:
        try:
            return int(float(val)) if val not in (None, "", "N/A", "-", "None") else default
        except (ValueError, TypeError):
            return default

    def _parse_evaluation_from_row(self, row: Dict[str, Any], day: int) -> EvaluationScore:
        correct = 0
        total_imgs = 0
        correct_pics_str = str(row.get("correct_pics", "")).strip()
        if "/" in correct_pics_str:
            parts = correct_pics_str.split("/")
            correct = self._int_or_default(parts[0])
            total_imgs = self._int_or_default(parts[1])

        return EvaluationScore(
            day=day,
            total_images=total_imgs,
            correct_predictions=correct,
            accuracy=self._float_or_default(row.get("accuracy")),
            macro_f1=self._float_or_default(row.get("f1_score")),
            tp=self._int_or_default(row.get("tp")),
            fp=self._int_or_default(row.get("fp")),
            fn=self._int_or_default(row.get("fn")),
            precision=self._float_or_default(row.get("precision")),
            recall=self._float_or_default(row.get("recall")),
            f1_score=self._float_or_default(row.get("f1_score")),
            mean_iou=self._float_or_default(row.get("mean_iou")),
            map_95=self._float_or_default(row.get("map_95")),
            class_accuracy=self._float_or_default(row.get("class_accuracy")),
            mean_processing_seconds=self._float_or_default(row.get("mean_processing_s")),
            mean_total_seconds=self._float_or_default(row.get("mean_total_s")),
            model_load_seconds=self._float_or_default(row.get("model_load_s")),
            total_wall_time_s=self._float_or_default(row.get("total_wall_time_s")),
            composite_score=self._float_or_default(row.get("composite_score"))
        )

    def _parse_hardware_from_row(self, row: Dict[str, Any]) -> HardwareReport:
        st = row.get("start_temp")
        et = row.get("end_temp")
        return HardwareReport(
            start_temp=self._float_or_default(st) if st not in (None, "", "N/A", "-", "None") else None,
            end_temp=self._float_or_default(et) if et not in (None, "", "N/A", "-", "None") else None,
            throttled_flag=str(row.get("throttled_flag", "0x0"))
        )

    def load_jobs_from_bundles(
        self,
        day: int,
        bundles_dir: Path,
        team_names_map: Optional[Dict[str, str]] = None
    ) -> int:
        """Loads all bundles in bundles_dir into the queue for the specified day."""
        with self._lock:
            if day not in self.jobs:
                self.jobs[day] = {}

            added_count = 0
            for bundle_path in bundles_dir.glob("*.zip"):
                team_id = bundle_path.stem
                job_id = f"day{day}_{team_id}"

                # Do not overwrite if already existing and completed
                if job_id in self.jobs[day] and self.jobs[day][job_id].status == JobStatus.COMPLETED:
                    continue

                display_name = team_names_map.get(team_id, team_id) if team_names_map else team_id

                job = Job(
                    job_id=job_id,
                    day=day,
                    team_name=display_name,
                    bundle_path=str(bundle_path),
                    status=JobStatus.QUEUED
                )
                self.jobs[day][job_id] = job
                added_count += 1
            return added_count

    def requeue_failed_from_leaderboard(
        self,
        day: int,
        leaderboard_csv: Path,
        bundles_dir: Path,
        include_untested: bool = True
    ) -> Dict[str, Any]:
        """
        Parses final_leaderboard.csv:
        - Preserves all COMPLETED jobs in memory so their ranks and scores remain intact.
        - Identifies all FAILED jobs and re-enqueues them with status QUEUED so slaves can retry them.
        - Optionally includes any bundles present in bundles_dir that were never evaluated.
        """
        if not leaderboard_csv.exists():
            raise FileNotFoundError(f"Leaderboard CSV not found at: {leaderboard_csv}")

        with self._lock:
            if day not in self.jobs:
                self.jobs[day] = {}

            with open(leaderboard_csv, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            requeued_jobs = []
            completed_preserved = []
            missing_bundles = []

            for row in rows:
                status_str = str(row.get("status", "")).strip().upper()
                job_id = str(row.get("job_id", "")).strip()
                team_name = str(row.get("team_name", "")).strip()
                err_msg = str(row.get("error_message", "")).strip()

                if not job_id:
                    continue

                # Determine team_id and expected bundle path
                prefix = f"day{day}_"
                team_id = job_id[len(prefix):] if job_id.startswith(prefix) else job_id

                bundle_path = bundles_dir / f"{team_id}.zip"
                if not bundle_path.exists():
                    matches = [b for b in bundles_dir.glob("*.zip") if b.stem.lower() == team_id.lower()]
                    if matches:
                        bundle_path = matches[0]

                if status_str == "COMPLETED":
                    # Keep existing completed job if already in memory, else reconstruct
                    if job_id not in self.jobs[day] or self.jobs[day][job_id].status != JobStatus.COMPLETED:
                        eval_score = self._parse_evaluation_from_row(row, day)
                        hw_report = self._parse_hardware_from_row(row)
                        self.jobs[day][job_id] = Job(
                            job_id=job_id,
                            day=day,
                            team_name=team_name or team_id,
                            bundle_path=str(bundle_path),
                            status=JobStatus.COMPLETED,
                            assigned_slave=row.get("slave_id"),
                            evaluation=eval_score,
                            hardware_report=hw_report
                        )
                    completed_preserved.append(job_id)

                elif status_str in ("FAILED", "TIMEOUT"):
                    if bundle_path.exists():
                        # Reset job to QUEUED for retry
                        self.jobs[day][job_id] = Job(
                            job_id=job_id,
                            day=day,
                            team_name=team_name or team_id,
                            bundle_path=str(bundle_path),
                            status=JobStatus.QUEUED,
                            assigned_slave=None,
                            assigned_at=None,
                            completed_at=None,
                            error_message=None,
                            evaluation=None,
                            hardware_report=None
                        )
                        requeued_jobs.append({
                            "job_id": job_id,
                            "team_id": team_id,
                            "team_name": team_name,
                            "bundle_file": bundle_path.name,
                            "previous_error": err_msg
                        })
                    else:
                        missing_bundles.append({
                            "job_id": job_id,
                            "team_id": team_id,
                            "expected_bundle": str(bundle_path)
                        })

            # Optionally enqueue bundles that were not evaluated at all
            if include_untested and bundles_dir.exists():
                for b_path in bundles_dir.glob("*.zip"):
                    t_id = b_path.stem
                    j_id = f"day{day}_{t_id}"
                    if j_id not in self.jobs[day]:
                        self.jobs[day][j_id] = Job(
                            job_id=j_id,
                            day=day,
                            team_name=t_id,
                            bundle_path=str(b_path),
                            status=JobStatus.QUEUED
                        )
                        requeued_jobs.append({
                            "job_id": j_id,
                            "team_id": t_id,
                            "team_name": t_id,
                            "bundle_file": b_path.name,
                            "previous_error": "Never previously evaluated"
                        })

            return {
                "day": day,
                "total_leaderboard_rows": len(rows),
                "requeued_count": len(requeued_jobs),
                "completed_preserved_count": len(completed_preserved),
                "missing_bundle_count": len(missing_bundles),
                "requeued_jobs": requeued_jobs,
                "missing_bundles": missing_bundles
            }

    def poll_job(self, day: int, slave_id: str) -> Optional[Job]:
        """Assigns the next available QUEUED job to the requesting slave."""
        with self._lock:
            if day not in self.jobs:
                return None

            now = time.time()
            # First check for timed out jobs and reset them to QUEUED
            for job in self.jobs[day].values():
                if job.status == JobStatus.ASSIGNED:
                    if job.assigned_at and (now - job.assigned_at > self.default_timeout_s):
                        job.status = JobStatus.QUEUED
                        job.assigned_slave = None
                        job.assigned_at = None

            # Find next queued job
            for job in self.jobs[day].values():
                if job.status == JobStatus.QUEUED:
                    job.status = JobStatus.ASSIGNED
                    job.assigned_slave = slave_id
                    job.assigned_at = now
                    return job
            return None

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._find_job_unlocked(job_id)

    def complete_job(
        self,
        job_id: str,
        result_req: JobReturnRequest,
        evaluation: Optional[EvaluationScore] = None
    ) -> Optional[Job]:
        with self._lock:
            job = self._find_job_unlocked(job_id)
            if not job:
                return None

            job.completed_at = time.time()
            job.hardware_report = result_req.hardware_report

            if result_req.status.upper() == "SUCCESS":
                job.status = JobStatus.COMPLETED
                job.evaluation = evaluation
            else:
                job.status = JobStatus.FAILED
                job.error_message = result_req.error_message or result_req.logs

            return job

    def get_queue_status(self, day: int) -> Dict[str, any]:
        with self._lock:
            day_jobs = self.jobs.get(day, {})
            total = len(day_jobs)
            queued = sum(1 for j in day_jobs.values() if j.status == JobStatus.QUEUED)
            assigned = sum(1 for j in day_jobs.values() if j.status == JobStatus.ASSIGNED)
            completed = sum(1 for j in day_jobs.values() if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in day_jobs.values() if j.status == JobStatus.FAILED)

            return {
                "day": day,
                "total_jobs": total,
                "queued": queued,
                "assigned": assigned,
                "completed": completed,
                "failed": failed,
                "jobs": [j.model_dump() for j in day_jobs.values()]
            }


queue_manager = QueueManager()
