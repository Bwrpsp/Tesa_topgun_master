import time
from typing import Dict, List, Optional
from pathlib import Path
from threading import Lock
from app.models.job import Job, JobStatus, JobReturnRequest, EvaluationScore


class QueueManager:
    """
    Thread-safe, multi-day job queue manager for master orchestrator.
    """

    def __init__(self, default_timeout_s: int = 300):
        self.default_timeout_s = default_timeout_s
        self._lock = Lock()
        # Storage: day -> {job_id: Job}
        self.jobs: Dict[int, Dict[str, Job]] = {1: {}, 2: {}, 3: {}}

    def load_jobs_from_bundles(self, day: int, bundles_dir: Path) -> int:
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

                job = Job(
                    job_id=job_id,
                    day=day,
                    team_name=team_id,
                    bundle_path=str(bundle_path),
                    status=JobStatus.QUEUED
                )
                self.jobs[day][job_id] = job
                added_count += 1
            return added_count

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
            for day_jobs in self.jobs.values():
                if job_id in day_jobs:
                    return day_jobs[job_id]
            return None

    def complete_job(
        self,
        job_id: str,
        result_req: JobReturnRequest,
        evaluation: Optional[EvaluationScore] = None
    ) -> Optional[Job]:
        with self._lock:
            job = self.get_job(job_id)
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
