from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from typing import Optional

from app.config import settings
from app.models.job import JobPollResponse, JobReturnRequest
from app.services.queue_manager import queue_manager
from app.services.evaluator import SubmissionEvaluator
from app.services.aggregator import aggregator

router = APIRouter(prefix="/api/v1/jobs", tags=["Slave Jobs"])
evaluator = SubmissionEvaluator()


@router.get("/poll", response_model=Optional[JobPollResponse])
async def poll_for_job(
    slave_id: str = Query(..., description="Unique hostname or identifier of the Raspberry Pi"),
    current_temp: Optional[float] = Query(None, description="Current CPU temperature of Pi"),
    throttled_flag: Optional[str] = Query("0x0", description="Current vcgencmd throttled hex string")
):
    """
    Slaves poll this endpoint continuously to receive evaluation tasks.
    If a job is ready for the current active day, returns 200 with job payload.
    If no jobs are queued, returns 204 No Content.
    """
    active_day = settings.ACTIVE_DAY
    job = queue_manager.poll_job(day=active_day, slave_id=slave_id)

    if not job:
        # 204 No Content instructs the slave to sleep and retry
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return JobPollResponse(
        job_id=job.job_id,
        day=job.day,
        team_name=job.team_name,
        bundle_url=f"/api/v1/jobs/{job.job_id}/bundle",
        test_data_dir_name=f"day_{job.day}",
        timeout_seconds=settings.JOB_TIMEOUT_SECONDS
    )


@router.get("/{job_id}/bundle")
async def download_bundle(job_id: str):
    """
    Slaves download the sanitized contestant bundle zip (containing infer.py,
    requirements.txt, weights/, and custom python modules).
    """
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    bundle_path = Path(job.bundle_path)
    if not bundle_path.exists():
        raise HTTPException(status_code=404, detail=f"Bundle file not found at {bundle_path}")

    return FileResponse(
        path=bundle_path,
        media_type="application/zip",
        filename=f"{job.team_name}_bundle.zip"
    )


@router.post("/{job_id}/return")
async def return_job_result(job_id: str, payload: JobReturnRequest):
    """
    Slaves post back execution outputs, timings, benchmark.json, logs, and hardware stats.
    Evaluator automatically scores predictions against the day's ground truth and
    updates final_leaderboard.csv immediately.
    """
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found in queue")

    evaluation = None
    if payload.status.upper() == "SUCCESS":
        gt_path = settings.get_ground_truth_path(job.day)
        if gt_path.exists():
            try:
                evaluation = evaluator.evaluate(
                    day=job.day,
                    ground_truth_path=gt_path,
                    predictions=payload.predictions,
                    timings=payload.timings,
                    benchmark=payload.benchmark,
                    total_wall_time_s=payload.total_wall_time_s
                )
            except Exception as e:
                # Ground truth evaluation error
                payload.status = "FAILED"
                payload.error_message = f"Evaluation scoring error: {e}"
        else:
            # If ground truth is not uploaded yet, compute latency metrics only
            evaluation = None

    # Update job status in queue
    updated_job = queue_manager.complete_job(job_id, payload, evaluation)

    # Immediately regenerate leaderboard CSV for this day
    day_jobs = [j for j in queue_manager.jobs.get(job.day, {}).values()]
    leaderboard_csv = settings.get_leaderboard_csv(job.day)
    aggregator.generate_leaderboard(day_jobs, leaderboard_csv)

    return {
        "message": f"Job {job_id} recorded successfully",
        "status": updated_job.status.value,
        "evaluation": updated_job.evaluation
    }
