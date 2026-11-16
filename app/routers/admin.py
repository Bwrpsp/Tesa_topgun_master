from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any

from app.config import settings
from app.services.downloader import HighSpeedDownloader
from app.services.packager import SubmissionPackager
from app.services.queue_manager import queue_manager

router = APIRouter(prefix="/api/v1/admin", tags=["Admin & Pipeline"])

downloader_instances: Dict[int, HighSpeedDownloader] = {}
packager = SubmissionPackager()


class SetActiveDayRequest(BaseModel):
    day: int


@router.post("/day/{day}/download-all")
async def trigger_download_all(day: int, background_tasks: BackgroundTasks):
    """
    Stage 1: Triggers high-speed concurrent download of all contestant zip files
    from Drive/links listed in data/day_{day}/contestants.csv.
    """
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    csv_path = settings.get_contestants_csv(day)
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"contestants.csv not found at {csv_path}. Please place the CSV file first."
        )

    raw_dir = settings.get_raw_zips_dir(day)
    downloader = HighSpeedDownloader(
        max_concurrent=settings.DOWNLOAD_MAX_CONCURRENT,
        timeout_seconds=settings.DOWNLOAD_TIMEOUT_SECONDS
    )
    downloader_instances[day] = downloader

    background_tasks.add_task(downloader.download_all, csv_path, raw_dir)

    return {
        "message": f"Started high-speed bulk download for Day {day}",
        "raw_destination": str(raw_dir),
        "status_endpoint": f"/api/v1/admin/day/{day}/download-status"
    }


@router.get("/day/{day}/download-status")
async def get_download_status(day: int):
    """Checks the progress of Stage 1 bulk downloads."""
    downloader = downloader_instances.get(day)
    if not downloader:
        # Check files directly in raw_zips
        raw_dir = settings.get_raw_zips_dir(day)
        existing_zips = list(raw_dir.glob("*.zip"))
        return {
            "is_running": False,
            "existing_raw_zips_count": len(existing_zips),
            "files": [f.name for f in existing_zips]
        }

    return downloader.status


@router.post("/day/{day}/prepare-queue")
async def prepare_queue_for_day(day: int):
    """
    Stage 2: Batch unpacks raw zips, validates required files (infer.py, requirements.txt, weights),
    creates clean sanitized bundles, and enqueues jobs for slaves.
    """
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    raw_dir = settings.get_raw_zips_dir(day)
    extracted_dir = settings.get_extracted_dir(day)
    bundles_dir = settings.get_bundles_dir(day)

    if not any(raw_dir.glob("*.zip")):
        raise HTTPException(
            status_code=400,
            detail=f"No raw zip files found in {raw_dir}. Run download first."
        )

    # Process all zips
    package_report = packager.process_all_for_day(raw_dir, extracted_dir, bundles_dir)

    # Load bundles into queue
    enqueued_count = queue_manager.load_jobs_from_bundles(day, bundles_dir)

    return {
        "message": f"Prepared bundles and populated queue for Day {day}",
        "package_report": package_report,
        "enqueued_jobs": enqueued_count
    }


@router.post("/set-active-day")
async def set_active_day(req: SetActiveDayRequest):
    """Switches the active competition day for slaves."""
    if req.day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")
    settings.ACTIVE_DAY = req.day
    return {"message": f"Active day switched to Day {req.day}", "active_day": settings.ACTIVE_DAY}


@router.get("/status")
async def get_overall_status():
    """Full overview of active day, queues, and system state."""
    active_day = settings.ACTIVE_DAY
    return {
        "active_day": active_day,
        "queue_status": queue_manager.get_queue_status(active_day),
        "data_dirs": {
            f"day_{d}": {
                "contestants_csv_exists": settings.get_contestants_csv(d).exists(),
                "ground_truth_exists": settings.get_ground_truth_csv(d).exists(),
                "raw_zips_count": len(list(settings.get_raw_zips_dir(d).glob("*.zip"))),
                "bundles_count": len(list(settings.get_bundles_dir(d).glob("*.zip")))
            } for d in (1, 2, 3)
        }
    }
