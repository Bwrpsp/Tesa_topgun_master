from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.config import settings
from app.services.downloader import HighSpeedDownloader
from app.services.packager import SubmissionPackager
from app.services.queue_manager import queue_manager

router = APIRouter(prefix="/api/v1/admin", tags=["Admin & Pipeline"])

downloader_instances: Dict[int, HighSpeedDownloader] = {}
packager = SubmissionPackager()


class SetActiveDayRequest(BaseModel):
    day: int


class EnqueueTeamRequest(BaseModel):
    team_id: str
    day: Optional[int] = None
    force: bool = True


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

    # Build team display name map from contestants.csv if available
    team_names_map = {}
    csv_path = settings.get_contestants_csv(day)
    if csv_path.exists():
        downloader_temp = HighSpeedDownloader()
        for item in downloader_temp.parse_contestants_csv(csv_path):
            tid = item["team_id"]
            tname = item.get("team_name")
            if tname and tname != tid:
                team_names_map[tid] = f"{tname} ({tid})"
            else:
                team_names_map[tid] = tid

    # Load bundles into queue
    enqueued_count = queue_manager.load_jobs_from_bundles(day, bundles_dir, team_names_map)

    return {
        "message": f"Prepared bundles and populated queue for Day {day}",
        "package_report": package_report,
        "enqueued_jobs": enqueued_count
    }


@router.post("/day/{day}/retry-failed")
async def retry_failed_jobs(
    day: int,
    include_untested: bool = Query(True, description="Also include valid bundles that were never evaluated")
):
    """
    Looks in final_leaderboard.csv for Day {day}, preserves all COMPLETED jobs,
    and re-enqueues all FAILED jobs into the queue so slaves can retry them.
    """
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    leaderboard_csv = settings.get_leaderboard_csv(day)
    if not leaderboard_csv.exists():
        raise HTTPException(
            status_code=404,
            detail=f"final_leaderboard.csv not found at {leaderboard_csv}. Please run initial evaluation first."
        )

    bundles_dir = settings.get_bundles_dir(day)
    if not bundles_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Bundles directory not found at {bundles_dir}."
        )

    report = queue_manager.requeue_failed_from_leaderboard(
        day=day,
        leaderboard_csv=leaderboard_csv,
        bundles_dir=bundles_dir,
        include_untested=include_untested
    )

    return {
        "message": f"Successfully re-enqueued {report['requeued_count']} failed jobs for Day {day}",
        "day": day,
        "requeued_count": report["requeued_count"],
        "completed_preserved_count": report["completed_preserved_count"],
        "missing_bundle_count": report["missing_bundle_count"],
        "requeued_jobs": report["requeued_jobs"],
        "missing_bundles": report["missing_bundles"]
    }


@router.post("/retry-failed")
async def retry_failed_jobs_active_day(
    day: Optional[int] = Query(None, description="Competition day (defaults to ACTIVE_DAY)"),
    include_untested: bool = Query(True, description="Also include valid bundles that were never evaluated")
):
    """Re-enqueues failed jobs from final_leaderboard.csv for the active competition day."""
    target_day = day if day is not None else settings.ACTIVE_DAY
    return await retry_failed_jobs(target_day, include_untested=include_untested)


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


def _resolve_and_enqueue_team(day: int, team_id: str, force: bool = True) -> Dict[str, Any]:
    """
    Core resolution logic for manually enqueuing a team:
    1. Normalizes team_id (handles TGR-XX, team_XX, case insensitivity, prefixes, .zip).
    2. Looks up team in contestants.csv for friendly name & download link.
    3. Finds existing bundle in bundles/ or creates it from raw_zips/ or downloads on demand.
    4. Enqueues job in queue_manager.
    """
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    raw_id = team_id.strip()
    if not raw_id:
        raise HTTPException(status_code=400, detail="team_id cannot be empty")

    # Strip day prefixes if user entered e.g. day1_TGR-01
    for pfx in (f"day{day}_", f"day_{day}_", "day1_", "day2_", "day3_"):
        if raw_id.lower().startswith(pfx):
            raw_id = raw_id[len(pfx):]
            break

    # Strip .zip extension if provided
    if raw_id.lower().endswith(".zip"):
        raw_id = raw_id[:-4]

    canonical_team_id = raw_id
    team_name = raw_id
    contestant_item = None

    # Check contestants.csv for friendly name and metadata
    csv_path = settings.get_contestants_csv(day)
    if csv_path.exists():
        dl_temp = HighSpeedDownloader()
        try:
            contestants = dl_temp.parse_contestants_csv(csv_path)
            for item in contestants:
                tid = item.get("team_id", "").strip()
                tname = item.get("team_name", "").strip()
                if tid.lower() == raw_id.lower() or (tname and tname.lower() == raw_id.lower()):
                    canonical_team_id = tid
                    team_name = f"{tname} ({tid})" if tname and tname != tid else tid
                    contestant_item = item
                    break
        except Exception as e:
            print(f"[Warning] Error parsing contestants.csv: {e}")

    bundles_dir = settings.get_bundles_dir(day)
    bundle_file = bundles_dir / f"{canonical_team_id}.zip"

    # Search for existing bundle (exact or case-insensitive)
    if not bundle_file.exists():
        matches = [f for f in bundles_dir.glob("*.zip") if f.stem.lower() == canonical_team_id.lower()]
        if matches:
            bundle_file = matches[0]
            canonical_team_id = bundle_file.stem

    # If bundle does not exist, check raw_zips or attempt on-demand download
    if not bundle_file.exists():
        raw_zips_dir = settings.get_raw_zips_dir(day)
        raw_zip = raw_zips_dir / f"{canonical_team_id}.zip"
        if not raw_zip.exists():
            matches = [f for f in raw_zips_dir.glob("*.zip") if f.stem.lower() == canonical_team_id.lower()]
            if matches:
                raw_zip = matches[0]
                canonical_team_id = raw_zip.stem

        if raw_zip.exists():
            extracted_dir = settings.get_extracted_dir(day)
            res = packager.process_team(canonical_team_id, raw_zip, extracted_dir, bundles_dir)
            if not res.get("is_valid"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Raw archive found for '{canonical_team_id}' but validation failed: {res.get('error')}"
                )
            bundle_file = Path(res["bundle_path"])
        elif contestant_item and contestant_item.get("url"):
            # Attempt on-demand download from contestants.csv link
            downloader = HighSpeedDownloader(
                max_concurrent=1,
                timeout_seconds=settings.DOWNLOAD_TIMEOUT_SECONDS
            )
            tid, success, err = downloader._download_item(contestant_item, raw_zips_dir, verbose=True)
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not download submission for team '{canonical_team_id}': {err}"
                )
            raw_zip = raw_zips_dir / f"{canonical_team_id}.zip"
            extracted_dir = settings.get_extracted_dir(day)
            res = packager.process_team(canonical_team_id, raw_zip, extracted_dir, bundles_dir)
            if not res.get("is_valid"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Downloaded archive for '{canonical_team_id}' but validation failed: {res.get('error')}"
                )
            bundle_file = Path(res["bundle_path"])
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Bundle for team '{canonical_team_id}' not found in {bundles_dir}, raw archive not found in {raw_zips_dir}, and no download link found in contestants.csv."
            )

    # Enqueue job
    job = queue_manager.enqueue_team_job(
        day=day,
        team_id=canonical_team_id,
        bundle_path=str(bundle_file),
        team_name=team_name,
        force=force
    )

    queued_jobs = [j for j in queue_manager.jobs.get(day, {}).values() if j.status.value == "QUEUED"]
    position = next((idx + 1 for idx, j in enumerate(queued_jobs) if j.job_id == job.job_id), len(queued_jobs))

    return {
        "message": f"Team '{canonical_team_id}' successfully enqueued for Day {day}",
        "job_id": job.job_id,
        "day": day,
        "team_id": canonical_team_id,
        "team_name": job.team_name,
        "bundle_path": str(bundle_file),
        "status": job.status.value,
        "queue_position": position,
        "total_queued_for_day": len(queued_jobs)
    }


@router.post("/queue/enqueue-team")
@router.post("/enqueue-team")
async def enqueue_team_endpoint(
    req: Optional[EnqueueTeamRequest] = None,
    team_id: Optional[str] = Query(None, description="Team ID from contestants.csv (e.g. TGR-01)"),
    day: Optional[int] = Query(None, description="Competition day (defaults to ACTIVE_DAY)"),
    force: bool = Query(True, description="Force re-enqueue even if previously evaluated")
):
    """
    Manually adds a team to the job queue using their team_id (e.g. 'TGR-01', 'TGR-23').
    Accepts input via JSON request body or URL query parameters.
    """
    target_team = req.team_id if (req and req.team_id) else team_id
    if not target_team:
        raise HTTPException(
            status_code=400,
            detail="team_id must be provided in request body ({'team_id': 'TGR-01'}) or query parameter (?team_id=TGR-01)"
        )

    target_day = req.day if (req and req.day is not None) else day
    if target_day is None:
        target_day = settings.ACTIVE_DAY

    target_force = req.force if req else force
    return _resolve_and_enqueue_team(day=target_day, team_id=target_team, force=target_force)


@router.post("/day/{day}/enqueue/{team_id}")
@router.get("/day/{day}/enqueue/{team_id}")
async def enqueue_team_by_path(
    day: int,
    team_id: str,
    force: bool = Query(True, description="Force re-enqueue even if previously evaluated")
):
    """
    Direct path endpoint to manually add a team to the queue:
    POST or GET /api/v1/admin/day/{day}/enqueue/{team_id}
    Example: /api/v1/admin/day/1/enqueue/TGR-01
    """
    return _resolve_and_enqueue_team(day=day, team_id=team_id, force=force)
