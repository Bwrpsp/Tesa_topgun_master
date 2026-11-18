import re
import json
import csv
import time
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, Tuple, List

from app.config import settings
from app.services.downloader import HighSpeedDownloader
from app.services.packager import SubmissionPackager
from app.services.queue_manager import queue_manager
from app.services.evaluator import SubmissionEvaluator
from app.services.aggregator import aggregator
from app.models.job import Job, JobStatus, HardwareReport

router = APIRouter(prefix="/api/v1/admin", tags=["Admin & Pipeline"])

downloader_instances: Dict[int, HighSpeedDownloader] = {}
packager = SubmissionPackager()
evaluator = SubmissionEvaluator()


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


def _match_contestant(extracted_id: str, contestants_csv: Path) -> Tuple[str, str]:
    """
    Matches extracted team ID against contestants.csv to get canonical team ID and display name.
    Returns: (canonical_team_id, display_team_name)
    """
    canonical_id = extracted_id
    display_name = extracted_id

    if not contestants_csv.exists():
        return canonical_id, display_name

    try:
        with open(contestants_csv, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        ext_norm = re.sub(r"[^a-zA-Z0-9]", "", extracted_id).lower()

        # 1. Exact match on Team ID
        for row in rows:
            cleaned = {k.strip().lower().replace(" ", "_").replace("-", "_"): (v.strip() if v else "") for k, v in row.items() if k}
            tid = cleaned.get("team_id") or cleaned.get("team") or cleaned.get("id") or ""
            tname = cleaned.get("team_name") or cleaned.get("name") or ""
            if tid.lower() == extracted_id.lower():
                canonical_id = tid
                display_name = f"{tname} ({tid})" if tname and tname != tid else tid
                return canonical_id, display_name

        # 2. Normalized alphanumeric match on Team ID
        for row in rows:
            cleaned = {k.strip().lower().replace(" ", "_").replace("-", "_"): (v.strip() if v else "") for k, v in row.items() if k}
            tid = cleaned.get("team_id") or cleaned.get("team") or cleaned.get("id") or ""
            tname = cleaned.get("team_name") or cleaned.get("name") or ""
            tid_norm = re.sub(r"[^a-zA-Z0-9]", "", tid).lower()
            if tid_norm and tid_norm == ext_norm:
                canonical_id = tid
                display_name = f"{tname} ({tid})" if tname and tname != tid else tid
                return canonical_id, display_name

        # 3. Match on Team Name
        for row in rows:
            cleaned = {k.strip().lower().replace(" ", "_").replace("-", "_"): (v.strip() if v else "") for k, v in row.items() if k}
            tid = cleaned.get("team_id") or cleaned.get("team") or cleaned.get("id") or ""
            tname = cleaned.get("team_name") or cleaned.get("name") or ""
            tname_norm = re.sub(r"[^a-zA-Z0-9]", "", tname).lower()
            if tname and (tname.lower() == extracted_id.lower() or tname_norm == ext_norm):
                canonical_id = tid or extracted_id
                display_name = f"{tname} ({canonical_id})" if tid and tname != canonical_id else tname
                return canonical_id, display_name

        # 4. Numeric matching heuristic (e.g. TGR-67 matching row with 67)
        digits = re.findall(r"\d+", extracted_id)
        if digits:
            num = digits[0]
            for row in rows:
                cleaned = {k.strip().lower().replace(" ", "_").replace("-", "_"): (v.strip() if v else "") for k, v in row.items() if k}
                tid = cleaned.get("team_id") or cleaned.get("team") or cleaned.get("id") or ""
                tname = cleaned.get("team_name") or cleaned.get("name") or ""
                if (tid and re.findall(r"\d+", tid) == [num]) or (tname and re.findall(r"\d+", tname) == [num]):
                    display_name = f"{tname} ({extracted_id})" if tname else extracted_id
                    return extracted_id, display_name

    except Exception as e:
        print(f"[Warning] Error parsing contestants CSV: {e}")

    return canonical_id, display_name


def prepare_single_team_and_enqueue(day: int, team_id: str) -> Dict[str, Any]:
    """
    Looks for that team's raw zip in raw_zips/ (or downloads it if link in contestants.csv),
    unpacks it, validates (infer.py, requirements.txt, weights/), packages it into a bundle in bundles/,
    and adds it to the queue.
    """
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    raw_id = team_id.strip()
    if not raw_id:
        raise HTTPException(status_code=400, detail="team_id cannot be empty")

    for pfx in (f"day{day}_", f"day_{day}_", "day1_", "day2_", "day3_"):
        if raw_id.lower().startswith(pfx):
            raw_id = raw_id[len(pfx):]
            break

    if raw_id.lower().endswith(".zip"):
        raw_id = raw_id[:-4]

    raw_dir = settings.get_raw_zips_dir(day)
    extracted_dir = settings.get_extracted_dir(day)
    bundles_dir = settings.get_bundles_dir(day)
    contestants_csv = settings.get_contestants_csv(day)

    canonical_id, display_name = _match_contestant(raw_id, contestants_csv)

    # Find raw zip
    raw_zip = raw_dir / f"{canonical_id}.zip"
    if not raw_zip.exists():
        raw_zip = raw_dir / f"{raw_id}.zip"

    if not raw_zip.exists():
        matches = [f for f in raw_dir.glob("*.zip") if f.stem.lower() in (canonical_id.lower(), raw_id.lower())]
        if matches:
            raw_zip = matches[0]
            canonical_id = raw_zip.stem

    if not raw_zip.exists():
        # Check if contestant has download link in contestants.csv
        contestant_item = None
        if contestants_csv.exists():
            dl_temp = HighSpeedDownloader()
            for item in dl_temp.parse_contestants_csv(contestants_csv):
                if item.get("team_id", "").lower() in (canonical_id.lower(), raw_id.lower()):
                    contestant_item = item
                    break

        if contestant_item and contestant_item.get("url"):
            dl = HighSpeedDownloader(max_concurrent=1, timeout_seconds=settings.DOWNLOAD_TIMEOUT_SECONDS)
            tid, success, err = dl._download_item(contestant_item, raw_dir, verbose=True)
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not download submission for team '{canonical_id}': {err}"
                )
            raw_zip = raw_dir / f"{canonical_id}.zip"
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Raw zip archive for team '{canonical_id}' not found in {raw_dir} and no download URL found."
            )

    # Extract, validate and create clean bundle
    package_res = packager.process_team(canonical_id, raw_zip, extracted_dir, bundles_dir)
    if not package_res.get("is_valid"):
        raise HTTPException(
            status_code=400,
            detail=f"Raw archive found for '{canonical_id}' but validation failed: {package_res.get('error')}"
        )

    bundle_file = Path(package_res["bundle_path"])

    # Enqueue job
    job = queue_manager.enqueue_team_job(
        day=day,
        team_id=canonical_id,
        bundle_path=str(bundle_file),
        team_name=display_name,
        force=True
    )

    queued_jobs = [j for j in queue_manager.jobs.get(day, {}).values() if j.status.value == "QUEUED"]
    position = next((idx + 1 for idx, j in enumerate(queued_jobs) if j.job_id == job.job_id), len(queued_jobs))

    return {
        "message": f"Successfully extracted, bundled, and queued team '{canonical_id}' for Day {day}",
        "day": day,
        "team_id": canonical_id,
        "team_name": display_name,
        "job_id": job.job_id,
        "status": job.status.value,
        "bundle_path": str(bundle_file),
        "weights_dir": package_res.get("weights_dir"),
        "weights_count": package_res.get("weights_count"),
        "weight_files": package_res.get("weight_files"),
        "queue_position": position,
        "total_queued_for_day": len(queued_jobs)
    }


@router.post("/day/{day}/prepare-queue")
@router.post("/day/{day}/prepare")
async def prepare_queue_for_day(
    day: int,
    team_id: Optional[str] = Query(None, description="Optional team_id to prepare only 1 team (e.g. TGR-67)")
):
    """
    Stage 2: Unpacks raw zip(s), validates required files (infer.py, requirements.txt, weights),
    creates clean sanitized bundle(s), and enqueues job(s) for slaves.
    If team_id is provided, only prepares and queues that single team.
    If team_id is omitted, batch prepares all raw zips for that day.
    """
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    if team_id:
        return prepare_single_team_and_enqueue(day=day, team_id=team_id)

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


@router.post("/day/{day}/prepare/{team_id}")
@router.get("/day/{day}/prepare/{team_id}")
async def prepare_single_team_endpoint(day: int, team_id: str):
    """
    Direct path endpoint to prepare, bundle, and enqueue a single team:
    POST or GET /api/v1/admin/day/{day}/prepare/{team_id}
    Example: /api/v1/admin/day/2/prepare/TGR-67
    """
    return prepare_single_team_and_enqueue(day=day, team_id=team_id)


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


# -------------------------------------------------------------
# Manual Predictions Evaluation Helpers & Endpoints
# -------------------------------------------------------------
def extract_team_id_from_prediction_filename(filename: str, day: int) -> str:
    """
    Extracts team identifier from prediction filenames such as:
    - day2_TGR-67_predictions.json -> TGR-67
    - TGR-03_predictions.json     -> TGR-03
    - TGR-12.json                 -> TGR-12
    - day1_team_03.csv            -> team_03
    """
    stem = Path(filename).name
    # Strip file extensions
    for ext in (".json", ".csv", ".txt"):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break

    # Strip suffixes like _predictions, _prediction, _preds, _pred
    stem = re.sub(r"[._](predictions|prediction|preds|pred)$", "", stem, flags=re.IGNORECASE)

    # Strip day prefixes like day1_, day2_, day_1_, day_2_
    stem = re.sub(r"^day_?\d+_", "", stem, flags=re.IGNORECASE)

    return stem.strip()


def evaluate_predictions_for_day(
    day: int,
    force: bool = False,
    target_team_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scans results/day_{day}/predictions/:
    1. Extracts team_id from prediction filename.
    2. Matches against contestants.csv to resolve display name.
    3. Checks final_leaderboard.csv: if team already has a completed score and force=False, skips.
    4. Evaluates predictions against ground_truth.json (Day 2/3) or ground_truth.csv (Day 1).
    5. Appends/updates the completed score and regenerates final_leaderboard.csv with updated ranks.
    """
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    gt_path = settings.get_ground_truth_path(day)
    if not gt_path.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth file not found at {gt_path} for Day {day}")

    predictions_dir = settings.get_day_results_dir(day) / "predictions"
    if not predictions_dir.exists():
        return {
            "message": f"Predictions directory does not exist at {predictions_dir}",
            "day": day,
            "total_files": 0,
            "evaluated_count": 0,
            "skipped_count": 0,
            "evaluated": [],
            "skipped": []
        }

    pred_files = [f for f in predictions_dir.iterdir() if f.is_file() and f.suffix.lower() in (".json", ".csv")]
    if not pred_files:
        return {
            "message": f"No prediction files (.json, .csv) found in {predictions_dir}",
            "day": day,
            "total_files": 0,
            "evaluated_count": 0,
            "skipped_count": 0,
            "evaluated": [],
            "skipped": []
        }

    leaderboard_csv = settings.get_leaderboard_csv(day)
    contestants_csv = settings.get_contestants_csv(day)

    # 1. Load existing jobs from final_leaderboard.csv to preserve all scores
    queue_manager.load_leaderboard_jobs(day, leaderboard_csv)

    # 2. Identify jobs that already have completed scores in final_leaderboard.csv
    existing_completed_job_ids = set()
    existing_completed_team_ids = set()
    if leaderboard_csv.exists():
        with open(leaderboard_csv, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                st = str(r.get("status", "")).strip().upper()
                jid = str(r.get("job_id", "")).strip()
                tname = str(r.get("team_name", "")).strip()
                try:
                    c_score = float(r.get("composite_score", 0.0))
                except (ValueError, TypeError):
                    c_score = 0.0

                if st == "COMPLETED" and c_score >= 0:
                    if jid:
                        existing_completed_job_ids.add(jid.lower())
                        prefix = f"day{day}_"
                        if jid.lower().startswith(prefix.lower()):
                            existing_completed_team_ids.add(jid[len(prefix):].lower())
                    if tname:
                        existing_completed_team_ids.add(tname.lower())

    evaluated_records = []
    skipped_records = []
    error_records = []

    for pred_file in pred_files:
        extracted_id = extract_team_id_from_prediction_filename(pred_file.name, day)
        if not extracted_id:
            continue

        if target_team_id:
            tgt_clean = target_team_id.strip().lower()
            prefix = f"day{day}_"
            if tgt_clean.startswith(prefix):
                tgt_clean = tgt_clean[len(prefix):]
            if extracted_id.lower() != tgt_clean:
                continue

        canonical_id, display_name = _match_contestant(extracted_id, contestants_csv)
        expected_job_id = f"day{day}_{canonical_id}"

        # Check if already scored in final_leaderboard
        is_already_scored = (
            expected_job_id.lower() in existing_completed_job_ids
            or canonical_id.lower() in existing_completed_team_ids
            or extracted_id.lower() in existing_completed_team_ids
        )

        if is_already_scored and not force:
            skipped_records.append({
                "file": pred_file.name,
                "team_id": canonical_id,
                "job_id": expected_job_id,
                "reason": "Already has completed score in final_leaderboard.csv (use force=True to re-evaluate)"
            })
            continue

        # Parse prediction file
        try:
            if pred_file.suffix.lower() == ".json":
                with open(pred_file, mode="r", encoding="utf-8") as f:
                    pred_data = json.load(f)
            else:
                with open(pred_file, mode="r", encoding="utf-8-sig") as f:
                    pred_data = list(csv.DictReader(f))
        except Exception as e:
            error_records.append({
                "file": pred_file.name,
                "team_id": canonical_id,
                "error": f"Failed to parse file: {e}"
            })
            continue

        # Look for optional timings or benchmark
        timings = None
        benchmark = None
        wall_time_s = 0.0

        for t_name in (f"{canonical_id}_timings.csv", f"day{day}_{canonical_id}_timings.csv", f"{extracted_id}_timings.csv"):
            t_path = pred_file.parent / t_name
            if t_path.exists():
                try:
                    with open(t_path, mode="r", encoding="utf-8-sig") as f:
                        timings = list(csv.DictReader(f))
                    break
                except Exception:
                    pass

        for b_name in (f"{canonical_id}_benchmark.json", f"day{day}_{canonical_id}_benchmark.json", f"{extracted_id}_benchmark.json"):
            b_path = pred_file.parent / b_name
            if b_path.exists():
                try:
                    with open(b_path, mode="r", encoding="utf-8") as f:
                        benchmark = json.load(f)
                    break
                except Exception:
                    pass

        # Run evaluation
        try:
            score = evaluator.evaluate(
                day=day,
                ground_truth_path=gt_path,
                predictions=pred_data,
                timings=timings,
                benchmark=benchmark,
                total_wall_time_s=wall_time_s
            )
        except Exception as e:
            error_records.append({
                "file": pred_file.name,
                "team_id": canonical_id,
                "error": f"Evaluation error: {e}"
            })
            continue

        bundle_path = settings.get_bundles_dir(day) / f"{canonical_id}.zip"
        job = Job(
            job_id=expected_job_id,
            day=day,
            team_name=display_name,
            bundle_path=str(bundle_path) if bundle_path.exists() else "",
            status=JobStatus.COMPLETED,
            assigned_slave="manual-eval",
            completed_at=time.time(),
            evaluation=score,
            hardware_report=HardwareReport(
                start_temp=40.0,
                end_temp=40.0,
                throttled_flag="0x0"
            )
        )

        with queue_manager._lock:
            queue_manager.jobs[day][expected_job_id] = job

        record = {
            "file": pred_file.name,
            "team_id": canonical_id,
            "team_name": display_name,
            "job_id": expected_job_id,
            "status": "COMPLETED",
            "composite_score": score.composite_score
        }
        if day == 2:
            record.update({
                "count": f"{score.count_accuracy:.2f}%",
                "iou95": score.iou95,
                "iou90": score.iou90,
                "iou_average": score.iou_average,
                "latency": score.latency
            })
        elif day == 1:
            record.update({
                "accuracy": f"{score.accuracy:.2f}%",
                "correct_pics": f"{score.correct_predictions}/{score.total_images}"
            })
        else:
            record.update({
                "map_95": score.map_95,
                "f1_score": score.f1_score
            })

        evaluated_records.append(record)

    # If any jobs were evaluated, update final_leaderboard.csv
    if evaluated_records:
        with queue_manager._lock:
            all_jobs = list(queue_manager.jobs.get(day, {}).values())
        leaderboard_rows = aggregator.generate_leaderboard(all_jobs, leaderboard_csv)

        # Attach rank from generated leaderboard to records
        for rec in evaluated_records:
            jid = rec["job_id"]
            rank_val = next((r["rank"] for r in leaderboard_rows if r.get("job_id") == jid), "-")
            rec["rank"] = rank_val

    return {
        "message": f"Processed {len(pred_files)} prediction file(s) for Day {day}: {len(evaluated_records)} evaluated, {len(skipped_records)} skipped, {len(error_records)} errors.",
        "day": day,
        "total_files": len(pred_files),
        "evaluated_count": len(evaluated_records),
        "skipped_count": len(skipped_records),
        "error_count": len(error_records),
        "evaluated": evaluated_records,
        "skipped": skipped_records,
        "errors": error_records
    }


@router.post("/day/{day}/evaluate-predictions")
@router.get("/day/{day}/evaluate-predictions")
async def evaluate_predictions_day_endpoint(
    day: int,
    force: bool = Query(False, description="Re-evaluate even if already scored in final_leaderboard.csv"),
    team_id: Optional[str] = Query(None, description="Optional team_id to evaluate only a specific prediction file")
):
    """
    Evaluates prediction files in results/day_{day}/predictions/:
    - Extracts team_id from filename.
    - Matches with contestants.csv for friendly name.
    - If team has no completed score in final_leaderboard.csv (or force=True), evaluates it.
    - Appends/updates the score and re-ranks final_leaderboard.csv immediately.
    """
    return evaluate_predictions_for_day(day=day, force=force, target_team_id=team_id)


@router.post("/evaluate-predictions")
@router.get("/evaluate-predictions")
async def evaluate_predictions_active_day_endpoint(
    day: Optional[int] = Query(None, description="Competition day (defaults to ACTIVE_DAY)"),
    force: bool = Query(False, description="Re-evaluate even if already scored in final_leaderboard.csv"),
    team_id: Optional[str] = Query(None, description="Optional team_id to evaluate only a specific prediction file")
):
    """Evaluates prediction files for the active competition day."""
    target_day = day if day is not None else settings.ACTIVE_DAY
    return evaluate_predictions_for_day(day=target_day, force=force, target_team_id=team_id)
