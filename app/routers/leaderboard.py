from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any

from app.config import settings
from app.services.queue_manager import queue_manager
from app.services.aggregator import aggregator

router = APIRouter(prefix="/api/v1/leaderboard", tags=["Leaderboard & Scoring"])


@router.get("/day/{day}")
async def get_leaderboard_data(day: int) -> List[Dict[str, Any]]:
    """Retrieves current ranked leaderboard for the specified day."""
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    day_jobs = list(queue_manager.jobs.get(day, {}).values())
    leaderboard_csv = settings.get_leaderboard_csv(day)
    return aggregator.generate_leaderboard(day_jobs, leaderboard_csv)


@router.get("/day/{day}/csv")
async def download_leaderboard_csv(day: int):
    """Downloads the final_leaderboard.csv for the specified day."""
    if day not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Day must be 1, 2, or 3")

    csv_path = settings.get_leaderboard_csv(day)
    if not csv_path.exists():
        # Try generating it
        day_jobs = list(queue_manager.jobs.get(day, {}).values())
        aggregator.generate_leaderboard(day_jobs, csv_path)

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Leaderboard CSV not yet generated")

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"final_leaderboard_day_{day}.csv"
    )
