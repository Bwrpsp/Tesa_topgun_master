import re
import csv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gdown
import httpx


def extract_gdrive_id(url: str) -> Optional[str]:
    """Extracts the Google Drive file ID from various link formats."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
        r"/uc\?id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


class HighSpeedDownloader:
    """
    Downloads all contestant submissions concurrently at maximum speed.
    Saves raw files directly into raw_zips/ directory first without decompressing.
    """

    def __init__(self, max_concurrent: int = 6, timeout_seconds: int = 180):
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.status = {
            "is_running": False,
            "total": 0,
            "completed": 0,
            "failed": 0,
            "errors": {}
        }

    def _download_single_gdrive(self, file_id: str, dest_path: Path) -> bool:
        """Downloads a file using gdown to handle Google Drive confirmation tokens."""
        url = f"https://drive.google.com/uc?id={file_id}"
        output = gdown.download(url, str(dest_path), quiet=True, fuzzy=True)
        return output is not None and dest_path.exists() and dest_path.stat().st_size > 0

    def _download_single_direct(self, url: str, dest_path: Path) -> bool:
        """Downloads standard direct HTTP/HTTPS files."""
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
        return dest_path.exists() and dest_path.stat().st_size > 0

    def _download_item(self, item: Dict[str, str], raw_dir: Path) -> Tuple[str, bool, Optional[str]]:
        team_id = item.get("team_id") or item.get("team_name")
        url = item.get("drive_url") or item.get("link") or item.get("url", "")
        if not team_id or not url:
            return str(team_id), False, "Missing team_id or URL in CSV"

        dest_file = raw_dir / f"{team_id}.zip"
        try:
            gdrive_id = extract_gdrive_id(url)
            if gdrive_id:
                success = self._download_single_gdrive(gdrive_id, dest_file)
            else:
                success = self._download_single_direct(url, dest_file)

            if success:
                return team_id, True, None
            else:
                return team_id, False, "Downloaded file was empty or missing"
        except Exception as e:
            return team_id, False, str(e)

    def parse_contestants_csv(self, csv_path: Path) -> List[Dict[str, str]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Contestants CSV not found at: {csv_path}")

        contestants = []
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize key names
                cleaned_row = {k.strip().lower(): v.strip() for k, v in row.items() if k and v}
                # Map fields
                team_id = cleaned_row.get("team_id") or cleaned_row.get("team") or cleaned_row.get("team_name")
                url = cleaned_row.get("drive_url") or cleaned_row.get("drive_link") or cleaned_row.get("link") or cleaned_row.get("url")
                if team_id and url:
                    contestants.append({
                        "team_id": team_id,
                        "team_name": cleaned_row.get("team_name", team_id),
                        "url": url
                    })
        return contestants

    async def download_all(self, contestants_csv: Path, output_dir: Path) -> Dict[str, any]:
        """
        Executes concurrent parallel downloads for all contestants listed in the CSV.
        All raw archives are saved into output_dir without decompression.
        """
        contestants = self.parse_contestants_csv(contestants_csv)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.status = {
            "is_running": True,
            "total": len(contestants),
            "completed": 0,
            "failed": 0,
            "errors": {}
        }

        loop = asyncio.get_running_loop()
        results = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = [
                loop.run_in_executor(pool, self._download_item, item, output_dir)
                for item in contestants
            ]
            for future in asyncio.as_completed(futures):
                team_id, success, error = await future
                if success:
                    self.status["completed"] += 1
                    results.append({"team_id": team_id, "status": "SUCCESS"})
                else:
                    self.status["failed"] += 1
                    self.status["errors"][team_id] = error
                    results.append({"team_id": team_id, "status": "FAILED", "error": error})

        self.status["is_running"] = False
        return {
            "total": self.status["total"],
            "completed": self.status["completed"],
            "failed": self.status["failed"],
            "results": results,
            "errors": self.status["errors"]
        }
