import re
import csv
import sys
import time
import shutil
import zipfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import gdown
import httpx
from bs4 import BeautifulSoup


def format_bytes(size: int) -> str:
    """Formats raw byte counts into human-readable strings."""
    s = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if s < 1024.0:
            return f"{s:.2f} {unit}"
        s /= 1024.0
    return f"{s:.2f} TB"


def render_progress_bar(downloaded: int, total: int, width: int = 28) -> str:
    """Renders a single-line ASCII progress bar."""
    if total > 0:
        pct = min(1.0, downloaded / total)
        filled = int(width * pct)
        bar = "=" * filled + (">" if filled < width else "")
        bar = bar.ljust(width, " ")
        return f"[{bar}] {pct * 100:5.1f}% | {format_bytes(downloaded)} / {format_bytes(total)}"
    return f"[{format_bytes(downloaded)} downloaded]"


def extract_gdrive_id(url: str) -> Tuple[Optional[str], bool]:
    """
    Extracts Google Drive ID and whether it is explicitly a folder URL.
    Returns: (id_string, is_folder)
    """
    if not url:
        return None, False

    # Folder patterns
    folder_patterns = [
        r"/drive/folders/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
    ]
    for pattern in folder_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), True

    # File patterns
    file_patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
        r"/uc\?id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in file_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), False

    return None, False


class HighSpeedDownloader:
    """
    Downloads all contestant submissions reliably and cleanly.
    Handles Google Drive files, Google Drive folders, large files with virus scan warnings,
    and direct HTTP/HTTPS URLs. Supports sequential verbose execution or concurrent execution.
    """

    def __init__(self, max_concurrent: int = 4, timeout_seconds: int = 180):
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.status: Dict[str, Any] = {
            "is_running": False,
            "total": 0,
            "completed": 0,
            "failed": 0,
            "current_team": "",
            "errors": {}
        }

    def _is_html_file(self, file_path: Path) -> bool:
        """Checks if a downloaded file is actually an HTML error/login page instead of a binary zip."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(100).lower()
                return b"<!doctype" in header or b"<html" in header or b"<head" in header
        except Exception:
            return False

    def _stream_response_to_file(
        self,
        response: httpx.Response,
        dest_path: Path,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Streams an HTTP response to disk with optional live ASCII progress bar."""
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        last_print = 0.0

        with open(dest_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if verbose:
                    now = time.time()
                    if now - last_print > 0.15 or (total > 0 and downloaded >= total):
                        last_print = now
                        bar_str = render_progress_bar(downloaded, total)
                        sys.stdout.write(f"\r      {bar_str}")
                        sys.stdout.flush()

        if verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if dest_path.exists() and dest_path.stat().st_size > 0:
            if self._is_html_file(dest_path):
                dest_path.unlink(missing_ok=True)
                return False, "Downloaded content was an HTML web page instead of a zip file. Check link permissions."
            return True, None

        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        return False, "Downloaded file was empty or missing."

    def _download_gdrive_file_stream(
        self,
        file_id: str,
        dest_path: Path,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Downloads a Google Drive file directly, automatically handling large file
        virus scan warnings, confirmation tokens, and redirects.
        """
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        try:
            with httpx.Client(follow_redirects=True, timeout=self.timeout_seconds) as client:
                with client.stream("GET", url) as response:
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/html" in content_type:
                        html_text = response.read().decode("utf-8", errors="ignore")
                        if "Virus scan warning" in html_text or "download anyway" in html_text.lower():
                            soup = BeautifulSoup(html_text, "html.parser")
                            form = soup.find("form")
                            action = form.get("action") if form else "https://drive.usercontent.google.com/download"
                            params = {}
                            if form:
                                for inp in form.find_all("input"):
                                    n = inp.get("name")
                                    v = inp.get("value", "")
                                    if n:
                                        params[n] = v
                            if "id" not in params:
                                params["id"] = file_id
                            if "confirm" not in params:
                                params["confirm"] = "t"

                            with client.stream("GET", action, params=params) as form_resp:
                                form_resp.raise_for_status()
                                return self._stream_response_to_file(form_resp, dest_path, verbose=verbose)
                        elif "Access denied" in html_text or "permission" in html_text.lower() or "sign in" in html_text.lower():
                            return False, "Permission denied on Google Drive link ('Anyone with the link' permission required)."
                        elif "File not found" in html_text or "404" in html_text:
                            return False, "File not found on Google Drive (HTTP 404)."
                        else:
                            return False, "Google Drive returned an HTML page instead of binary file. Check permissions."
                    else:
                        response.raise_for_status()
                        return self._stream_response_to_file(response, dest_path, verbose=verbose)
        except Exception as e:
            return False, str(e)

    def _download_gdrive_folder(
        self,
        folder_id: str,
        dest_zip_path: Path,
        temp_base: Path,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Inspects a Google Drive folder. If it contains a .zip archive, downloads that zip archive.
        Otherwise, downloads all loose files into a temporary directory and creates the zip archive.
        """
        try:
            items = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
        except Exception:
            items = None

        if items:
            zip_items = [it for it in items if it.path.lower().endswith(".zip")]
            if zip_items:
                target_zip = zip_items[0]
                if verbose:
                    print(f"      [Folder Archive Found] {target_zip.path}")
                return self._download_gdrive_file_stream(target_zip.id, dest_zip_path, verbose=verbose)

            # Loose files in folder: package them
            if verbose:
                print(f"      [Packaging Folder] Found {len(items)} loose files. Downloading and packaging...")
            temp_dir = temp_base / f"tmp_{folder_id}"
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
            try:
                for it in items:
                    sub_file = temp_dir / it.path
                    sub_file.parent.mkdir(parents=True, exist_ok=True)
                    ok, err = self._download_gdrive_file_stream(it.id, sub_file, verbose=False)
                    if not ok:
                        return False, f"Failed to download folder file {it.path}: {err}"

                with zipfile.ZipFile(dest_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in temp_dir.rglob("*"):
                        if f.is_file():
                            zf.write(f, arcname=f.relative_to(temp_dir).as_posix())

                if dest_zip_path.exists() and dest_zip_path.stat().st_size > 0:
                    return True, None
                return False, "Failed to create zip from folder contents."
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Fallback to single file stream if folder inspection returned no items
        return self._download_gdrive_file_stream(folder_id, dest_zip_path, verbose=verbose)

    def _download_gdrive_submission(
        self,
        gdrive_id: str,
        is_folder: bool,
        dest_file: Path,
        temp_base: Path,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Handles any Google Drive submission URL regardless of whether it was shared as folder or file."""
        # 1. Check if it's a folder (or if URL was explicitly /drive/folders/)
        try:
            items = gdown.download_folder(id=gdrive_id, skip_download=True, quiet=True)
            if items:
                return self._download_gdrive_folder(gdrive_id, dest_file, temp_base, verbose=verbose)
        except Exception:
            pass

        # 2. Try single file stream download
        ok, err = self._download_gdrive_file_stream(gdrive_id, dest_file, verbose=verbose)
        if ok:
            return True, None

        # 3. Fallback to standard gdown download if httpx stream was blocked
        try:
            out = gdown.download(id=gdrive_id, output=str(dest_file), quiet=not verbose)
            if out and dest_file.exists() and dest_file.stat().st_size > 0 and not self._is_html_file(dest_file):
                return True, None
        except Exception as e:
            err = str(e)

        return False, err or "Failed to download from Google Drive."

    def _download_direct_http(
        self,
        url: str,
        dest_path: Path,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Downloads standard direct HTTP/HTTPS files with live progress bar."""
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    return self._stream_response_to_file(response, dest_path, verbose=verbose)
        except Exception as e:
            return False, str(e)

    def _download_item(
        self,
        item: Dict[str, str],
        raw_dir: Path,
        verbose: bool = False,
        skip_existing: bool = False
    ) -> Tuple[str, bool, Optional[str]]:
        team_id = item.get("team_id") or item.get("team_name")
        url = item.get("url") or item.get("drive_url") or item.get("link", "")
        if not team_id or not url:
            return str(team_id), False, "Missing team_id or URL in CSV"

        dest_file = raw_dir / f"{team_id}.zip"

        if skip_existing and dest_file.exists() and dest_file.stat().st_size > 0:
            if not self._is_html_file(dest_file):
                if verbose:
                    print(f"      [SKIPPED] Already exists ({format_bytes(dest_file.stat().st_size)})")
                return team_id, True, None

        try:
            gdrive_id, is_folder = extract_gdrive_id(url)
            if gdrive_id:
                success, error = self._download_gdrive_submission(
                    gdrive_id, is_folder, dest_file, raw_dir, verbose=verbose
                )
            else:
                success, error = self._download_direct_http(url, dest_file, verbose=verbose)

            if success:
                return team_id, True, None
            else:
                if dest_file.exists():
                    dest_file.unlink(missing_ok=True)
                return team_id, False, error or "Download failed"
        except Exception as e:
            if dest_file.exists():
                dest_file.unlink(missing_ok=True)
            return team_id, False, str(e)

    def parse_contestants_csv(self, csv_path: Path) -> List[Dict[str, str]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Contestants CSV not found at: {csv_path}")

        contestants = []
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = {
                    k.strip().lower().replace(" ", "_").replace("-", "_"): (v.strip() if v else "")
                    for k, v in row.items() if k
                }

                team_id = (
                    cleaned_row.get("team_id")
                    or cleaned_row.get("team")
                    or cleaned_row.get("id")
                    or cleaned_row.get("team_name")
                )

                url = (
                    cleaned_row.get("drive_link")
                    or cleaned_row.get("drive_url")
                    or cleaned_row.get("link")
                    or cleaned_row.get("url")
                    or cleaned_row.get("drivelink")
                    or cleaned_row.get("google_drive")
                )

                team_name = (
                    cleaned_row.get("team_name")
                    or cleaned_row.get("name")
                    or cleaned_row.get("team")
                    or team_id
                )

                if team_id and url and url.strip():
                    contestants.append({
                        "team_id": team_id.strip(),
                        "team_name": team_name.strip(),
                        "url": url.strip(),
                        "university": cleaned_row.get("university", ""),
                        "status": cleaned_row.get("status", "")
                    })
        return contestants

    def download_sequential(
        self,
        contestants_csv: Path,
        output_dir: Path,
        verbose: bool = True,
        skip_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Downloads contestant submissions sequentially (one-by-one).
        Avoids Google Drive concurrent rate limits and provides clear real-time progress.
        """
        contestants = self.parse_contestants_csv(contestants_csv)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.status = {
            "is_running": True,
            "total": len(contestants),
            "completed": 0,
            "failed": 0,
            "current_team": "",
            "errors": {}
        }

        if verbose:
            print("=" * 80)
            print("TESA TopGun 2026 - Master Downloader")
            print(f"Total Teams: {len(contestants)} | Mode: Sequential | Verbose: ON")
            print(f"Destination: {output_dir}")
            print("=" * 80)

        results = []
        for idx, item in enumerate(contestants, 1):
            team_id = item["team_id"]
            team_name = item.get("team_name", "")
            url = item.get("url", "")
            self.status["current_team"] = team_id

            if verbose:
                display_name = f"{team_id} ({team_name})" if team_name and team_name != team_id else team_id
                print(f"\n[{idx}/{len(contestants)}] Team: {display_name}")
                print(f"      Link: {url}")

            tid, success, error = self._download_item(
                item,
                output_dir,
                verbose=verbose,
                skip_existing=skip_existing
            )

            if success:
                self.status["completed"] += 1
                dest_file = output_dir / f"{tid}.zip"
                size_str = format_bytes(dest_file.stat().st_size) if dest_file.exists() else "unknown"
                if verbose:
                    print(f"      [SUCCESS] Saved {tid}.zip ({size_str})")
                results.append({"team_id": tid, "status": "SUCCESS", "size": size_str})
            else:
                self.status["failed"] += 1
                self.status["errors"][tid] = error
                if verbose:
                    print(f"      [FAILED] {tid}: {error}")
                results.append({"team_id": tid, "status": "FAILED", "error": error})

        self.status["is_running"] = False

        if verbose:
            print("\n" + "=" * 80)
            print("DOWNLOAD SUMMARY")
            print(f"  Total Teams:     {self.status['total']}")
            print(f"  Successful:      {self.status['completed']}")
            print(f"  Failed:          {self.status['failed']}")
            print("=" * 80)
            if self.status["errors"]:
                print("\nFailed Submissions:")
                for tid, err in self.status["errors"].items():
                    print(f"  - {tid}: {err}")
                print("=" * 80)

        return {
            "total": self.status["total"],
            "completed": self.status["completed"],
            "failed": self.status["failed"],
            "results": results,
            "errors": self.status["errors"]
        }

    async def download_all(
        self,
        contestants_csv: Path,
        output_dir: Path,
        parallel: bool = False,
        verbose: bool = False,
        skip_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point for downloading. If parallel=False, runs sequential.
        If parallel=True, runs concurrently with ThreadPoolExecutor.
        """
        if not parallel:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self.download_sequential,
                contestants_csv,
                output_dir,
                verbose,
                skip_existing
            )

        contestants = self.parse_contestants_csv(contestants_csv)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.status = {
            "is_running": True,
            "total": len(contestants),
            "completed": 0,
            "failed": 0,
            "current_team": "",
            "errors": {}
        }

        loop = asyncio.get_running_loop()
        results = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = [
                loop.run_in_executor(pool, self._download_item, item, output_dir, verbose, skip_existing)
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
