"""
CLI entry point to run the master server or pipeline tasks directly.

Usage:
    python run_master.py serve                      # Start FastAPI server
    python run_master.py download --day 1          # Bulk download raw zips for Day 1
    python run_master.py prepare --day 1           # Batch unpack, validate, and prepare queue
"""
import argparse
import asyncio
import sys
from pathlib import Path
import uvicorn

from app.config import settings
from app.services.downloader import HighSpeedDownloader
from app.services.packager import SubmissionPackager
from app.services.queue_manager import queue_manager


def serve():
    print(f"Starting TESA Master Server on http://{settings.MASTER_HOST}:{settings.MASTER_PORT} ...")
    uvicorn.run(
        "app.main:app",
        host=settings.MASTER_HOST,
        port=settings.MASTER_PORT,
        reload=False
    )


def download_cli(day: int):
    csv_path = settings.get_contestants_csv(day)
    raw_dir = settings.get_raw_zips_dir(day)
    print(f"[Day {day}] Reading contestants from {csv_path} ...")
    print(f"[Day {day}] Downloading all archives concurrently into {raw_dir} ...")

    downloader = HighSpeedDownloader(
        max_concurrent=settings.DOWNLOAD_MAX_CONCURRENT,
        timeout_seconds=settings.DOWNLOAD_TIMEOUT_SECONDS
    )
    result = asyncio.run(downloader.download_all(csv_path, raw_dir))
    print(f"[Day {day}] Bulk Download Complete! Total: {result['total']}, Success: {result['completed']}, Failed: {result['failed']}")
    if result["errors"]:
        print("Failed details:", result["errors"])


def prepare_cli(day: int):
    raw_dir = settings.get_raw_zips_dir(day)
    extracted_dir = settings.get_extracted_dir(day)
    bundles_dir = settings.get_bundles_dir(day)

    print(f"[Day {day}] Batch processing raw zips in {raw_dir} ...")
    packager = SubmissionPackager()
    report = packager.process_all_for_day(raw_dir, extracted_dir, bundles_dir)
    print(f"[Day {day}] Validation Complete! Valid: {report['valid']}, Invalid: {report['invalid']}")
    for d in report["details"]:
        status_str = "VALID" if d["is_valid"] else f"INVALID ({d.get('error')})"
        print(f"  - {d['team_id']}: {status_str}")


def main():
    parser = argparse.ArgumentParser(description="TESA Master Orchestrator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Server command
    subparsers.add_parser("serve", help="Start the FastAPI server")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Run bulk parallel download")
    dl_parser.add_argument("--day", type=int, choices=[1, 2, 3], default=settings.ACTIVE_DAY, help="Competition day")

    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Unpack, validate and prepare bundles")
    prep_parser.add_argument("--day", type=int, choices=[1, 2, 3], default=settings.ACTIVE_DAY, help="Competition day")

    args = parser.parse_args()

    if args.command == "serve":
        serve()
    elif args.command == "download":
        download_cli(args.day)
    elif args.command == "prepare":
        prepare_cli(args.day)


if __name__ == "__main__":
    main()
