"""
CLI entry point to run the master server or pipeline tasks directly.

Usage:
    python run_master.py serve                            # Start FastAPI server
    python run_master.py download --day 1                # Sequential, verbose download
    python run_master.py download --day 1 --skip-existing # Skip already downloaded files
    python run_master.py download --day 1 --parallel     # Concurrent download
    python run_master.py prepare --day 1                 # Batch unpack, validate, and prepare queue
"""
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
import uvicorn

# Ensure master directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def download_cli(day: int, parallel: bool = False, verbose: bool = True, skip_existing: bool = False):
    csv_path = settings.get_contestants_csv(day)
    raw_dir = settings.get_raw_zips_dir(day)

    downloader = HighSpeedDownloader(
        max_concurrent=settings.DOWNLOAD_MAX_CONCURRENT,
        timeout_seconds=settings.DOWNLOAD_TIMEOUT_SECONDS
    )

    if not parallel:
        downloader.download_sequential(csv_path, raw_dir, verbose=verbose, skip_existing=skip_existing)
    else:
        print(f"[Day {day}] Reading contestants from {csv_path} ...")
        print(f"[Day {day}] Downloading all archives concurrently into {raw_dir} ...")
        result = asyncio.run(
            downloader.download_all(
                csv_path,
                raw_dir,
                parallel=True,
                verbose=verbose,
                skip_existing=skip_existing
            )
        )
        print(f"[Day {day}] Download Complete! Total: {result['total']}, Success: {result['completed']}, Failed: {result['failed']}")
        if result["errors"]:
            print("Failed details:", result["errors"])


def prepare_cli(day: int, team_id: Optional[str] = None):
    # Try communicating with running master server first
    import httpx
    if team_id:
        endpoint_url = f"http://{settings.MASTER_HOST}:{settings.MASTER_PORT}/api/v1/admin/day/{day}/prepare/{team_id}"
    else:
        endpoint_url = f"http://{settings.MASTER_HOST}:{settings.MASTER_PORT}/api/v1/admin/day/{day}/prepare-queue"

    try:
        resp = httpx.post(endpoint_url, timeout=60.0)
        if resp.status_code == 200:
            data = resp.json()
            if team_id:
                print(f"[Day {day}] Success! {data['message']}")
                print(f"  - Job ID:         {data['job_id']}")
                print(f"  - Team Name:      {data['team_name']}")
                print(f"  - Bundle:         {data['bundle_path']}")
                print(f"  - Weights Count:  {data.get('weights_count')} files ({data.get('weights_dir')})")
                print(f"  - Queue Position: #{data['queue_position']} (Total in Queue: {data.get('total_queued_for_day')})")
            else:
                print(f"[Day {day}] Success! {data['message']}")
                rep = data.get("package_report", {})
                print(f"  - Valid Bundles:  {rep.get('valid')}")
                print(f"  - Invalid:        {rep.get('invalid')}")
                print(f"  - Enqueued Jobs:  {data.get('enqueued_jobs')}")
            return
        elif team_id:
            print(f"[Master Error] HTTP {resp.status_code}: {resp.text}")
            return
    except Exception:
        pass

    # If server is not running, run directly
    if team_id:
        from app.routers.admin import prepare_single_team_and_enqueue
        try:
            data = prepare_single_team_and_enqueue(day=day, team_id=team_id)
            print(f"[Day {day}] Success! {data['message']}")
            print(f"  - Job ID:         {data['job_id']}")
            print(f"  - Team Name:      {data['team_name']}")
            print(f"  - Bundle:         {data['bundle_path']}")
            print(f"  - Weights Count:  {data.get('weights_count')} files ({data.get('weights_dir')})")
            print(f"  - Queue Position: #{data['queue_position']}")
            print("\nNote: Master server was not running. Start it with 'python run_master.py serve' to serve slaves.")
        except Exception as e:
            print(f"[Prepare Failed] {e}")
        return

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

    enqueued_count = queue_manager.load_jobs_from_bundles(day, bundles_dir, team_names_map)
    print(f"[Day {day}] Enqueued {enqueued_count} jobs from bundles.")
    print("\nNote: Master server was not running. Start it with 'python run_master.py serve' to serve slaves.")


def retry_failed_cli(day: int):
    leaderboard_csv = settings.get_leaderboard_csv(day)
    bundles_dir = settings.get_bundles_dir(day)

    if not leaderboard_csv.exists():
        print(f"[Error] final_leaderboard.csv not found at {leaderboard_csv}")
        return

    # Try communicating with running master server first
    import httpx
    endpoint_url = f"http://{settings.MASTER_HOST}:{settings.MASTER_PORT}/api/v1/admin/day/{day}/retry-failed"
    try:
        resp = httpx.post(endpoint_url, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            print(f"[Day {day}] Success! Requeued {data['requeued_count']} failed jobs in running Master Server.")
            print(f"  - Preserved Completed Jobs: {data['completed_preserved_count']}")
            print(f"  - Requeued to Retry: {data['requeued_count']}")
            if data["missing_bundle_count"] > 0:
                print(f"  - Warning: {data['missing_bundle_count']} jobs had missing bundle archives.")
            for rj in data["requeued_jobs"]:
                print(f"    * {rj['job_id']} ({rj['team_name']})")
            return
    except Exception:
        pass

    # If server is not running, run directly against queue_manager
    report = queue_manager.requeue_failed_from_leaderboard(day, leaderboard_csv, bundles_dir)
    print(f"[Day {day}] Requeued {report['requeued_count']} failed jobs from {leaderboard_csv.name}.")
    print(f"  - Preserved Completed Jobs: {report['completed_preserved_count']}")
    print(f"  - Requeued to Retry: {report['requeued_count']}")
    if report["missing_bundle_count"] > 0:
        print(f"  - Warning: {report['missing_bundle_count']} jobs had missing bundle archives.")
    for rj in report["requeued_jobs"]:
        print(f"    * {rj['job_id']} ({rj['team_name']})")
    print("\nNote: Master server was not running. Start it with 'python run_master.py serve' to serve slaves.")


def enqueue_cli(team_id: str, day: int, force: bool = True):
    from app.routers.admin import _resolve_and_enqueue_team
    import httpx

    # Try communicating with running master server first
    endpoint_url = f"http://{settings.MASTER_HOST}:{settings.MASTER_PORT}/api/v1/admin/day/{day}/enqueue/{team_id}?force={str(force).lower()}"
    try:
        resp = httpx.post(endpoint_url, timeout=15.0)
        if resp.status_code == 200:
            data = resp.json()
            print(f"[Day {day}] Success! Enqueued '{data['team_id']}' into running Master Server.")
            print(f"  - Job ID:         {data['job_id']}")
            print(f"  - Team Name:      {data['team_name']}")
            print(f"  - Status:         {data['status']}")
            print(f"  - Queue Position: #{data['queue_position']} (Total in Queue: {data['total_queued_for_day']})")
            return
        else:
            print(f"[Master Error] HTTP {resp.status_code}: {resp.text}")
            return
    except Exception:
        pass

    # If server is not running, run directly against queue_manager
    try:
        data = _resolve_and_enqueue_team(day=day, team_id=team_id, force=force)
        print(f"[Day {day}] Success! Enqueued '{data['team_id']}'.")
        print(f"  - Job ID:         {data['job_id']}")
        print(f"  - Team Name:      {data['team_name']}")
        print(f"  - Status:         {data['status']}")
        print(f"  - Bundle:         {data['bundle_path']}")
        print(f"  - Queue Position: #{data['queue_position']}")
        print("\nNote: Master server was not running. Start it with 'python run_master.py serve' to serve slaves.")
    except Exception as e:
        print(f"[Enqueue Failed] {e}")


def evaluate_predictions_cli(day: int, force: bool = False, team_id: Optional[str] = None):
    import httpx

    params = f"?force={str(force).lower()}"
    if team_id:
        params += f"&team_id={team_id}"
    endpoint_url = f"http://{settings.MASTER_HOST}:{settings.MASTER_PORT}/api/v1/admin/day/{day}/evaluate-predictions{params}"
    try:
        resp = httpx.post(endpoint_url, timeout=120.0)
        if resp.status_code == 200:
            data = resp.json()
            print(f"[Day {day}] {data['message']}")
            for ev in data.get("evaluated", []):
                print(f"  + [EVALUATED] Rank #{ev.get('rank', '-')}: {ev['team_name']} ({ev['job_id']}) -> Composite Score: {ev['composite_score']}")
            for sk in data.get("skipped", []):
                print(f"  - [SKIPPED] {sk['file']} ({sk['team_id']}): {sk['reason']}")
            for er in data.get("errors", []):
                print(f"  ! [ERROR] {er['file']}: {er['error']}")
            return
    except Exception:
        pass

    # If server is not running, run directly
    from app.routers.admin import evaluate_predictions_for_day
    try:
        data = evaluate_predictions_for_day(day=day, force=force, target_team_id=team_id)
        print(f"[Day {day}] {data['message']}")
        for ev in data.get("evaluated", []):
            print(f"  + [EVALUATED] Rank #{ev.get('rank', '-')}: {ev['team_name']} ({ev['job_id']}) -> Composite Score: {ev['composite_score']}")
        for sk in data.get("skipped", []):
            print(f"  - [SKIPPED] {sk['file']} ({sk['team_id']}): {sk['reason']}")
        for er in data.get("errors", []):
            print(f"  ! [ERROR] {er['file']}: {er['error']}")
    except Exception as e:
        print(f"[Evaluation Failed] {e}")


def main():
    parser = argparse.ArgumentParser(description="TESA Master Orchestrator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Server command
    subparsers.add_parser("serve", help="Start the FastAPI server")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download contestant submissions")
    dl_parser.add_argument(
        "--day",
        type=int,
        choices=[1, 2, 3],
        default=settings.ACTIVE_DAY,
        help="Competition day (default: 1)"
    )
    dl_parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run concurrent downloads instead of sequential (default: False)"
    )
    dl_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress detailed download progress and bars"
    )
    dl_parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip files that already exist on disk"
    )

    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Unpack, validate and prepare bundles")
    prep_parser.add_argument(
        "team_id",
        nargs="?",
        default=None,
        help="Optional team_id to prepare only 1 team (e.g. TGR-67)"
    )
    prep_parser.add_argument(
        "--day",
        type=int,
        choices=[1, 2, 3],
        default=settings.ACTIVE_DAY,
        help="Competition day"
    )
    prep_parser.add_argument(
        "--team-id",
        dest="team_id_opt",
        type=str,
        default=None,
        help="Optional flag syntax for team_id (e.g. --team-id TGR-67)"
    )

    # Retry-failed command
    retry_parser = subparsers.add_parser("retry-failed", help="Requeue failed jobs from final_leaderboard.csv")
    retry_parser.add_argument(
        "--day",
        type=int,
        choices=[1, 2, 3],
        default=settings.ACTIVE_DAY,
        help="Competition day (default: 1)"
    )

    # Enqueue single team command
    enq_parser = subparsers.add_parser("enqueue", help="Manually enqueue a specific team (e.g. TGR-01)")
    enq_parser.add_argument(
        "team_id",
        type=str,
        help="Team ID from contestants.csv (e.g. TGR-01, TGR-23)"
    )
    enq_parser.add_argument(
        "--day",
        type=int,
        choices=[1, 2, 3],
        default=settings.ACTIVE_DAY,
        help="Competition day (default: active day)"
    )
    enq_parser.add_argument(
        "--no-force",
        action="store_true",
        default=False,
        help="Do not reset job if already completed/queued"
    )

    # Evaluate-predictions command
    eval_pred_parser = subparsers.add_parser(
        "evaluate-predictions",
        help="Evaluate manual prediction files in results/day_{day}/predictions"
    )
    eval_pred_parser.add_argument(
        "--day",
        type=int,
        choices=[1, 2, 3],
        default=settings.ACTIVE_DAY,
        help="Competition day (default: active day)"
    )
    eval_pred_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-evaluate even if team already has a completed score in final_leaderboard.csv"
    )
    eval_pred_parser.add_argument(
        "--team-id",
        type=str,
        default=None,
        help="Optional team_id to evaluate only a specific prediction file"
    )

    args = parser.parse_args()

    if args.command == "serve":
        serve()
    elif args.command == "download":
        download_cli(
            day=args.day,
            parallel=args.parallel,
            verbose=not args.quiet,
            skip_existing=args.skip_existing
        )
    elif args.command == "prepare":
        team_id_val = args.team_id_opt or args.team_id
        prepare_cli(day=args.day, team_id=team_id_val)
    elif args.command == "retry-failed":
        retry_failed_cli(args.day)
    elif args.command == "enqueue":
        enqueue_cli(team_id=args.team_id, day=args.day, force=not args.no_force)
    elif args.command == "evaluate-predictions":
        evaluate_predictions_cli(day=args.day, force=args.force, team_id=args.team_id)


if __name__ == "__main__":
    main()
