"""Scheduled cleanup job for Modal Volume retention management."""

import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any
import modal

from modal_runtime.app import app, data_volume, VOLUME_CONFIG, api_secret


def safe_api_call(
    url: str, json_data: dict, headers: dict, operation_name: str
) -> bool:
    """Helper function to make API calls with error handling."""
    try:
        import requests

        response = requests.post(url, json=json_data, headers=headers, timeout=10)
        if response.status_code == 200:
            print(f"  ✓ {operation_name}")
            return True
        else:
            print(f"  ⚠ {operation_name} failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ⚠ {operation_name} error: {e}")
        return False


def get_all_run_directories(base_path: str = "/data/runs") -> List[Dict[str, Any]]:
    """Scan Modal Volume for all run directories."""
    base_dir = Path(base_path)

    if not base_dir.exists():
        return []

    runs = []

    # Iterate through user_id directories
    for user_dir in base_dir.iterdir():
        if not user_dir.is_dir():
            continue

        user_id = user_dir.name

        # Iterate through run_id directories
        for run_dir in user_dir.iterdir():
            if not run_dir.is_dir():
                continue

            run_id = run_dir.name

            # Get last modification time
            try:
                mtime = run_dir.stat().st_mtime
                last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc)

                runs.append(
                    {
                        "user_id": user_id,
                        "run_id": run_id,
                        "path": str(run_dir),
                        "last_modified": last_modified,
                    }
                )
            except Exception as e:
                print(f"Warning: Could not read {run_dir}: {e}")
                continue

    return runs


def delete_run_directory(run_path: str) -> int:
    """Delete a run directory and return bytes freed."""
    run_dir = Path(run_path)

    if not run_dir.exists():
        return 0

    # Calculate size before deletion
    total_size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())

    # Delete directory
    shutil.rmtree(run_dir)

    return total_size


@app.function(
    image=modal.Image.debian_slim().pip_install("requests"),
    volumes=VOLUME_CONFIG,
    secrets=[api_secret],  # For SIGNAL_INTERNAL_SECRET and SIGNAL_API_URL
    schedule=modal.Period(days=1),  # Run daily
    timeout=3600,  # 1 hour timeout
)
def cleanup_old_runs(
    retention_days: int = 7,
    dry_run: bool = False,
    api_url: str = None,
):
    """Scheduled cleanup job to remove old runs from Modal Volume.

    Runs daily to clean up completed training runs older than retention_days.
    Artifacts remain in S3 for long-term storage."""
    print("=" * 80)
    print("Starting Modal Volume cleanup job")
    print(f"Retention policy: {retention_days} days")
    print(f"Dry run: {dry_run}")
    print("=" * 80)

    # Get API URL from environment if not provided
    if api_url is None:
        api_url = os.environ.get("SIGNAL_API_URL")

    if api_url:
        print(f"API URL configured: {api_url}")
    else:
        print("No API URL configured - database will not be updated")

    # Calculate cutoff date
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
    print(f"Cutoff date: {cutoff_date.isoformat()}")

    # Scan for all runs
    print("\nScanning Modal Volume for runs...")
    all_runs = get_all_run_directories()
    print(f"Found {len(all_runs)} total runs")

    # Filter runs older than cutoff
    old_runs = [run for run in all_runs if run["last_modified"] < cutoff_date]

    print(f"Found {len(old_runs)} runs older than {retention_days} days")

    if not old_runs:
        print("No runs to clean up")
        return {
            "status": "success",
            "scanned": len(all_runs),
            "deleted": 0,
            "bytes_freed": 0,
        }

    # Delete old runs
    deleted_count = 0
    total_bytes_freed = 0
    errors = []

    for run in old_runs:
        user_id = run["user_id"]
        run_id = run["run_id"]
        run_path = run["path"]
        age_days = (datetime.now(timezone.utc) - run["last_modified"]).days

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Cleaning up run:")
        print(f"  User: {user_id}")
        print(f"  Run: {run_id}")
        print(f"  Age: {age_days} days")
        print(f"  Path: {run_path}")

        if dry_run:
            # Just log, don't delete
            deleted_count += 1
            continue

        try:
            # Delete directory
            bytes_freed = delete_run_directory(run_path)
            total_bytes_freed += bytes_freed
            deleted_count += 1

            print(f"  ✓ Deleted ({bytes_freed / (1024**3):.2f} GB freed)")

            # Update database if API URL is provided
            if api_url:
                # Get internal service key from environment
                internal_key = os.environ.get("SIGNAL_INTERNAL_SECRET")
                headers = {}
                if internal_key:
                    headers["X-Internal-Key"] = internal_key

                safe_api_call(
                    url=f"{api_url}/internal/mark-volume-cleaned",
                    json_data={
                        "run_id": run_id,
                        "user_id": user_id,
                        "bytes_freed": bytes_freed,
                    },
                    headers=headers,
                    operation_name="Database update",
                )

        except Exception as e:
            error_msg = f"Failed to delete {run_path}: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)

    # Commit volume changes
    if not dry_run and deleted_count > 0:
        print("\nCommitting volume changes...")
        data_volume.commit()
        print("✓ Volume committed")

    # Summary
    print("\n" + "=" * 80)
    print("Cleanup Summary")
    print("=" * 80)
    print(f"Scanned: {len(all_runs)} runs")
    print(f"Deleted: {deleted_count} runs")
    print(f"Space freed: {total_bytes_freed / (1024**3):.2f} GB")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

    return {
        "status": "success" if not errors else "partial",
        "scanned": len(all_runs),
        "deleted": deleted_count,
        "bytes_freed": total_bytes_freed,
        "errors": errors,
        "dry_run": dry_run,
    }


@app.function(
    image=modal.Image.debian_slim().pip_install("requests", "supabase"),
    secrets=[api_secret],
    schedule=modal.Period(minutes=5),  # Run every 5 minutes
    timeout=300,  # 5 minute timeout
)
def cleanup_stale_runs(
    stale_minutes: int = 30,
    dry_run: bool = False,
):
    """Mark runs without recent activity as failed.

    Prevents stuck runs when clients crash or lose connection.
    Uses the database's updated_at column which auto-updates on any API activity."""
    from supabase import create_client

    print("=" * 80)
    print("Starting stale run cleanup job")
    print(f"Stale threshold: {stale_minutes} minutes")
    print(f"Dry run: {dry_run}")
    print("=" * 80)

    # Get Supabase credentials from environment
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Missing SUPABASE_URL or SUPABASE_KEY")
        return {"status": "error", "message": "Missing Supabase credentials"}

    try:
        # Connect to Supabase
        supabase = create_client(supabase_url, supabase_key)

        # Use database function to mark stale runs as failed
        if dry_run:
            # Query what would be marked
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)
            result = (
                supabase.table("runs")
                .select("id, user_id, updated_at, status")
                .in_("status", ["initialized", "running"])
                .lt("updated_at", cutoff_time.isoformat())
                .execute()
            )

            stale_runs = result.data or []
            print(f"\n[DRY RUN] Would mark {len(stale_runs)} runs as failed:")
            for run in stale_runs:
                updated_at = datetime.fromisoformat(run["updated_at"])
                minutes_stale = (
                    datetime.now(timezone.utc) - updated_at
                ).total_seconds() / 60
                print(f"  - {run['id']} (inactive for {minutes_stale:.1f} minutes)")

            return {
                "status": "success",
                "dry_run": True,
                "stale_runs": len(stale_runs),
                "marked_failed": 0,
            }
        else:
            # Actually mark runs as failed using database function
            result = supabase.rpc(
                "mark_stale_runs_as_failed", {"p_stale_minutes": stale_minutes}
            ).execute()

            marked_runs = result.data or []
            print(f"\n✓ Marked {len(marked_runs)} runs as failed:")

            # Now charge remaining costs for each failed run
            api_url = os.environ.get("SIGNAL_API_URL")
            internal_key = os.environ.get("SIGNAL_INTERNAL_SECRET")

            charged_count = 0
            charge_errors = []

            for run in marked_runs:
                run_id = run["run_id"]
                user_id = run["user_id"]
                minutes_stale = run["minutes_stale"]

                print(f"  - {run_id} (inactive for {minutes_stale:.1f} minutes)")

                # Try to charge remaining cost via API
                if api_url and internal_key:
                    success = safe_api_call(
                        url=f"{api_url}/internal/charge-final-cost",
                        json_data={
                            "run_id": run_id,
                            "user_id": user_id,
                        },
                        headers={"X-Internal-Key": internal_key},
                        operation_name="Charge final cost",
                    )
                    if success:
                        charged_count += 1
                    else:
                        charge_errors.append(run_id)

            print("\n" + "=" * 80)
            print(f"Marked {len(marked_runs)} runs as failed")
            print(f"Charged {charged_count} runs successfully")
            if charge_errors:
                print(f"Failed to charge {len(charge_errors)} runs: {charge_errors}")
            print("=" * 80)

            return {
                "status": "success",
                "dry_run": False,
                "stale_runs": len(marked_runs),
                "marked_failed": len(marked_runs),
                "charged": charged_count,
                "charge_errors": charge_errors,
            }

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def run_cleanup_now(
    retention_days: int = 7,
    dry_run: bool = True,
):
    """Manual trigger for cleanup job (for testing).

    Usage:
        modal run modal_runtime.cleanup --retention-days 7 --dry-run true
    """
    result = cleanup_old_runs.remote(
        retention_days=retention_days,
        dry_run=dry_run,
    )

    print("\n" + "=" * 80)
    print("Cleanup Result:")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Scanned: {result['scanned']}")
    print(f"Deleted: {result['deleted']}")
    print(f"Bytes freed: {result['bytes_freed'] / (1024**3):.2f} GB")

    if result.get("errors"):
        print(f"Errors: {len(result['errors'])}")
