"""Scheduled cleanup job for Modal Volume retention management."""
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any
import modal

from modal_runtime.app import app, data_volume, VOLUME_CONFIG, api_secret


def get_all_run_directories(base_path: str = "/data/runs") -> List[Dict[str, Any]]:
    """Scan Modal Volume for all run directories.
    
    Args:
        base_path: Base path for runs on Modal Volume
        
    Returns:
        List of dicts with run info (user_id, run_id, path, last_modified)
    """
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
                
                runs.append({
                    "user_id": user_id,
                    "run_id": run_id,
                    "path": str(run_dir),
                    "last_modified": last_modified,
                })
            except Exception as e:
                print(f"Warning: Could not read {run_dir}: {e}")
                continue
    
    return runs


def delete_run_directory(run_path: str) -> int:
    """Delete a run directory and return bytes freed.
    
    Args:
        run_path: Path to run directory
        
    Returns:
        Number of bytes freed
    """
    run_dir = Path(run_path)
    
    if not run_dir.exists():
        return 0
    
    # Calculate size before deletion
    total_size = sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())
    
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
    Artifacts remain in S3 for long-term storage.
    
    Args:
        retention_days: Number of days to retain runs on Modal Volume (default: 7)
        dry_run: If True, only log what would be deleted without actually deleting
        api_url: Optional API URL to update database (from env if not provided)
    """
    print("=" * 80)
    print(f"Starting Modal Volume cleanup job")
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
    old_runs = [
        run for run in all_runs
        if run["last_modified"] < cutoff_date
    ]
    
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
                try:
                    import requests
                    
                    # Get internal service key from environment
                    internal_key = os.environ.get("SIGNAL_INTERNAL_SECRET")
                    headers = {}
                    if internal_key:
                        headers["X-Internal-Key"] = internal_key
                    
                    response = requests.post(
                        f"{api_url}/internal/mark-volume-cleaned",
                        json={
                            "run_id": run_id,
                            "user_id": user_id,
                            "bytes_freed": bytes_freed,
                        },
                        headers=headers,
                        timeout=10,
                    )
                    
                    if response.status_code == 200:
                        print(f"  ✓ Database updated")
                    else:
                        print(f"  ⚠ Database update failed: {response.status_code}")
                        
                except Exception as e:
                    print(f"  ⚠ Database update error: {e}")
                    # Continue anyway - cleanup succeeded
            
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

