"""Run tracking and metadata storage via Supabase."""

import uuid
import os
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from supabase import Client

from api.supabase_client import get_supabase

logger = logging.getLogger(__name__)


class RunRegistry:
    """Registry for tracking training runs in Supabase PostgreSQL."""

    # Resource limits per user
    MAX_CONCURRENT_RUNS_PER_USER = int(os.getenv("MAX_CONCURRENT_RUNS_PER_USER", "5"))

    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the run registry."""
        self.supabase = supabase_client or get_supabase()

    def create_run(self, user_id: str, base_model: str, config: Dict[str, Any]) -> str:
        """Create a new training run with race-condition-safe concurrent check.

        Uses atomic database function to check limit and insert in single transaction."""
        run_id = f"run_{uuid.uuid4().hex[:16]}"

        try:
            # Use atomic database function to prevent race conditions
            # This checks the concurrent run count and inserts in a single transaction
            result = self.supabase.rpc(
                "create_run_if_allowed",
                {
                    "p_run_id": run_id,
                    "p_user_id": user_id,
                    "p_base_model": base_model,
                    "p_config": config,
                    "p_max_concurrent": self.MAX_CONCURRENT_RUNS_PER_USER,
                },
            ).execute()

            # Check if run was created successfully
            if not result.data:
                raise Exception(
                    f"Maximum {self.MAX_CONCURRENT_RUNS_PER_USER} concurrent runs reached. "
                    f"Please wait for existing runs to complete or cancel them."
                )

            logger.info(f"Created run {run_id} for user {user_id}")
            return run_id

        except Exception as e:
            # If the error is about concurrent runs, re-raise with clean message
            if "concurrent" in str(e).lower() or "maximum" in str(e).lower():
                raise
            # Otherwise log and re-raise
            logger.error(f"Failed to create run for user {user_id}: {e}")
            raise Exception(f"Failed to create training run: {str(e)}")

    def _normalize_run(self, run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ensure run records always include expected optional fields."""

        if not run:
            return run

        # Ensure config exists so downstream callers can safely update it
        config = run.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        run["config"] = config

        # Populate GPU fields with sensible defaults
        current_gpu = run.get("current_gpu") or config.get("gpu_config")
        run["current_gpu"] = current_gpu

        target_gpu = run.get("target_gpu")
        run["target_gpu"] = target_gpu

        # Ensure migration history is always a list
        history = run.get("migration_history") or []
        if not isinstance(history, list):
            history = []
        run["migration_history"] = history

        # Provide placeholder for status messages
        run.setdefault("status_message", None)

        # Some callers expect run_id key in addition to id
        run.setdefault("run_id", run.get("id"))

        return run

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run metadata."""
        result = (
            self.supabase.table("runs").select("*").eq("id", run_id).single().execute()
        )
        return self._normalize_run(result.data)

    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        current_step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        *,
        config_updates: Optional[Dict[str, Any]] = None,
        status_message: Optional[str] = None,
        current_gpu: Optional[str] = None,
        target_gpu: Optional[str] = None,
        migration_event: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update run metadata and optionally add metrics."""
        try:
            # Update run metadata
            update_data = {}
            if status is not None:
                update_data["status"] = status
                from datetime import datetime, timezone

                # Track when run actually starts
                if status == "running":
                    update_data["started_at"] = datetime.now(timezone.utc).isoformat()
                # Track when run completes
                elif status in ("completed", "failed", "cancelled"):
                    update_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            if current_step is not None:
                update_data["current_step"] = current_step

            if status_message is not None:
                update_data["status_message"] = status_message

            run_snapshot: Optional[Dict[str, Any]] = None

            if config_updates:
                run_snapshot = run_snapshot or self.get_run(run_id)
                if not run_snapshot:
                    logger.error(
                        "Cannot update config for run %s because it was not found",
                        run_id,
                    )
                    return False

                current_config = deepcopy(run_snapshot.get("config") or {})
                new_config = deepcopy(current_config)
                for key, value in config_updates.items():
                    if value is None:
                        new_config.pop(key, None)
                    else:
                        new_config[key] = value

                update_data["config"] = new_config

                old_gpu = current_config.get("gpu_config")
                new_gpu = new_config.get("gpu_config")

                if old_gpu != new_gpu:
                    event = migration_event
                    if event is None:
                        event = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "from_gpu": old_gpu,
                            "to_gpu": new_gpu,
                            "reason": status_message or "config_update",
                        }

                    history = list(run_snapshot.get("migration_history") or [])
                    history.append(event)
                    update_data["migration_history"] = history

                    if status == "migrating":
                        update_data.setdefault(
                            "current_gpu",
                            run_snapshot.get("current_gpu") or old_gpu or new_gpu,
                        )
                        update_data.setdefault("target_gpu", new_gpu)
                    else:
                        update_data["current_gpu"] = new_gpu
                        update_data.setdefault("target_gpu", None)
                elif "gpu_config" in config_updates and new_gpu is not None:
                    update_data.setdefault(
                        "current_gpu",
                        run_snapshot.get("current_gpu") or new_gpu,
                    )

            if current_gpu is not None:
                update_data["current_gpu"] = current_gpu

            if target_gpu is not None:
                update_data["target_gpu"] = target_gpu

            if status is not None and status != "migrating":
                update_data.setdefault("target_gpu", None)

            if update_data:
                self.supabase.table("runs").update(update_data).eq(
                    "id", run_id
                ).execute()

            # Add metrics if provided
            if metrics is not None:
                # Get current step from run if not provided
                if current_step is None:
                    run_snapshot = run_snapshot or self.get_run(run_id)
                    current_step = run_snapshot["current_step"] if run_snapshot else 0

                self.supabase.table("run_metrics").insert(
                    {
                        "run_id": run_id,
                        "step": current_step,
                        "loss": metrics.get("loss"),
                        "grad_norm": metrics.get("grad_norm"),
                        "learning_rate": metrics.get("learning_rate"),
                        "gpu_utilization": metrics.get("gpu_utilization"),
                        "throughput": metrics.get("throughput"),
                        "cost_so_far": metrics.get("cost_so_far"),
                        "metrics_data": metrics,
                    }
                ).execute()

            return True
        except Exception as e:
            logger.error(f"Error updating run {run_id}: {e}")
            return False

    def list_runs(self, user_id: str) -> List[Dict[str, Any]]:
        """List all runs for a user."""
        try:
            result = (
                self.supabase.table("runs")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )

            runs = result.data or []
            return [self._normalize_run(run) for run in runs]
        except Exception as e:
            logger.error(f"Error listing runs for user {user_id}: {e}")
            return []

    def delete_run(self, run_id: str, user_id: str) -> bool:
        """Delete a run and all associated metrics."""
        try:
            # Delete run (metrics cascade delete automatically)
            result = (
                self.supabase.table("runs")
                .delete()
                .eq("id", run_id)
                .eq("user_id", user_id)
                .execute()
            )

            return len(result.data) > 0 if result.data else False
        except Exception as e:
            logger.error(f"Error deleting run {run_id}: {e}")
            return False

    def get_metrics(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get all metrics for a run."""
        result = (
            self.supabase.table("run_metrics")
            .select("*")
            .eq("run_id", run_id)
            .order("step", desc=False)
            .execute()
        )

        return result.data

    def get_latest_metric(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent metric for a run."""
        result = (
            self.supabase.table("run_metrics")
            .select("*")
            .eq("run_id", run_id)
            .order("step", desc=True)
            .limit(1)
            .execute()
        )

        if result.data and len(result.data) > 0:
            return result.data[0]
        return None

    def get_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of run with latest metrics."""
        run = self.get_run(run_id)
        if not run:
            return None

        latest_metric = self.get_latest_metric(run_id)

        return {"run": run, "latest_metric": latest_metric}

    # Artifact tracking (S3 Storage)

    def record_artifact(
        self,
        run_id: str,
        step: int,
        mode: str,
        s3_uri: str,
        manifest: Dict[str, Any],
        file_size_bytes: Optional[int] = None,
    ) -> bool:
        """Record a saved artifact in the artifacts table."""
        try:
            # Insert or update artifact record (UPSERT on run_id, step, mode)
            self.supabase.table("artifacts").upsert(
                {
                    "run_id": run_id,
                    "step": step,
                    "mode": mode,
                    "s3_uri": s3_uri,
                    "manifest": manifest,
                    "file_size_bytes": file_size_bytes,
                }
            ).execute()

            logger.info(f"Recorded artifact for run {run_id}, step {step}, mode {mode}")
            return True
        except Exception as e:
            logger.error(f"Error recording artifact for run {run_id}: {e}")
            return False

    def get_run_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all artifacts for a run."""
        result = (
            self.supabase.table("artifacts")
            .select("*")
            .eq("run_id", run_id)
            .order("step", desc=True)
            .execute()
        )

        return result.data or []

    def update_artifact_download(
        self,
        run_id: str,
        step: int,
        mode: str,
    ) -> bool:
        """Track artifact download by incrementing download_count."""
        try:
            # Get current artifact
            result = (
                self.supabase.table("artifacts")
                .select("download_count")
                .eq("run_id", run_id)
                .eq("step", step)
                .eq("mode", mode)
                .single()
                .execute()
            )

            if not result.data:
                logger.warning(f"Artifact not found: {run_id}/{step}/{mode}")
                return False

            current_count = result.data.get("download_count", 0) or 0

            # Update download count and last_downloaded_at
            from datetime import datetime, timezone

            self.supabase.table("artifacts").update(
                {
                    "download_count": current_count + 1,
                    "last_downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("run_id", run_id).eq("step", step).eq("mode", mode).execute()

            return True
        except Exception as e:
            logger.error(f"Error updating artifact download: {e}")
            return False

    def update_run_s3_uri(
        self,
        run_id: str,
        s3_uri: str,
    ) -> bool:
        """Update the run's latest S3 artifact URI."""
        try:
            from datetime import datetime, timezone

            self.supabase.table("runs").update(
                {
                    "s3_artifact_uri": s3_uri,
                    "last_access_time": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("id", run_id).execute()

            return True
        except Exception as e:
            logger.error(f"Error updating run S3 URI for {run_id}: {e}")
            return False

    def mark_volume_cleaned(
        self,
        run_id: str,
        bytes_freed: Optional[int] = None,
    ) -> bool:
        """Mark a run's Modal Volume data as cleaned up."""
        try:
            self.supabase.table("runs").update(
                {
                    "volume_cleaned": True,
                }
            ).eq("id", run_id).execute()

            logger.info(
                f"Marked volume cleaned for run {run_id} ({bytes_freed} bytes freed)"
            )
            return True
        except Exception as e:
            logger.error(f"Error marking volume cleaned for {run_id}: {e}")
            return False

    # Cost tracking

    def get_run_duration(self, run_id: str) -> Optional[float]:
        """Get run duration in hours."""
        run = self.get_run(run_id)
        if not run or not run.get("started_at"):
            return None

        from datetime import datetime, timezone

        started_at = datetime.fromisoformat(run["started_at"])

        # Use completed_at if available, else current time
        if run["status"] == "completed" and run.get("completed_at"):
            completed_at = datetime.fromisoformat(run["completed_at"])
        else:
            completed_at = datetime.now(timezone.utc)

        duration_seconds = (completed_at - started_at).total_seconds()
        return duration_seconds / 3600.0

    def get_total_storage_bytes(self, run_id: str) -> int:
        """Get total storage used by run artifacts."""
        artifacts = self.get_run_artifacts(run_id)
        total_bytes = sum(
            artifact.get("file_size_bytes", 0) or 0 for artifact in artifacts
        )
        return total_bytes

    def update_charged_amount(self, run_id: str, total_charged: float) -> bool:
        """Update how much has been charged for a run."""
        try:
            self.supabase.table("runs").update(
                {"last_charged_amount": total_charged}
            ).eq("id", run_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error updating charged amount for run {run_id}: {e}")
            return False

    def get_cost_since_last_charge(self, run_id: str) -> float:
        """Calculate cost since last charge."""
        run = self.get_run(run_id)
        if not run or not run.get("started_at"):
            return 0.0

        from datetime import datetime, timezone

        started_at = datetime.fromisoformat(run["started_at"])
        now = datetime.now(timezone.utc)
        elapsed_hours = (now - started_at).total_seconds() / 3600

        from api.pricing import get_gpu_hourly_rate

        if (
            run.get("status") == "migrating"
            and run.get("target_gpu")
            and isinstance(run.get("target_gpu"), str)
        ):
            gpu_config = run["target_gpu"]
        else:
            gpu_config = (
                run.get("current_gpu")
                or (run["config"].get("gpu_config") if run.get("config") else None)
                or "l40s:1"
            )
        total_cost = get_gpu_hourly_rate(gpu_config) * elapsed_hours

        last_charged = run.get("last_charged_amount", 0)
        return total_cost - last_charged
