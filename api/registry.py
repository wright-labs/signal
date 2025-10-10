"""Run tracking and metadata storage via Supabase."""
import uuid
import os
import logging
from typing import Optional, List, Dict, Any
from supabase import Client
from api.supabase_client import get_supabase

logger = logging.getLogger(__name__)


class RunRegistry:
    """Registry for tracking training runs in Supabase PostgreSQL."""
    
    # Resource limits per user
    MAX_CONCURRENT_RUNS_PER_USER = int(os.getenv("MAX_CONCURRENT_RUNS_PER_USER", "5"))
    
    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the run registry.
        
        Args:
            supabase_client: Optional Supabase client (for testing)
        """
        self.supabase = supabase_client or get_supabase()
    
    def create_run(
        self,
        user_id: str,
        base_model: str,
        config: Dict[str, Any]
    ) -> str:
        """Create a new training run.
        
        Args:
            user_id: User UUID from Supabase Auth
            base_model: Base model name (e.g., "meta-llama/Llama-3.1-8B")
            config: Run configuration dictionary
            
        Returns:
            Generated run_id
            
        Raises:
            Exception: If database operation fails or resource limits exceeded
        """
        # Check concurrent run limit
        active_runs = self.list_runs(user_id)
        active = [r for r in active_runs if r["status"] in ["initialized", "active"]]
        
        if len(active) >= self.MAX_CONCURRENT_RUNS_PER_USER:
            raise Exception(
                f"Maximum {self.MAX_CONCURRENT_RUNS_PER_USER} concurrent runs reached. "
                f"Please wait for existing runs to complete or cancel them."
            )
        
        run_id = f"run_{uuid.uuid4().hex[:16]}"
        
        self.supabase.table("runs").insert({
            "id": run_id,
            "user_id": user_id,
            "base_model": base_model,
            "status": "initialized",
            "current_step": 0,
            "config": config
        }).execute()
        
        return run_id
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run metadata.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run metadata dictionary or None if not found
        """
        try:
            result = self.supabase.table("runs").select("*").eq("id", run_id).single().execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching run {run_id}: {e}")
            return None
    
    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        current_step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update run metadata and optionally add metrics.
        
        Args:
            run_id: Run identifier
            status: New status (e.g., "active", "completed", "failed")
            current_step: Current training step number
            metrics: Metrics to record for this step
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Update run metadata
            update_data = {}
            if status is not None:
                update_data["status"] = status
            if current_step is not None:
                update_data["current_step"] = current_step
            
            if update_data:
                self.supabase.table("runs").update(update_data).eq("id", run_id).execute()
            
            # Add metrics if provided
            if metrics is not None:
                # Get current step from run if not provided
                if current_step is None:
                    run = self.get_run(run_id)
                    current_step = run["current_step"] if run else 0
                
                self.supabase.table("run_metrics").insert({
                    "run_id": run_id,
                    "step": current_step,
                    "loss": metrics.get("loss"),
                    "grad_norm": metrics.get("grad_norm"),
                    "learning_rate": metrics.get("learning_rate"),
                    "gpu_utilization": metrics.get("gpu_utilization"),
                    "throughput": metrics.get("throughput"),
                    "cost_so_far": metrics.get("cost_so_far"),
                    "metrics_data": metrics  # Store full metrics in JSONB column
                }).execute()
            
            return True
        except Exception as e:
            logger.error(f"Error updating run {run_id}: {e}")
            return False
    
    def list_runs(self, user_id: str) -> List[Dict[str, Any]]:
        """List all runs for a user.
        
        Args:
            user_id: User UUID
            
        Returns:
            List of run metadata dictionaries, sorted by creation date (newest first)
        """
        try:
            result = self.supabase.table("runs").select(
                "*"
            ).eq("user_id", user_id).order("created_at", desc=True).execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error listing runs for user {user_id}: {e}")
            return []
    
    def delete_run(self, run_id: str, user_id: str) -> bool:
        """Delete a run and all associated metrics.
        
        Args:
            run_id: Run identifier
            user_id: User UUID (for authorization check)
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Delete run (metrics cascade delete automatically)
            result = self.supabase.table("runs").delete().eq(
                "id", run_id
            ).eq("user_id", user_id).execute()
            
            return len(result.data) > 0 if result.data else False
        except Exception as e:
            logger.error(f"Error deleting run {run_id}: {e}")
            return False
    
    def get_metrics(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get all metrics for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of metrics dictionaries ordered by step, or None if error
        """
        try:
            result = self.supabase.table("run_metrics").select(
                "*"
            ).eq("run_id", run_id).order("step", desc=False).execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Error fetching metrics for run {run_id}: {e}")
            return None
    
    def get_latest_metric(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent metric for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Latest metric dictionary or None if not found
        """
        try:
            result = self.supabase.table("run_metrics").select(
                "*"
            ).eq("run_id", run_id).order("step", desc=True).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching latest metric for run {run_id}: {e}")
            return None
    
    def get_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of run with latest metrics.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with run info and latest metric, or None if not found
        """
        run = self.get_run(run_id)
        if not run:
            return None
        
        latest_metric = self.get_latest_metric(run_id)
        
        return {
            "run": run,
            "latest_metric": latest_metric
        }
    
    # =============================================================================
    # ARTIFACT TRACKING (S3 Storage)
    # =============================================================================
    
    def record_artifact(
        self,
        run_id: str,
        step: int,
        mode: str,
        s3_uri: str,
        manifest: Dict[str, Any],
        file_size_bytes: Optional[int] = None,
    ) -> bool:
        """Record a saved artifact in the artifacts table.
        
        Args:
            run_id: Run identifier
            step: Training step number
            mode: Artifact mode ('adapter', 'merged', 'state')
            s3_uri: S3 URI for the artifact
            manifest: Full manifest metadata dictionary
            file_size_bytes: Total size of artifact in bytes
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            # Insert or update artifact record (UPSERT on run_id, step, mode)
            self.supabase.table("artifacts").upsert({
                "run_id": run_id,
                "step": step,
                "mode": mode,
                "s3_uri": s3_uri,
                "manifest": manifest,
                "file_size_bytes": file_size_bytes,
            }).execute()
            
            logger.info(f"Recorded artifact for run {run_id}, step {step}, mode {mode}")
            return True
        except Exception as e:
            logger.error(f"Error recording artifact for run {run_id}: {e}")
            return False
    
    def get_run_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all artifacts for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of artifact dictionaries, sorted by step (newest first)
        """
        try:
            result = self.supabase.table("artifacts").select(
                "*"
            ).eq("run_id", run_id).order("step", desc=True).execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching artifacts for run {run_id}: {e}")
            return []
    
    def update_artifact_download(
        self,
        run_id: str,
        step: int,
        mode: str,
    ) -> bool:
        """Track artifact download by incrementing download_count.
        
        Args:
            run_id: Run identifier
            step: Training step number
            mode: Artifact mode
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Get current artifact
            result = self.supabase.table("artifacts").select(
                "download_count"
            ).eq("run_id", run_id).eq("step", step).eq("mode", mode).single().execute()
            
            if not result.data:
                logger.warning(f"Artifact not found: {run_id}/{step}/{mode}")
                return False
            
            current_count = result.data.get("download_count", 0) or 0
            
            # Update download count and last_downloaded_at
            from datetime import datetime, timezone
            self.supabase.table("artifacts").update({
                "download_count": current_count + 1,
                "last_downloaded_at": datetime.now(timezone.utc).isoformat(),
            }).eq("run_id", run_id).eq("step", step).eq("mode", mode).execute()
            
            return True
        except Exception as e:
            logger.error(f"Error updating artifact download: {e}")
            return False
    
    def update_run_s3_uri(
        self,
        run_id: str,
        s3_uri: str,
    ) -> bool:
        """Update the run's latest S3 artifact URI.
        
        Args:
            run_id: Run identifier
            s3_uri: S3 URI for the latest artifact
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            from datetime import datetime, timezone
            self.supabase.table("runs").update({
                "s3_artifact_uri": s3_uri,
                "last_access_time": datetime.now(timezone.utc).isoformat(),
            }).eq("id", run_id).execute()
            
            return True
        except Exception as e:
            logger.error(f"Error updating run S3 URI for {run_id}: {e}")
            return False
    
    def mark_volume_cleaned(
        self,
        run_id: str,
        bytes_freed: Optional[int] = None,
    ) -> bool:
        """Mark a run's Modal Volume data as cleaned up.
        
        Called by the cleanup job after deleting old run data from Modal Volume.
        
        Args:
            run_id: Run identifier
            bytes_freed: Number of bytes freed (optional, for logging)
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            self.supabase.table("runs").update({
                "volume_cleaned": True,
            }).eq("id", run_id).execute()
            
            logger.info(f"Marked volume cleaned for run {run_id} ({bytes_freed} bytes freed)")
            return True
        except Exception as e:
            logger.error(f"Error marking volume cleaned for {run_id}: {e}")
            return False

