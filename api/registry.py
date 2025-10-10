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

