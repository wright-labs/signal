"""Signal API - Main application.

Core training API for fine-tuning language models on Modal.
Billing and API key management are handled by the Frontier Backend.
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import sys
import os
import logging
from pathlib import Path

# Add current directory to path for modal_runtime imports
sys.path.insert(0, str(Path(__file__).parent))

from api.auth import AuthManager, get_client_ip
from api.registry import RunRegistry
from api.models import ModelRegistry
from api.logging_config import security_logger
# from api.openai_compat import router as openai_router
from api.frontier_client import get_frontier_client
from api.pricing import calculate_estimated_cost, calculate_actual_cost, get_gpu_hourly_rate, calculate_run_cost
from api.schemas import (
    RunConfig,
    RunResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    OptimStepRequest,
    OptimStepResponse,
    SampleRequest,
    SampleResponse,
    SaveStateRequest,
    SaveStateResponse,
    RunStatus,
    RunMetrics,
    TokenizeRequest,
    TokenizeResponse,
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizerInfoResponse,
    ModelInfoResponse,
    ApplyChatTemplateRequest,
    ApplyChatTemplateResponse,
    StreamSampleRequest,
    StreamChunk,
    EmbeddingsRequest,
    EmbeddingsResponse,
)

import modal

_training_session_cls_cache = {}

def get_training_session(run_id: str, gpu_config: str = "l40s:1"):
    """Get stateful training session instance for a run with specific GPU config.
    
    Args:
        run_id: Run identifier
        gpu_config: GPU configuration (e.g., "l40s:2", "a100:4")
    
    Modal automatically routes calls with same instance to same container.
    """
    global _training_session_cls_cache
    
    try:
        # Map GPU config to class name
        gpu_to_class = {
            "l40s:1": "TrainingSession_L40S_1",
            "l40s:2": "TrainingSession_L40S_2",
            "l40s:4": "TrainingSession_L40S_4",
            "a100:1": "TrainingSession_A100_1",
            "a100:2": "TrainingSession_A100_2",
            "a100:4": "TrainingSession_A100_4",
            "h100:1": "TrainingSession_H100_1",
            "h100:2": "TrainingSession_H100_2",
        }
        
        class_name = gpu_to_class.get(gpu_config)
        if class_name is None:
            # Fallback to single GPU
            logging.warning(f"GPU config '{gpu_config}' not found, using l40s:1")
            class_name = "TrainingSession_L40S_1"
            gpu_config = "l40s:1"
        
        # Cache per GPU config
        if gpu_config not in _training_session_cls_cache:
            _training_session_cls_cache[gpu_config] = modal.Cls.from_name(
                "signal", 
                class_name, 
                environment_name="main"
            )
        
        # Return new instance
        return _training_session_cls_cache[gpu_config]()
        
    except Exception as e:
        import traceback
        logging.error(f"Failed to lookup TrainingSession class for {gpu_config}: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Training infrastructure not available for GPU config '{gpu_config}'. Please ensure Modal is deployed."
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    logging.info("🚀 Signal API starting...")
    
    # Verify database connectivity
    try:
        from api.supabase_client import get_supabase
        get_supabase().table("runs").select("id").limit(1).execute()
        logging.info("✅ Database connected")
    except Exception as e:
        logging.error(f"⚠️ Database unavailable: {e}")
    
    # Verify model registry
    model_count = len(model_registry.list_models())
    if model_count == 0:
        logging.warning("⚠️ No models loaded from config/models.yaml")
    else:
        logging.info(f"✅ Loaded {model_count} models")
    
    # Check Modal credentials configured
    if os.getenv("MODAL_TOKEN_ID"):
        logging.info("✅ Modal credentials configured")
    else:
        logging.warning("⚠️ MODAL_TOKEN_ID not set")
    
    logging.info("📡 Modal functions will be looked up on first request")
    
    yield
    
    logging.info("🛑 Shutting down Signal API...")
    from api.supabase_client import get_supabase
    get_supabase.cache_clear()
    logging.info("✅ Cleanup complete")


app = FastAPI(
    title="Signal API",
    description="Open source training API for fine-tuning language models",
    version="0.1.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    lifespan=lifespan,
)

# Middleware
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        max_size = int(os.getenv("MAX_REQUEST_SIZE_MB", "20")) * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            return JSONResponse(
                status_code=413,
                content={"error": f"Request body too large (max {max_size // 1024 // 1024}MB)"}
            )
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting by user_id or IP
def get_rate_limit_key(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    return f"user:{user_id}" if user_id else f"ip:{get_remote_address(request)}"

limiter = Limiter(key_func=get_rate_limit_key, default_limits=["500/hour"])

if os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true":
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(openai_router)

# Initialize managers
auth_manager = AuthManager()
run_registry = RunRegistry()
model_registry = ModelRegistry()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# AUTHENTICATION

async def verify_auth(
    authorization: Optional[str] = Header(None),
    request: Request = None
) -> str:
    """Verify authentication (JWT or API key) and return user_id."""
    if not authorization:
        if request:
            ip = get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            security_logger.log_auth_failure(ip, user_agent, "Missing authorization header")
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        if request:
            ip = get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            security_logger.log_auth_failure(ip, user_agent, "Invalid authorization header format")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = parts[1]
    
    # Try API key (sk-) or JWT token
    if token.startswith("sk-"):
        user_id = await auth_manager.validate_api_key(token)
        if user_id:
            if request:
                request.state.user_id = user_id
            return user_id
    else:
        user_id = await auth_manager.validate_jwt_token(token)
        if user_id:
            if request:
                request.state.user_id = user_id
            return user_id
    
    if request:
        ip = get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        security_logger.log_auth_failure(ip, user_agent, "Invalid credentials")
    
    raise HTTPException(status_code=401, detail="Invalid credentials")


async def get_authorized_run(run_id: str, user_id: str) -> Dict[str, Any]:
    """Get run and verify user owns it."""
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return run


def validate_gpu_config(gpu_config: str) -> bool:
    """Validate GPU config format (e.g., 'l40s:1', 'a100:4').
    
    
    """
    if not gpu_config or ":" not in gpu_config:
        raise HTTPException(
            status_code=400,
            detail="Invalid GPU config format. Expected format: 'gpu_type:count' (e.g., 'l40s:1')"
        )
    
    parts = gpu_config.split(":")
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail="Invalid GPU config format. Expected format: 'gpu_type:count' (e.g., 'l40s:1')"
        )
    
    gpu_type, count_str = parts
    if not gpu_type or not count_str:
        raise HTTPException(
            status_code=400,
            detail="Invalid GPU config format. GPU type and count cannot be empty"
        )
    
    try:
        count = int(count_str)
        if count < 1 or count > 8:
            raise HTTPException(
                status_code=400,
                detail="GPU count must be between 1 and 8"
            )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="GPU count must be a valid integer"
        )
    
    return True


async def check_and_charge_incremental(
    run_id: str,
    user_id: str,
    gpu_config: str,
) -> None:
    """Check balance and charge if needed based on time elapsed. Raises 402 if insufficient.
    
    Now uses time-based checks (every 2 minutes) instead of step-based checks,
    which is safer for expensive GPUs with slow training steps.
    """
    from datetime import datetime, timezone
    
    run = run_registry.get_run(run_id)
    if not run or not run.get("started_at"):
        return
    
    # Check if we need to check balance (time-based, not step-based)
    last_check = run.get("last_balance_check_at")
    if last_check:
        last_check_dt = datetime.fromisoformat(last_check)
        minutes_since_check = (datetime.now(timezone.utc) - last_check_dt).total_seconds() / 60
        
        # Skip if checked recently (default: every 2 minutes)
        balance_check_interval_minutes = float(os.getenv("BALANCE_CHECK_INTERVAL_MINUTES", "2.0"))
        if minutes_since_check < balance_check_interval_minutes:
            return  # Not time to check yet
    
    # Calculate cost so far (including storage)
    started_at = datetime.fromisoformat(run["started_at"])
    now = datetime.now(timezone.utc)
    elapsed_hours = (now - started_at).total_seconds() / 3600
    
    # Calculate GPU cost
    gpu_cost = get_gpu_hourly_rate(gpu_config) * elapsed_hours
    
    # Calculate storage cost (incremental charging)
    storage_bytes = run_registry.get_total_storage_bytes(run_id)
    storage_gb = storage_bytes / (1024 ** 3)
    storage_cost_per_hour = (0.023 / 730)  # $0.023/GB/month converted to hourly
    storage_cost = storage_gb * storage_cost_per_hour * elapsed_hours
    
    cost_so_far = gpu_cost + storage_cost
    
    # Charge in increments (default $1)
    charge_increment = float(os.getenv("CHARGE_INCREMENT", "1.0"))
    last_charged = run.get("last_charged_amount", 0) or 0
    amount_to_charge = cost_so_far - last_charged
    
    frontier_client = get_frontier_client()
    
    if amount_to_charge >= charge_increment:
        # Try to charge
        success = await frontier_client.charge_increment(
            user_id, amount_to_charge, run_id, run["current_step"]
        )
        if not success:
            raise HTTPException(402, "Insufficient credits to continue")
        
        # Update tracking
        run_registry.update_charged_amount(run_id, cost_so_far)
        logging.info(f"Charged ${amount_to_charge:.4f} for run {run_id} (total: ${cost_so_far:.4f})")
    
    # Update last balance check timestamp
    from api.supabase_client import get_supabase
    supabase = get_supabase()
    try:
        supabase.table("runs").update({
            "last_balance_check_at": now.isoformat()
        }).eq("id", run_id).execute()
    except Exception as e:
        logging.warning(f"Failed to update last_balance_check_at: {e}")
    
    # Check remaining balance
    balance = await frontier_client.get_balance(user_id)
    min_balance = float(os.getenv("MIN_BALANCE_THRESHOLD", "0.50"))
    if balance < min_balance:
        raise HTTPException(402, f"Credits depleted. Balance: ${balance:.2f}. Add funds to continue.")


# Endpoints

@app.get("/")
async def root():
    return {
        "name": "Signal API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/debug/modal")
async def debug_modal(user_id: str = Depends(verify_auth)):
    """Debug endpoint to test Modal connection. Development only."""
    if os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(status_code=403, detail="Debug endpoint disabled in production")
    
    import modal as modal_lib
    
    modal_token_id = os.getenv("MODAL_TOKEN_ID", "NOT SET")
    modal_token_secret = os.getenv("MODAL_TOKEN_SECRET", "NOT SET")
    
    debug_info = {
        "modal_token_id": modal_token_id[:15] + "..." if modal_token_id != "NOT SET" else "NOT SET",
        "modal_token_secret": "SET" if modal_token_secret != "NOT SET" else "NOT SET",
        "environment": "main",
        "authenticated_user": user_id,
    }
    
    try:
        func = modal_lib.Function.from_name("signal", "create_run", environment_name="main")
        result = func.remote(
            user_id=user_id,
            run_id="debug-001",
            base_model="Qwen/Qwen2.5-3B",
            framework="transformers",
            gpu_config="l40s:1"
        )
        return {"status": "success", "modal_works": True, "result": result, "debug": debug_info}
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "modal_works": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "debug": debug_info
        }


@app.post("/internal/mark-volume-cleaned")
async def mark_volume_cleaned(
    request: Request,
    run_id: str = None,
    bytes_freed: int = None,
):
    """Internal endpoint for cleanup job to mark Modal Volume as cleaned."""
    internal_key = os.getenv("SIGNAL_INTERNAL_SECRET")
    provided_key = request.headers.get("X-Internal-Key")
    
    if not internal_key or provided_key != internal_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing internal service key"
        )
    
    # Parse request body if not provided as query params
    if not run_id:
        try:
            body = await request.json()
            run_id = body.get("run_id")
            bytes_freed = body.get("bytes_freed")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid request body")
    
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    
    try:
        # Mark volume as cleaned in database
        success = run_registry.mark_volume_cleaned(
            run_id=run_id,
            bytes_freed=bytes_freed,
        )
        
        if success:
            return {
                "status": "success",
                "run_id": run_id,
                "volume_cleaned": True,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to update database"
            )
    
    except Exception as e:
        logging.error(f"Error marking volume cleaned for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/internal/charge-final-cost")
async def charge_final_cost(request: Request):
    """Internal endpoint to charge remaining cost for a completed/failed run."""
    internal_key = os.getenv("SIGNAL_INTERNAL_SECRET")
    provided_key = request.headers.get("X-Internal-Key")
    
    if not internal_key or provided_key != internal_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing internal service key"
        )
    
    try:
        body = await request.json()
        run_id = body.get("run_id")
        user_id = body.get("user_id")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body")
    
    if not run_id or not user_id:
        raise HTTPException(status_code=400, detail="run_id and user_id required")
    
    try:
        from datetime import datetime, timezone
        
        # Get run details
        run = run_registry.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Verify user_id matches
        if run["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="User mismatch")
        
        # Calculate actual cost
        if not run.get("started_at"):
            logging.warning(f"Run {run_id} has no started_at timestamp - skipping charge")
            return {
                "success": True,
                "run_id": run_id,
                "message": "No charge needed - run never started",
                "charged": 0.0
            }
        
        started_at = datetime.fromisoformat(run["started_at"])
        ended_at = datetime.fromisoformat(
            run.get("completed_at") or datetime.now(timezone.utc).isoformat()
        )
        gpu_config = run["config"].get("gpu_config", "l40s:1")
        storage_bytes = run_registry.get_total_storage_bytes(run_id)
        
        # Use unified cost calculation
        cost_breakdown = calculate_run_cost(
            gpu_config=gpu_config,
            started_at=started_at,
            ended_at=ended_at,
            storage_bytes=storage_bytes,
            include_storage=True,
        )
        actual_cost = cost_breakdown["total_cost"]
        
        # Charge only what hasn't been charged yet
        already_charged = run.get("last_charged_amount", 0) or 0
        remaining_cost = max(0, actual_cost - already_charged)
        
        if remaining_cost > 0:
            frontier_client = get_frontier_client()
            success = await frontier_client.charge_increment(
                user_id=user_id,
                amount=remaining_cost,
                run_id=run_id,
                step=run["current_step"]
            )
            
            if success:
                run_registry.update_charged_amount(run_id, actual_cost)
                logging.info(f"Charged final ${remaining_cost:.4f} for run {run_id}")
            else:
                logging.warning(f"Failed to charge final cost for run {run_id}")
                return {
                    "success": False,
                    "run_id": run_id,
                    "message": "Insufficient credits",
                    "remaining_cost": remaining_cost
                }
        
        return {
            "success": True,
            "run_id": run_id,
            "total_cost": actual_cost,
            "already_charged": already_charged,
            "charged": remaining_cost
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error charging final cost for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models for fine-tuning."""
    return {
        "models": model_registry.list_models(),
    }


@app.post("/runs", response_model=RunResponse)
@limiter.limit("5/minute")
async def create_run(
    config: RunConfig,
    request: Request,
    user_id: str = Depends(verify_auth),
):
    """Create a new training run."""
    try:
        # Log run creation
        ip = get_client_ip(request)
        security_logger.log_run_created(user_id, "pending", config.base_model, ip)
        
        # Validate model
        if not model_registry.is_supported(config.base_model):
            raise HTTPException(
                status_code=400,
                detail=f"Model {config.base_model} not supported. Use /models to see available models.",
            )
        
        # Get model config
        model_config = model_registry.get_model(config.base_model)
        framework = model_config["framework"]
        
        # Use user-provided GPU config, or fallback to model's default
        if config.gpu_config:
            gpu_config = config.gpu_config
            logging.info(f"Using user-provided GPU config: {gpu_config}")
        else:
            gpu_config = model_config["gpu"]
            logging.info(f"Using model's default GPU config: {gpu_config}")
        
        # Validate GPU config format
        validate_gpu_config(gpu_config)
        
        # Get config as dict
        config_dict = config.model_dump()
        config_dict["gpu_config"] = gpu_config
        
        # Calculate minimum balance based on GPU cost
        # Require enough for at least MIN_TRAINING_MINUTES of GPU time + checkpoint cost
        min_training_minutes = float(os.getenv("MIN_TRAINING_MINUTES", "30"))
        checkpoint_cost_multiplier = float(os.getenv("CHECKPOINT_COST_MULTIPLIER", "1.5"))
        
        gpu_hourly_rate = get_gpu_hourly_rate(gpu_config)
        min_gpu_cost = gpu_hourly_rate * (min_training_minutes / 60)
        min_balance_required = min_gpu_cost * checkpoint_cost_multiplier
        
        # Apply absolute minimum floor (default $2)
        absolute_min = float(os.getenv("ABSOLUTE_MIN_BALANCE", "2.0"))
        min_balance_required = max(min_balance_required, absolute_min)
        
        frontier_client = get_frontier_client()
        user_balance = await frontier_client.get_balance(user_id)
        
        if user_balance < min_balance_required:
            raise HTTPException(
                status_code=402,
                detail=f"Minimum balance required: ${min_balance_required:.2f} "
                       f"({min_training_minutes:.0f} minutes @ ${gpu_hourly_rate:.2f}/hr + checkpoint buffer). "
                       f"Current balance: ${user_balance:.2f}. Please add credits to continue."
            )
        
        # Fetch user integrations from Frontier Backend
        integrations = await frontier_client.get_integrations(user_id)
        
        # Create run in registry
        run_id = run_registry.create_run(
            user_id=user_id,
            base_model=config.base_model,
            config=config_dict,
        )
        
        # Initialize training run on Modal with proper error handling
        logging.info(f"Creating run with GPU config: {gpu_config}")
        
        try:
            # Initialize stateful container with GPU config
            session = get_training_session(run_id, gpu_config=gpu_config)
            result = session.initialize.remote(
                user_id=user_id,
                run_id=run_id,
                base_model=config.base_model,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                lora_target_modules=config.lora_target_modules,
                optimizer=config.optimizer,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                max_seq_length=config.max_seq_length,
                bf16=config.bf16,
                gradient_checkpointing=config.gradient_checkpointing,
                load_in_8bit=False,
                load_in_4bit=True,
                accumulation_steps=1,
                auto_checkpoint_interval=100,
            )
            
            # Update registry to running status ONLY if Modal succeeds
            run_registry.update_run(run_id, status="running")
            
        except Exception as modal_error:
            # CRITICAL: Clean up failed run to free up concurrent run slot
            run_registry.update_run(run_id, status="failed")
            logging.error(f"Modal initialization failed for run {run_id}: {modal_error}")
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize training infrastructure. Please try again."
            )
        
        # Get run info
        run_info = run_registry.get_run(run_id)
        
        return RunResponse(
            run_id=run_id,
            user_id=user_id,
            base_model=config.base_model,
            status=run_info["status"],
            created_at=run_info["created_at"],
            config=run_info["config"],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error creating run for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create training run. Please try again or contact support."
        )


@app.post("/runs/{run_id}/forward_backward", response_model=ForwardBackwardResponse)
@limiter.limit("1000/minute")
async def forward_backward(
    run_id: str,
    fb_request: ForwardBackwardRequest,
    request: Request,
    user_id: str = Depends(verify_auth),
):
    """Perform forward-backward pass."""
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        # Get current step
        current_step = run["current_step"]
        
        # Get GPU config from run config
        gpu_config = run["config"].get("gpu_config", "l40s:1")
        logging.info(f"Forward-backward with GPU config: {gpu_config}")
        
        # Check balance (time-based, not step-based)
        await check_and_charge_incremental(run_id, user_id, gpu_config)
        
        # Use stateful container (fast, model already loaded)
        session = get_training_session(run_id, gpu_config=gpu_config)
        result = session.forward_backward.remote(
            batch_data=fb_request.batch_data,
            loss_fn=fb_request.loss_fn,
            loss_kwargs=fb_request.loss_kwargs,
        )
        
        # Update registry with metrics
        run_registry.update_run(
            run_id,
            metrics={
                "loss": result["loss"],
                "grad_norm": result["grad_norm"],
                "gpu_utilization": result.get("gpu_utilization", 85.0),
                "throughput": result.get("throughput"),
            },
        )
        
        return ForwardBackwardResponse(
            loss=result["loss"],
            step=result["step"],
            grad_norm=result.get("grad_norm"),
            grad_stats=result.get("grad_stats"),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error in forward_backward for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Forward-backward pass failed. Please check your batch data or try again."
        )


@app.post("/runs/{run_id}/optim_step", response_model=OptimStepResponse)
@limiter.limit("300/minute")
async def optim_step(
    run_id: str,
    request: OptimStepRequest,
    req: Request,
    user_id: str = Depends(verify_auth),
):
    """Apply optimizer step."""
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        # Get current step
        current_step = run["current_step"]
        
        # Get GPU config from run config
        gpu_config = run["config"].get("gpu_config", "l40s:1")
        
        # Check balance (time-based, not step-based)
        await check_and_charge_incremental(run_id, user_id, gpu_config)
        
        # Use stateful container (fast, optimizer already loaded)
        session = get_training_session(run_id, gpu_config=gpu_config)
        result = session.optim_step.remote(
            learning_rate=request.learning_rate,
        )
        
        # Update registry with step
        run_registry.update_run(
            run_id,
            current_step=result["step"],
            metrics={
                "learning_rate": result["learning_rate"],
                "gpu_utilization": result.get("gpu_utilization", 85.0),
            }
        )
        
        return OptimStepResponse(
            step=result["step"],
            learning_rate=result["learning_rate"],
            metrics=result.get("metrics", {}),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error in optim_step for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Optimizer step failed. Please try again."
        )


@app.post("/runs/{run_id}/sample", response_model=SampleResponse)
@limiter.limit("20/minute")
async def sample(
    run_id: str,
    request: SampleRequest,
    req: Request,
    user_id: str = Depends(verify_auth),
):
    """Generate samples from the model."""
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        # Get current step
        current_step = run["current_step"]
        
        # Use single GPU for inference regardless of training GPU count
        gpu_config = run["config"].get("gpu_config", "l40s:1")
        
        # Check balance before expensive operations
        await check_and_charge_incremental(run_id, user_id, gpu_config)
        
        # Use stateful container (fast, model already loaded)
        session = get_training_session(run_id, gpu_config=gpu_config)
        result = session.sample.remote(
            prompts=request.prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if hasattr(request, 'top_k') else None,
            return_logprobs=request.return_logprobs,
        )
        
        return SampleResponse(
            outputs=result["outputs"],
            token_ids=result.get("token_ids", []),
            tokens=result.get("tokens", []),
            logprobs=result.get("logprobs"),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error in sample for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Sample generation failed. Please check your prompts or try again."
        )


@app.get("/runs/{run_id}/session_state")
@limiter.limit("100/minute")
async def get_session_state(
    run_id: str,
    user_id: str = Depends(verify_auth),
):
    """Get current stateful session state."""
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        # Get GPU config and session state
        gpu_config = run["config"].get("gpu_config", "l40s:1")
        session = get_training_session(run_id, gpu_config=gpu_config)
        state = session.get_state.remote()
        
        return state
    
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error getting session state for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get session state."
        )


@app.post("/runs/{run_id}/sample/stream")
@limiter.limit("20/minute")
async def sample_stream(
    run_id: str,
    request: StreamSampleRequest,
    req: Request,
    user_id: str = Depends(verify_auth),
):
    """Stream generated text token-by-token using Server-Sent Events."""
    # TODO: Implement streaming with stateful session
    raise HTTPException(
        status_code=501,
        detail="Streaming is temporarily unavailable. Use /runs/{run_id}/sample instead."
    )


@app.post("/runs/{run_id}/embeddings", response_model=EmbeddingsResponse)
@limiter.limit("20/minute")
async def generate_embeddings(
    run_id: str,
    request: EmbeddingsRequest,
    req: Request,
    user_id: str = Depends(verify_auth),
):
    """Generate embeddings from the model."""
    # TODO: Implement embeddings with stateful session
    raise HTTPException(
        status_code=501,
        detail="Embeddings endpoint is temporarily unavailable."
    )


@app.post("/runs/{run_id}/save_state", response_model=SaveStateResponse)
@limiter.limit("5/minute")
async def save_state(
    run_id: str,
    request: SaveStateRequest,
    req: Request,
    user_id: str = Depends(verify_auth),
):
    """Save model state."""
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        # Get current step
        current_step = run["current_step"]
        
        # Use single GPU for save operation
        gpu_config = run["config"].get("gpu_config", "l40s:1")
        
        # Check balance before expensive operations
        await check_and_charge_incremental(run_id, user_id, gpu_config)
        
        # Use stateful container's save_state method
        session = get_training_session(run_id, gpu_config=gpu_config)
        result = session.save_state.remote(
            mode=request.mode,
            push_to_hub=request.push_to_hub,
            hub_model_id=request.hub_model_id,
        )
        
        # Record S3 artifact in database
        if result.get("s3_uri"):
            logging.info(f"Artifact saved to S3: {result['s3_uri']}")
            
            artifact_recorded = run_registry.record_artifact(
                run_id=run_id,
                step=current_step,
                mode=request.mode,
                s3_uri=result["s3_uri"],
                manifest=result.get("manifest", {}),
                file_size_bytes=result.get("manifest", {}).get("total_size_bytes"),
            )
            
            if artifact_recorded:
                run_registry.update_run_s3_uri(run_id=run_id, s3_uri=result["s3_uri"])
            else:
                logging.warning(f"Failed to record artifact in database for run {run_id}")
        
        return SaveStateResponse(
            artifact_uri=result.get("local_path", ""),
            local_path=result.get("local_path"),
            checkpoint_path=result.get("local_path", ""),
            s3_uri=result.get("s3_uri"),
            download_url=result.get("download_url"),
            download_expires_at=result.get("download_expires_at"),
            manifest=result.get("manifest"),
            pushed_to_hub=result.get("pushed_to_hub", False),
            hub_model_id=result.get("hub_model_id"),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error in save_state for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save model state. Please try again."
        )


@app.post("/runs/{run_id}/tokenize", response_model=TokenizeResponse)
async def tokenize(
    run_id: str,
    request: TokenizeRequest,
    user_id: str = Depends(verify_auth),
):
    """Tokenize text using the model's tokenizer."""
    # TODO: Implement using stateful session
    raise HTTPException(
        status_code=501,
        detail="This endpoint is temporarily unavailable during the stateful migration. Use the training endpoints instead."
    )


@app.post("/runs/{run_id}/detokenize", response_model=DetokenizeResponse)
async def detokenize(
    run_id: str,
    request: DetokenizeRequest,
    user_id: str = Depends(verify_auth),
):
    """Detokenize token IDs using the model's tokenizer."""
    # TODO: Implement using stateful session
    raise HTTPException(
        status_code=501,
        detail="This endpoint is temporarily unavailable during the stateful migration."
    )


@app.get("/runs/{run_id}/tokenizer_info", response_model=TokenizerInfoResponse)
async def get_tokenizer_info(
    run_id: str,
    user_id: str = Depends(verify_auth),
):
    """Get tokenizer configuration information."""
    # TODO: Implement using stateful session
    raise HTTPException(
        status_code=501,
        detail="This endpoint is temporarily unavailable during the stateful migration."
    )


@app.get("/runs/{run_id}/model_info", response_model=ModelInfoResponse)
async def get_model_info(
    run_id: str,
    user_id: str = Depends(verify_auth),
):
    """Get model architecture information."""
    # TODO: Implement using stateful session
    raise HTTPException(
        status_code=501,
        detail="This endpoint is temporarily unavailable during the stateful migration."
    )


@app.post("/runs/{run_id}/apply_chat_template", response_model=ApplyChatTemplateResponse)
async def apply_chat_template(
    run_id: str,
    request: ApplyChatTemplateRequest,
    user_id: str = Depends(verify_auth),
):
    """Apply the model's chat template to format messages."""
    # TODO: Implement using stateful session
    raise HTTPException(
        status_code=501,
        detail="This endpoint is temporarily unavailable during the stateful migration."
    )


@app.get("/runs/{run_id}/status", response_model=RunStatus)
async def get_run_status(
    run_id: str,
    user_id: str = Depends(verify_auth),
):
    """Get run status with real-time cost information."""
    try:
        run = await get_authorized_run(run_id, user_id)
        
        # Calculate cost so far
        cost_so_far = 0.0
        cost_per_hour = 0.0
        if run.get("started_at"):
            from datetime import datetime, timezone
            started_at = datetime.fromisoformat(run["started_at"])
            
            if run["status"] == "completed" and run.get("completed_at"):
                end_time = datetime.fromisoformat(run["completed_at"])
            else:
                end_time = datetime.now(timezone.utc)
            
            elapsed_hours = (end_time - started_at).total_seconds() / 3600
            gpu_config = run["config"].get("gpu_config", "l40s:1")
            cost_per_hour = get_gpu_hourly_rate(gpu_config)
            cost_so_far = cost_per_hour * elapsed_hours
        
        # Add cost info to response
        response_data = dict(run)
        response_data["cost_so_far"] = cost_so_far
        response_data["charged_so_far"] = run.get("last_charged_amount", 0) or 0
        response_data["cost_per_hour"] = cost_per_hour
        
        return RunStatus(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}/artifacts")
async def get_run_artifacts(
    run_id: str,
    user_id: str = Depends(verify_auth),
):
    """Get all artifacts for a run with signed download URLs."""
    try:
        _ = await get_authorized_run(run_id, user_id)
        
        artifacts_data = run_registry.get_run_artifacts(run_id)
        
        from datetime import datetime, timezone, timedelta
        from modal_runtime.s3_client import generate_signed_url
        
        artifacts = []
        for artifact in artifacts_data:
            try:
                download_url = generate_signed_url(artifact["s3_uri"], expiration=3600)
                expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
                
                artifacts.append({
                    "id": artifact.get("id"),
                    "step": artifact["step"],
                    "mode": artifact["mode"],
                    "s3_uri": artifact["s3_uri"],
                    "download_url": download_url,
                    "download_expires_at": expires_at,
                    "created_at": artifact.get("created_at"),
                    "file_size_bytes": artifact.get("file_size_bytes"),
                    "download_count": artifact.get("download_count", 0),
                    "manifest": artifact.get("manifest", {}),
                })
                
            except Exception as e:
                logging.error(f"Failed to generate signed URL for artifact {artifact.get('id')}: {e}")
                artifacts.append({
                    "id": artifact.get("id"),
                    "step": artifact["step"],
                    "mode": artifact["mode"],
                    "s3_uri": artifact["s3_uri"],
                    "download_url": None,
                    "download_expires_at": None,
                    "created_at": artifact.get("created_at"),
                    "file_size_bytes": artifact.get("file_size_bytes"),
                    "download_count": artifact.get("download_count", 0),
                    "manifest": artifact.get("manifest", {}),
                    "error": "Failed to generate signed URL",
                })
        
        return {
            "run_id": run_id,
            "artifacts": artifacts,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}/metrics", response_model=RunMetrics)
async def get_run_metrics(
    run_id: str,
    user_id: str = Depends(verify_auth),
):
    """Get run metrics."""
    try:
        run = await get_authorized_run(run_id, user_id)
        metrics = run_registry.get_metrics(run_id)
        
        return RunMetrics(
            run_id=run_id,
            step=run["current_step"],
            metrics=metrics or [],
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/runs/{run_id}/complete")
@limiter.limit("10/minute")
async def complete_run(
    run_id: str,
    request: Request,
    user_id: str = Depends(verify_auth),
):
    """Mark run as complete and charge remaining credits."""
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        if run["status"] == "completed":
            return {"message": "Run already completed"}
        
        # Update run status to completed (tracks completed_at timestamp)
        run_registry.update_run(run_id, status="completed")
        
        # Calculate actual cost based on GPU time and storage
        run = run_registry.get_run(run_id)  # Refresh to get completed_at
        if not run.get("started_at"):
            logging.warning(f"Run {run_id} has no started_at timestamp")
            actual_cost = 0.0
        else:
            from datetime import datetime, timezone
            started_at = datetime.fromisoformat(run["started_at"])
            completed_at = datetime.fromisoformat(run.get("completed_at", datetime.now(timezone.utc).isoformat()))
            gpu_config = run["config"].get("gpu_config", "l40s:1")
            storage_bytes = run_registry.get_total_storage_bytes(run_id)
            
            # Use unified cost calculation
            cost_breakdown = calculate_run_cost(
                gpu_config=gpu_config,
                started_at=started_at,
                ended_at=completed_at,
                storage_bytes=storage_bytes,
                include_storage=True,
            )
            actual_cost = cost_breakdown["total_cost"]
        
        # Charge only what hasn't been charged yet
        already_charged = run.get("last_charged_amount", 0) or 0
        remaining_cost = max(0, actual_cost - already_charged)
        
        if remaining_cost > 0:
            frontier_client = get_frontier_client()
            success = await frontier_client.charge_increment(
                user_id=user_id,
                amount=remaining_cost,
                run_id=run_id,
                step=run["current_step"]
            )
            if success:
                run_registry.update_charged_amount(run_id, actual_cost)
            else:
                logging.warning(f"Final charge failed for run {run_id}, user may have insufficient credits")
        
        return {
            "success": True,
            "run_id": run_id,
            "status": "completed",
            "total_cost": actual_cost,
            "charged_during_run": already_charged,
            "final_charge": remaining_cost
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error completing run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to complete run. Please try again or contact support."
        )


@app.get("/runs")
async def list_runs(user_id: str = Depends(verify_auth)):
    """List all runs for the authenticated user."""
    try:
        runs = run_registry.list_runs(user_id)
        return {"runs": runs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error Handling

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 400:
        ip = request.client.host if request.client else "unknown"
        logging.warning(f"HTTP {exc.status_code}: {exc.detail} from {ip}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

