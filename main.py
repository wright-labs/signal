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
from api.openai_compat import router as openai_router
from api.frontier_client import get_frontier_client
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
    ErrorResponse,
)

# Import Modal for remote function lookups
import modal

# Lookup deployed Modal functions (lazy-loaded on first use)
# These are looked up from the deployed "signal" app
_modal_functions_cache = {}

def get_modal_function(name: str):
    """Get Modal function by name from deployed environment."""
    try:
        # Lookup from deployed Modal app in the main workspace environment
        # The environment_name="main" refers to the default environment in your workspace
        return modal.Function.from_name("signal", name, environment_name="main")
    except Exception as e:
        import traceback
        logging.error(f"Failed to lookup Modal function '{name}': {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        logging.error(f"Modal token ID: {os.getenv('MODAL_TOKEN_ID', 'NOT SET')[:10]}...")
        raise HTTPException(
            status_code=500,
            detail=f"Training infrastructure not available. Please ensure Modal functions are deployed."
        )


# Function wrappers for consistent interface
modal_create_run = lambda: get_modal_function("create_run")
modal_forward_backward = lambda: get_modal_function("forward_backward")
modal_optim_step = lambda: get_modal_function("optim_step")
modal_sample = lambda: get_modal_function("sample")
modal_save_state = lambda: get_modal_function("save_state")

# =============================================================================
# LIFESPAN EVENTS
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logging.info("🚀 Signal API starting...")
    logging.info("📡 Modal functions will be looked up on first request")
    
    # Note: We don't verify Modal connection at startup to avoid:
    # 1. Startup failures if Modal is temporarily unavailable
    # 2. Unnecessary GPU provisioning costs
    # 3. Slow startup times
    # The first request will validate connectivity naturally.
    
    yield
    
    # Shutdown
    logging.info("Shutting down Signal API...")

# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Signal API",
    description="Self-hostable training API for fine-tuning language models on Modal",
    version="0.1.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    lifespan=lifespan,
)

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

# Custom middleware for security headers and request validation
# Note: These are custom implementations for fine-grained control over security settings.
# FastAPI doesn't have built-in security headers middleware, and this approach gives us
# flexibility to adjust headers based on environment (e.g., HSTS only in production).

# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if os.getenv("ENVIRONMENT") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Request Size Limit Middleware
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

# GZip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate Limiting
def get_user_id_for_rate_limit(request: Request) -> str:
    """Get identifier for rate limiting (IP address)."""
    return get_remote_address(request)

limiter = Limiter(
    key_func=get_user_id_for_rate_limit,
    default_limits=["1000/hour"]
)

if os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true":
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include OpenAI-compatible router for Verifiers integration
app.include_router(openai_router)

# Initialize managers
auth_manager = AuthManager()
run_registry = RunRegistry()
model_registry = ModelRegistry()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# =============================================================================
# AUTHENTICATION
# =============================================================================


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    return request.client.host if request.client else "unknown"


async def verify_auth(
    authorization: Optional[str] = Header(None),
    request: Request = None
) -> str:
    """Verify authentication (JWT or API key) and return user_id.
    
    Supports hybrid authentication:
    - API keys (sk-xxx format) for programmatic access
    - JWT tokens from Supabase Auth for web app users
    """
    if not authorization:
        # Log auth failure
        if request:
            ip = get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            security_logger.log_auth_failure(ip, user_agent, "Missing authorization header")
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        if request:
            ip = get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            security_logger.log_auth_failure(ip, user_agent, "Invalid authorization header format")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = parts[1]
    
    # Try API key first (sk- prefix)
    if token.startswith("sk-"):
        user_id = await auth_manager.validate_api_key(token)
        if user_id:
            return user_id
    else:
        # Try JWT token from Supabase Auth
        user_id = await auth_manager.validate_jwt_token(token)
        if user_id:
            return user_id
    
    # Log auth failure
    if request:
        ip = get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        security_logger.log_auth_failure(ip, user_agent, "Invalid credentials")
    
    raise HTTPException(status_code=401, detail="Invalid credentials")


async def get_authorized_run(run_id: str, user_id: str) -> Dict[str, Any]:
    """Get run and verify user owns it.
    
    Args:
        run_id: Run identifier
        user_id: User ID from authentication
        
    Returns:
        Run data dictionary
        
    Raises:
        HTTPException: If run not found or unauthorized
    """
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return run


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Signal API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/debug/modal")
async def debug_modal():
    """Debug endpoint to test Modal connection."""
    import modal as modal_lib
    
    # Check Modal environment
    modal_token_id = os.getenv("MODAL_TOKEN_ID", "NOT SET")
    modal_token_secret = os.getenv("MODAL_TOKEN_SECRET", "NOT SET")
    
    debug_info = {
        "modal_token_id": modal_token_id[:15] + "..." if modal_token_id != "NOT SET" else "NOT SET",
        "modal_token_secret": "SET" if modal_token_secret != "NOT SET" else "NOT SET",
        "environment": "main",
    }
    
    try:
        func = modal_lib.Function.from_name("signal", "create_run", environment_name="main")
        # Try to actually call it (this provisions a GPU so only use for debugging!)
        result = func.remote(
            user_id="debug-test",
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
    user_id: str = None,
    bytes_freed: int = None,
):
    """Internal endpoint for cleanup job to mark Modal Volume as cleaned.
    
    This endpoint is called by the cleanup job after deleting old run data
    from Modal Volume. It updates the database to reflect that the volume
    has been cleaned.
    
    Protected by internal service key for service-to-service auth.
    """
    # Check internal service key
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
            user_id = body.get("user_id")
            bytes_freed = body.get("bytes_freed")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid request body")
    
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    
    try:
        # Mark volume as cleaned in database
        registry = RunRegistry()
        success = registry.mark_volume_cleaned(
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


@app.get("/models")
async def list_models():
    """List supported models."""
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
        gpu_config = model_config["gpu"]
        
        # Get config as dict
        config_dict = config.model_dump()
        
        # STEP 1: Validate user has sufficient credits
        frontier_client = get_frontier_client()
        estimated_cost = 5.0  # TODO: Calculate based on model size and config
        
        has_credits = await frontier_client.validate_credits(user_id, estimated_cost)
        if not has_credits:
            raise HTTPException(
                status_code=402,  # Payment Required
                detail=f"Insufficient credits. Estimated cost: ${estimated_cost:.2f}. "
                       "Please purchase more credits to continue."
            )
        
        # STEP 2: Fetch user integrations from Frontier Backend
        integrations = await frontier_client.get_integrations(user_id)
        
        # Create run in registry
        run_id = run_registry.create_run(
            user_id=user_id,
            base_model=config.base_model,
            config=config_dict,
        )
        
        # Call Modal function to initialize run with integrations
        # TODO: GPU config is currently fixed in Modal function definition
        # For dynamic GPU allocation, consider deploying separate functions per GPU type
        logging.info(f"Creating run with GPU config: {gpu_config}")
        
        result = modal_create_run().remote(
            user_id=user_id,
            run_id=run_id,
            base_model=config.base_model,
            framework=framework,
            gpu_config=gpu_config,
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
            integrations=integrations,  # Pass decrypted integrations to Modal
        )
        
        # Update registry to running status
        run_registry.update_run(run_id, status="running")
        
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/runs/{run_id}/forward_backward", response_model=ForwardBackwardResponse)
@limiter.limit("30/minute")
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
        
        # Call Modal function
        # TODO: GPU config is currently fixed in Modal function definition
        result = modal_forward_backward().remote(
            user_id=user_id,
            run_id=run_id,
            batch_data=fb_request.batch_data,
            step=current_step,
            accumulate=fb_request.accumulate,
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/runs/{run_id}/optim_step", response_model=OptimStepResponse)
async def optim_step(
    run_id: str,
    request: OptimStepRequest,
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
        
        # Call Modal function
        # TODO: GPU config is currently fixed in Modal function definition
        result = modal_optim_step().remote(
            user_id=user_id,
            run_id=run_id,
            step=current_step,
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/runs/{run_id}/sample", response_model=SampleResponse)
async def sample(
    run_id: str,
    request: SampleRequest,
    user_id: str = Depends(verify_auth),
):
    """Generate samples from the model."""
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        # Get current step
        current_step = run["current_step"]
        
        # Get GPU config from run config (use single GPU for inference)
        # For sampling, we use single GPU regardless of training GPU count
        gpu_config = run["config"].get("gpu_config", "l40s:1")
        # Extract just the GPU type for single-GPU inference
        gpu_type = gpu_config.split(":")[0]
        inference_gpu_config = f"{gpu_type}:1"
        
        # Call Modal function
        # TODO: Inference GPU config is currently fixed in Modal function definition
        result = modal_sample().remote(
            user_id=user_id,
            run_id=run_id,
            prompts=request.prompts,
            step=current_step,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            return_logprobs=request.return_logprobs,
        )
        
        return SampleResponse(
            outputs=result["outputs"],
            logprobs=result.get("logprobs"),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/runs/{run_id}/save_state", response_model=SaveStateResponse)
async def save_state(
    run_id: str,
    request: SaveStateRequest,
    user_id: str = Depends(verify_auth),
):
    """Save model state."""
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        # Get current step
        current_step = run["current_step"]
        
        # Get GPU config from run config (use single GPU for saving)
        gpu_config = run["config"].get("gpu_config", "l40s:1")
        # Extract just the GPU type for single-GPU save operation
        gpu_type = gpu_config.split(":")[0]
        save_gpu_config = f"{gpu_type}:1"
        
        # Call Modal function
        # TODO: Save GPU config is currently fixed in Modal function definition
        result = modal_save_state().remote(
            user_id=user_id,
            run_id=run_id,
            step=current_step,
            mode=request.mode,
            push_to_hub=request.push_to_hub,
            hub_model_id=request.hub_model_id,
            training_metrics=None,  # Could fetch from registry if needed
        )
        
        # Update database with S3 artifact information
        if result.get("s3_uri"):
            logging.info(f"Artifact saved to S3: {result['s3_uri']}")
            
            # Record artifact in artifacts table
            registry = RunRegistry()
            artifact_recorded = registry.record_artifact(
                run_id=run_id,
                step=current_step,
                mode=request.mode,
                s3_uri=result["s3_uri"],
                manifest=result.get("manifest", {}),
                file_size_bytes=result.get("manifest", {}).get("total_size_bytes"),
            )
            
            if artifact_recorded:
                # Update run's S3 URI to point to latest artifact
                registry.update_run_s3_uri(
                    run_id=run_id,
                    s3_uri=result["s3_uri"],
                )
            else:
                logging.warning(f"Failed to record artifact in database for run {run_id}")
        
        return SaveStateResponse(
            artifact_uri=result["artifact_uri"],
            local_path=result.get("local_path"),
            checkpoint_path=result.get("local_path", result["artifact_uri"]),  # Backward compat
            s3_uri=result.get("s3_uri"),
            download_url=result.get("download_url"),
            download_expires_at=result.get("download_expires_at"),
            manifest=result.get("manifest"),
            pushed_to_hub=result["pushed_to_hub"],
            hub_model_id=result.get("hub_model_id"),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}/status", response_model=RunStatus)
async def get_run_status(
    run_id: str,
    user_id: str = Depends(verify_auth),
):
    """Get run status."""
    try:
        run = await get_authorized_run(run_id, user_id)
        return RunStatus(**run)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}/artifacts")
async def get_run_artifacts(
    run_id: str,
    user_id: str = Depends(verify_auth),
):
    """Get all artifacts for a run from S3.
    
    Returns list of saved artifacts with download URLs.
    """
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        # Query artifacts from database
        registry = RunRegistry()
        artifacts_data = registry.get_run_artifacts(run_id)
        
        # Generate fresh signed URLs for each artifact
        from datetime import datetime, timezone, timedelta
        from modal_runtime.s3_client import generate_signed_url
        
        artifacts = []
        for artifact in artifacts_data:
            try:
                # Generate fresh signed URL (1 hour expiration)
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
                
                # Track that this artifact was accessed (for download analytics)
                # Note: We only increment download_count when actually downloading, not just listing
                
            except Exception as e:
                logging.error(f"Failed to generate signed URL for artifact {artifact.get('id')}: {e}")
                # Still include artifact but without download URL
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
async def complete_run(
    run_id: str,
    request: Request,
    user_id: str = Depends(verify_auth),
):
    """Mark run as complete and deduct final credits.
    
    Called by client when training is finished.
    """
    try:
        # Verify run belongs to user
        run = await get_authorized_run(run_id, user_id)
        
        if run["status"] == "completed":
            return {"message": "Run already completed"}
        
        # Calculate final cost based on steps and GPU time
        # TODO: Implement actual cost calculation
        final_cost = 10.0  # Placeholder
        
        # Update run status
        run_registry.update_run(run_id, status="completed")
        
        # Deduct credits via Frontier Backend
        frontier_client = get_frontier_client()
        await frontier_client.deduct_credits(
            user_id=user_id,
            amount=final_cost,
            run_id=run_id,
            description=f"Training run {run_id}"
        )
        
        return {
            "success": True,
            "run_id": run_id,
            "status": "completed",
            "cost": final_cost
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs")
async def list_runs(user_id: str = Depends(verify_auth)):
    """List all runs for the authenticated user."""
    try:
        runs = run_registry.list_runs(user_id)
        return {"runs": runs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ERROR HANDLING
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    if exc.status_code >= 400:
        ip = request.client.host if request.client else "unknown"
        logging.warning(f"HTTP {exc.status_code}: {exc.detail} from {ip}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with sanitized error messages."""
    # Log the full exception server-side
    logging.exception(f"Unhandled exception: {exc}")
    
    # Return generic error to client (don't leak internal details)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

