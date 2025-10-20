"""OpenAI-compatible API endpoints for Verifiers integration."""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional, Literal, Union
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import time
import uuid
import logging

from api.auth import verify_auth
from api.registry import RunRegistry
import tiktoken

logger = logging.getLogger(__name__)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def count_tokens_messages(messages: list, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in chat messages using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(str(message.get("content", ""))))
    return total_tokens

try:
    from modal_runtime import sample as modal_sample
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal_sample = None


router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])
run_registry = RunRegistry()

class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""
    model: str = Field(..., description="Model ID (format: signal-run-{run_id})")
    messages: List[ChatMessage]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, le=4096)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=10)
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class CompletionRequest(BaseModel):
    """OpenAI completion request."""
    model: str
    prompt: Union[str, List[str]]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, le=4096)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=10)
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    """Completion choice."""
    text: str
    index: int
    finish_reason: str


class CompletionResponse(BaseModel):
    """OpenAI completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: ChatCompletionUsage


def extract_run_id_from_model(model: str) -> str:
    """
    Extract Signal run ID from model identifier.
    
    Expected format: "signal-run-{run_id}" or just "{run_id}"
    """
    if model.startswith("signal-run-"):
        return model.replace("signal-run-", "")
    elif "-" in model:
        parts = model.split("-")
        if len(parts) >= 2:
            return "-".join(parts[1:])
    
    # Assume the entire string is a run_id
    return model


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    user_id: str = Depends(verify_auth),
) -> Dict[str, Any]:
    """
    OpenAI-compatible chat completions endpoint."""
    try:
        # Extract run ID from model field
        run_id = extract_run_id_from_model(request.model)
        
        # Verify run belongs to user
        run = run_registry.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Convert messages to prompt
        # Format: system message + conversation history
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt = "\n".join(prompt_parts)
        if not prompt.endswith("Assistant:"):
            prompt += "\nAssistant:"
        
        # Call Signal's sample function
        if not MODAL_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Modal runtime not available. Please ensure Modal is properly configured."
            )
        
        result = modal_sample.remote(
            user_id=user_id,
            run_id=run_id,
            prompts=[prompt] * request.n,  # Generate n completions
            step=run["current_step"],
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            return_logprobs=False,
        )
        
        # Convert to OpenAI format
        choices = []
        for i, output in enumerate(result["outputs"]):
            # Extract just the assistant's response
            # (remove the prompt part)
            response_text = output.replace(prompt, "").strip()
            
            choices.append(ChatCompletionChoice(
                index=i,
                message=ChatMessage(
                    role="assistant",
                    content=response_text
                ),
                finish_reason="stop"
            ))
        
        # Count tokens accurately using tiktoken
        messages_for_counting = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt_tokens = count_tokens_messages(messages_for_counting, model=request.model)
        completion_tokens = sum(count_tokens(c.message.content, model=request.model) for c in choices)
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed")


@router.post("/completions")
async def completions(
    request: CompletionRequest,
    user_id: str = Depends(verify_auth),
) -> Dict[str, Any]:
    """
    OpenAI-compatible completions endpoint."""
    try:
        # Extract run ID from model field
        run_id = extract_run_id_from_model(request.model)
        
        # Verify run belongs to user
        run = run_registry.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Handle prompt (can be string or list)
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        
        # Expand for n completions per prompt
        all_prompts = []
        for prompt in prompts:
            all_prompts.extend([prompt] * request.n)
        
        # Call Signal's sample function
        if not MODAL_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Modal runtime not available. Please ensure Modal is properly configured."
            )
        
        result = modal_sample.remote(
            user_id=user_id,
            run_id=run_id,
            prompts=all_prompts,
            step=run["current_step"],
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            return_logprobs=False,
        )
        
        # Convert to OpenAI format
        choices = []
        for i, output in enumerate(result["outputs"]):
            # Extract just the completion (remove prompt)
            original_prompt = all_prompts[i]
            completion_text = output.replace(original_prompt, "").strip()
            
            choices.append(CompletionChoice(
                text=completion_text,
                index=i,
                finish_reason="stop"
            ))
        
        # Count tokens accurately using tiktoken
        prompt_tokens = sum(count_tokens(p, model=request.model) for p in all_prompts)
        completion_tokens = sum(count_tokens(c.text, model=request.model) for c in choices)
        
        response = CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed")
