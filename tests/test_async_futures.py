"""Tests for async execution with futures."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_async_forward_backward_with_future():
    """Test forward_backward with return_future=True returns APIFuture."""
    from client.rewardsignal import AsyncSignalClient
    from client.rewardsignal.futures import APIFuture
    
    # Mock the API response
    mock_response = {
        "future_id": "test-future-123",
        "status": "pending",
    }
    
    with patch.object(AsyncSignalClient, '_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        
        async with AsyncSignalClient(api_key="test-key", base_url="http://localhost:8000") as client:
            result = await client.forward_backward(
                run_id="test-run",
                batch=[{"text": "Hello world"}],
                return_future=True,
            )
            
            # Should return APIFuture
            assert isinstance(result, APIFuture)
            assert result.future_id == "test-future-123"
            
            # Verify request was made with correct params
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "/forward_backward?return_future=true" in call_args[0][1]


@pytest.mark.asyncio
async def test_async_optim_step_with_future():
    """Test optim_step with return_future=True returns APIFuture."""
    from client.rewardsignal import AsyncSignalClient
    from client.rewardsignal.futures import APIFuture
    
    mock_response = {
        "future_id": "test-future-456",
        "status": "pending",
    }
    
    with patch.object(AsyncSignalClient, '_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        
        async with AsyncSignalClient(api_key="test-key", base_url="http://localhost:8000") as client:
            result = await client.optim_step(
                run_id="test-run",
                learning_rate=3e-4,
                return_future=True,
            )
            
            assert isinstance(result, APIFuture)
            assert result.future_id == "test-future-456"


@pytest.mark.asyncio
async def test_async_sample_with_future():
    """Test sample with return_future=True returns APIFuture."""
    from client.rewardsignal import AsyncSignalClient
    from client.rewardsignal.futures import APIFuture
    
    mock_response = {
        "future_id": "test-future-789",
        "status": "pending",
    }
    
    with patch.object(AsyncSignalClient, '_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        
        async with AsyncSignalClient(api_key="test-key", base_url="http://localhost:8000") as client:
            result = await client.sample(
                run_id="test-run",
                prompts=["Hello"],
                return_future=True,
            )
            
            assert isinstance(result, APIFuture)
            assert result.future_id == "test-future-789"


@pytest.mark.asyncio
async def test_async_save_state_with_future():
    """Test save_state with return_future=True returns APIFuture."""
    from client.rewardsignal import AsyncSignalClient
    from client.rewardsignal.futures import APIFuture
    
    mock_response = {
        "future_id": "test-future-abc",
        "status": "pending",
    }
    
    with patch.object(AsyncSignalClient, '_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        
        async with AsyncSignalClient(api_key="test-key", base_url="http://localhost:8000") as client:
            result = await client.save_state(
                run_id="test-run",
                mode="adapter",
                return_future=True,
            )
            
            assert isinstance(result, APIFuture)
            assert result.future_id == "test-future-abc"


@pytest.mark.asyncio
async def test_future_polling():
    """Test APIFuture polls until completion."""
    from client.rewardsignal.futures import APIFuture
    from client.rewardsignal import AsyncSignalClient
    
    # Mock client
    mock_client = AsyncMock(spec=AsyncSignalClient)
    
    # Simulate polling: pending -> pending -> completed
    mock_client._request.side_effect = [
        {"status": "pending"},
        {"status": "pending"},
        {"status": "completed", "result": {"loss": 0.5, "step": 1}},
    ]
    
    future = APIFuture(client=mock_client, future_id="test-123")
    
    # Await the future
    result = await future
    
    # Should poll 3 times and return result
    assert mock_client._request.call_count == 3
    assert result["loss"] == 0.5
    assert result["step"] == 1


@pytest.mark.asyncio
async def test_future_error_handling():
    """Test APIFuture handles errors."""
    from client.rewardsignal.futures import APIFuture
    from client.rewardsignal import AsyncSignalClient
    
    mock_client = AsyncMock(spec=AsyncSignalClient)
    
    # Simulate error response
    mock_client._request.return_value = {
        "status": "failed",
        "error": "Out of memory",
    }
    
    future = APIFuture(client=mock_client, future_id="test-456")
    
    # Should raise exception
    with pytest.raises(Exception, match="Out of memory"):
        await future


@pytest.mark.asyncio
async def test_synchronous_execution_still_works():
    """Test that synchronous execution (return_future=False) still works."""
    from client.rewardsignal import AsyncSignalClient
    
    mock_response = {
        "loss": 0.5,
        "step": 1,
        "grad_norm": 0.1,
    }
    
    with patch.object(AsyncSignalClient, '_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        
        async with AsyncSignalClient(api_key="test-key", base_url="http://localhost:8000") as client:
            result = await client.forward_backward(
                run_id="test-run",
                batch=[{"text": "Hello world"}],
                return_future=False,  # Explicit sync
            )
            
            # Should return result directly (not a future)
            assert isinstance(result, dict)
            assert result["loss"] == 0.5
            assert result["step"] == 1


@pytest.mark.asyncio
async def test_future_group():
    """Test FutureGroup for batch operations."""
    from client.rewardsignal.futures import APIFuture, FutureGroup
    from client.rewardsignal import AsyncSignalClient
    
    mock_client = AsyncMock(spec=AsyncSignalClient)
    
    # Create group
    group = FutureGroup()
    
    # Add multiple futures
    for i in range(3):
        future = APIFuture(client=mock_client, future_id=f"test-{i}")
        group.add(future)
    
    assert len(group) == 3
    
    # Mock responses
    mock_client._request.side_effect = [
        {"status": "completed", "result": {"loss": 0.1}},
        {"status": "completed", "result": {"loss": 0.2}},
        {"status": "completed", "result": {"loss": 0.3}},
    ]
    
    # Wait for all
    results = await group.wait_all()
    
    assert len(results) == 3
    assert results[0]["loss"] == 0.1
    assert results[1]["loss"] == 0.2
    assert results[2]["loss"] == 0.3

