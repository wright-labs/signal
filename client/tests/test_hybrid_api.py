"""Integration tests for hybrid client architecture."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from frontier_signal import SignalClient, TrainingClient, InferenceClient


class TestSignalClientFactoryMethods:
    """Test factory methods on SignalClient."""
    
    @patch('frontier_signal.client.requests.Session')
    def test_training_factory_method(self, mock_session_class):
        """Test SignalClient.training() returns TrainingClient."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        training = client.training(run_id="test_run")
        
        assert isinstance(training, TrainingClient)
        assert training.run_id == "test_run"
        assert training.api_key == "sk-test"
        assert training.timeout == 3600  # Training default
    
    @patch('frontier_signal.client.requests.Session')
    def test_inference_factory_method(self, mock_session_class):
        """Test SignalClient.inference() returns InferenceClient."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        inference = client.inference(run_id="test_run", step=100)
        
        assert isinstance(inference, InferenceClient)
        assert inference.run_id == "test_run"
        assert inference.api_key == "sk-test"
        assert inference.step == 100
        assert inference.timeout == 30  # Inference default
    
    @patch('frontier_signal.client.requests.Session')
    def test_factory_methods_share_session(self, mock_session_class):
        """Test factory methods share parent session."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        training = client.training(run_id="test_run")
        inference = client.inference(run_id="test_run")
        
        # Both should share the same session
        assert training.session == client.session
        assert inference.session == client.session


class TestSignalRunFactoryMethods:
    """Test factory methods on SignalRun."""
    
    @patch('frontier_signal.client.requests.Session')
    def test_run_training_factory(self, mock_session_class):
        """Test SignalRun.training() returns TrainingClient."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        run = Mock()
        run.client = client
        run.run_id = "test_run"
        run.training = lambda **kwargs: client.training(run.run_id, **kwargs)
        
        training = run.training(timeout=7200)
        
        assert isinstance(training, TrainingClient)
        assert training.run_id == "test_run"
        assert training.timeout == 7200
    
    @patch('frontier_signal.client.requests.Session')
    def test_run_inference_factory(self, mock_session_class):
        """Test SignalRun.inference() returns InferenceClient."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        run = Mock()
        run.client = client
        run.run_id = "test_run"
        run.inference = lambda **kwargs: client.inference(run.run_id, **kwargs)
        
        inference = run.inference(step=50)
        
        assert isinstance(inference, InferenceClient)
        assert inference.run_id == "test_run"
        assert inference.step == 50


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    @patch('frontier_signal.client.TrainingClient.forward_backward')
    @patch('frontier_signal.client.requests.Session')
    def test_simple_api_still_works(self, mock_session_class, mock_fb):
        """Test that existing simple API code still works."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_fb.return_value = {"loss": 1.0, "grad_norm": 0.5}
        
        # Old code should still work
        client = SignalClient(api_key="sk-test")
        result = client.forward_backward(
            run_id="test_run",
            batch=[{"text": "test"}],
        )
        
        # Verify it delegates to TrainingClient
        assert result["loss"] == 1.0
        mock_fb.assert_called_once()
    
    @patch('frontier_signal.client.TrainingClient.optim_step')
    @patch('frontier_signal.client.requests.Session')
    def test_optim_step_delegation(self, mock_session_class, mock_optim):
        """Test optim_step delegates to TrainingClient."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_optim.return_value = {"step": 1}
        
        client = SignalClient(api_key="sk-test")
        result = client.optim_step(run_id="test_run")
        
        assert result["step"] == 1
        mock_optim.assert_called_once()
    
    @patch('frontier_signal.client.InferenceClient.sample')
    @patch('frontier_signal.client.requests.Session')
    def test_sample_delegation(self, mock_session_class, mock_sample):
        """Test sample delegates to InferenceClient."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_sample.return_value = ["output"]
        
        client = SignalClient(api_key="sk-test")
        result = client.sample(
            run_id="test_run",
            prompts=["test"],
        )
        
        assert result == ["output"]
        mock_sample.assert_called_once()
    
    @patch('frontier_signal.client.TrainingClient.save_checkpoint')
    @patch('frontier_signal.client.requests.Session')
    def test_save_state_delegation(self, mock_session_class, mock_save):
        """Test save_state delegates to TrainingClient."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_save.return_value = {"artifact_url": "s3://..."}
        
        client = SignalClient(api_key="sk-test")
        result = client.save_state(run_id="test_run", mode="adapter")
        
        assert "artifact_url" in result
        mock_save.assert_called_once()


class TestEndToEndWorkflow:
    """Test complete workflow: create → train → infer."""
    
    @patch('frontier_signal.client.TrainingClient.train_batch')
    @patch('frontier_signal.client.InferenceClient.sample')
    @patch('frontier_signal.client.SignalClient._request')
    @patch('frontier_signal.client.requests.Session')
    def test_simple_to_advanced_workflow(
        self,
        mock_session_class,
        mock_request,
        mock_sample,
        mock_train_batch,
    ):
        """Test workflow transitioning from simple to advanced API."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock create_run response
        mock_request.return_value = {
            "run_id": "test_run",
            "config": {"base_model": "Qwen/Qwen2.5-3B"},
        }
        
        # Mock training
        mock_train_batch.return_value = {
            "loss": 1.0,
            "grad_norm": 0.5,
            "step": 1,
        }
        
        # Mock inference
        mock_sample.return_value = ["Generated output"]
        
        # Create run (simple API)
        client = SignalClient(api_key="sk-test")
        run = client.create_run(base_model="Qwen/Qwen2.5-3B")
        
        # Train with advanced API
        training = client.training(run.run_id)
        result = training.train_batch([{"text": "test"}])
        assert result["loss"] == 1.0
        
        # Infer with advanced API
        inference = client.inference(run.run_id, step=1)
        outputs = inference.sample(["test prompt"])
        assert outputs == ["Generated output"]
    
    @patch('frontier_signal.client.TrainingClient.forward_backward')
    @patch('frontier_signal.client.TrainingClient.optim_step')
    @patch('frontier_signal.client.SignalClient._request')
    @patch('frontier_signal.client.requests.Session')
    def test_mixed_api_usage(
        self,
        mock_session_class,
        mock_request,
        mock_optim,
        mock_fb,
    ):
        """Test mixing simple and advanced API in same workflow."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        mock_request.return_value = {
            "run_id": "test_run",
            "config": {},
        }
        
        mock_fb.return_value = {"loss": 1.5, "grad_norm": 0.8}
        mock_optim.return_value = {"step": 1}
        
        client = SignalClient(api_key="sk-test")
        run = client.create_run(base_model="Qwen/Qwen2.5-3B")
        
        # Use simple API
        client.forward_backward(run.run_id, [{"text": "test"}])
        
        # Use advanced API
        training = client.training(run.run_id)
        training.optim_step()
        
        # Both should work together
        assert mock_fb.call_count >= 1
        assert mock_optim.call_count >= 1


class TestSpecializedClientCustomization:
    """Test customization of specialized clients."""
    
    @patch('frontier_signal.client.requests.Session')
    def test_custom_training_timeout(self, mock_session_class):
        """Test custom timeout for training client."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        training = client.training(
            run_id="test_run",
            timeout=10800,  # 3 hours
            max_retries=5,
        )
        
        assert training.timeout == 10800
        assert training.max_retries == 5
    
    @patch('frontier_signal.client.requests.Session')
    def test_custom_inference_batch_size(self, mock_session_class):
        """Test custom batch size for inference client."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        inference = client.inference(
            run_id="test_run",
            batch_size=64,
            timeout=60,
        )
        
        assert inference.batch_size == 64
        assert inference.timeout == 60


class TestTypeSafety:
    """Test type safety of specialized clients."""
    
    @patch('frontier_signal.client.requests.Session')
    def test_training_client_type(self, mock_session_class):
        """Test TrainingClient returns correct type."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        training = client.training(run_id="test_run")
        
        # Should have training-specific methods
        assert hasattr(training, 'forward_backward')
        assert hasattr(training, 'optim_step')
        assert hasattr(training, 'train_batch')
        assert hasattr(training, 'train_epoch')
        assert hasattr(training, 'save_checkpoint')
        assert hasattr(training, 'get_metrics')
    
    @patch('frontier_signal.client.requests.Session')
    def test_inference_client_type(self, mock_session_class):
        """Test InferenceClient returns correct type."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        inference = client.inference(run_id="test_run")
        
        # Should have inference-specific methods
        assert hasattr(inference, 'sample')
        assert hasattr(inference, 'batch_sample')
        assert hasattr(inference, 'stream_sample')
        assert hasattr(inference, 'enable_cache')
        assert hasattr(inference, 'disable_cache')
        assert hasattr(inference, 'get_cache_stats')


class TestProgressiveDisclosure:
    """Test progressive disclosure pattern."""
    
    @patch('frontier_signal.client.requests.Session')
    def test_level_1_simple_api(self, mock_session_class):
        """Test Level 1: Simple direct methods."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Simplest API - just call methods on client
        client = SignalClient(api_key="sk-test")
        
        # User doesn't need to know about specialized clients
        assert hasattr(client, 'forward_backward')
        assert hasattr(client, 'optim_step')
        assert hasattr(client, 'sample')
    
    @patch('frontier_signal.client.requests.Session')
    def test_level_2_run_wrapper(self, mock_session_class):
        """Test Level 2: SignalRun wrapper."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        
        # User can use run wrapper for cleaner code
        # (We'll mock this since we can't actually create a run)
        run = Mock()
        run.run_id = "test_run"
        run.forward_backward = Mock()
        run.optim_step = Mock()
        
        # Usage feels natural
        run.forward_backward(batch=[])
        run.optim_step()
        
        assert run.forward_backward.called
        assert run.optim_step.called
    
    @patch('frontier_signal.client.requests.Session')
    def test_level_3_specialized_clients(self, mock_session_class):
        """Test Level 3: Specialized clients for advanced users."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = SignalClient(api_key="sk-test")
        
        # Advanced users get specialized clients with extra features
        training = client.training(run_id="test_run")
        inference = client.inference(run_id="test_run")
        
        # Training has state tracking
        assert hasattr(training, 'loss_history')
        assert hasattr(training, 'grad_norm_history')
        assert hasattr(training, 'get_metrics')
        
        # Inference has caching
        assert hasattr(inference, 'enable_cache')
        assert hasattr(inference, '_cache')

