"""End-to-end integration tests."""
import pytest
from client.python_client import SignalClient, SignalRun
from tests.conftest import create_test_run


class TestBasicTrainingWorkflow:
    """Tests for basic training workflow."""
    
    def test_complete_training_flow(self, test_client, test_api_key, sample_batch, sample_prompts):
        """Test a complete training workflow."""
        # Create client
        client = SignalClient(
            api_key=test_api_key,
            base_url="http://testserver",
        )
        
        # Override session to use test_client
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        # Step 1: List available models
        models = client.list_models()
        assert len(models) > 0
        assert "meta-llama/Llama-3.2-3B" in models
        
        # Step 2: Create a training run
        run = client.create_run(
            base_model="meta-llama/Llama-3.2-3B",
            lora_r=32,
            lora_alpha=64,
            learning_rate=3e-4,
        )
        
        assert isinstance(run, SignalRun)
        assert run.run_id is not None
        
        # Step 3: Perform forward-backward pass
        fb_result = run.forward_backward(batch=sample_batch)
        assert "loss" in fb_result
        assert fb_result["loss"] > 0
        
        # Step 4: Apply optimizer step
        optim_result = run.optim_step()
        assert "step" in optim_result
        assert optim_result["step"] > 0
        
        # Step 5: Sample from model
        sample_result = run.sample(prompts=sample_prompts, temperature=0.7)
        assert "outputs" in sample_result
        assert len(sample_result["outputs"]) == len(sample_prompts)
        
        # Step 6: Save adapter
        save_result = run.save_state(mode="adapter")
        assert "checkpoint_path" in save_result
        
        # Step 7: Check status
        status = run.get_status()
        assert status["run_id"] == run.run_id
        assert status["current_step"] > 0
    
    def test_training_multiple_steps(self, test_client, test_api_key, sample_batch):
        """Test training for multiple steps."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Train for 5 steps
        losses = []
        for i in range(5):
            fb_result = run.forward_backward(batch=sample_batch)
            losses.append(fb_result["loss"])
            
            optim_result = run.optim_step()
            assert optim_result["step"] == i + 1
        
        # Verify we have 5 loss values
        assert len(losses) == 5
        
        # Check final status
        status = run.get_status()
        assert status["current_step"] == 5
    
    def test_sample_at_intervals(self, test_client, test_api_key, sample_batch):
        """Test sampling at intervals during training."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Train for 10 steps, sample every 5 steps
        for i in range(10):
            run.forward_backward(batch=sample_batch)
            run.optim_step()
            
            if (i + 1) % 5 == 0:
                sample_result = run.sample(prompts=["Test prompt"])
                assert len(sample_result["outputs"]) == 1


class TestGradientAccumulation:
    """Tests for gradient accumulation workflow."""
    
    def test_gradient_accumulation_workflow(self, test_client, test_api_key, sample_batch):
        """Test gradient accumulation across multiple batches."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Accumulate gradients over 3 batches
        run.forward_backward(batch=sample_batch, accumulate=False)  # First batch
        run.forward_backward(batch=sample_batch, accumulate=True)   # Accumulate
        run.forward_backward(batch=sample_batch, accumulate=True)   # Accumulate
        
        # Apply accumulated gradients
        optim_result = run.optim_step()
        assert optim_result["step"] == 1
    
    def test_alternating_accumulation_and_steps(self, test_client, test_api_key, sample_batch):
        """Test alternating between accumulation and optimizer steps."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Do 3 rounds of accumulation + step
        for round_num in range(3):
            # Accumulate over 2 batches
            run.forward_backward(batch=sample_batch, accumulate=False)
            run.forward_backward(batch=sample_batch, accumulate=True)
            
            # Apply step
            result = run.optim_step()
            assert result["step"] == round_num + 1


class TestMultipleRuns:
    """Tests for managing multiple runs."""
    
    def test_create_multiple_runs(self, test_client, test_api_key):
        """Test creating multiple runs for same user."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        # Create 3 runs
        runs = []
        for i in range(3):
            run = client.create_run(
                base_model="meta-llama/Llama-3.2-3B",
                lora_r=32 + i * 16,
            )
            runs.append(run)
        
        # Verify all runs are different
        run_ids = [r.run_id for r in runs]
        assert len(set(run_ids)) == 3
        
        # List runs
        all_runs = client.list_runs()
        assert len(all_runs) >= 3
    
    def test_runs_are_isolated(self, test_client, api_key_manager, sample_batch):
        """Test that runs are isolated between users."""
        # Create two clients for different users
        user1_key = api_key_manager.generate_key("user1", "Key 1")
        user2_key = api_key_manager.generate_key("user2", "Key 2")
        
        client1 = SignalClient(api_key=user1_key, base_url="http://testserver")
        client1.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        client2 = SignalClient(api_key=user2_key, base_url="http://testserver")
        client2.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        # Create runs for each user
        run1 = client1.create_run(base_model="meta-llama/Llama-3.2-3B")
        run2 = client2.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Train both runs
        run1.forward_backward(batch=sample_batch)
        run1.optim_step()
        
        run2.forward_backward(batch=sample_batch)
        run2.optim_step()
        
        # Each user should only see their own runs
        user1_runs = client1.list_runs()
        user2_runs = client2.list_runs()
        
        user1_run_ids = [r["run_id"] for r in user1_runs]
        user2_run_ids = [r["run_id"] for r in user2_runs]
        
        assert run1.run_id in user1_run_ids
        assert run1.run_id not in user2_run_ids
        
        assert run2.run_id in user2_run_ids
        assert run2.run_id not in user1_run_ids


class TestDifferentDataFormats:
    """Tests for different data formats."""
    
    def test_plain_text_format(self, test_client, test_api_key):
        """Test training with plain text format."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Use text format
        batch = [
            {"text": "Example 1"},
            {"text": "Example 2"},
        ]
        
        result = run.forward_backward(batch=batch)
        assert "loss" in result
    
    def test_messages_format(self, test_client, test_api_key, sample_messages_batch):
        """Test training with messages format (chat templates)."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        result = run.forward_backward(batch=sample_messages_batch)
        assert "loss" in result


class TestCustomConfigurations:
    """Tests for custom configurations."""
    
    def test_custom_lora_config(self, test_client, test_api_key):
        """Test training with custom LoRA configuration."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(
            base_model="meta-llama/Llama-3.2-3B",
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        
        assert run.config["lora_r"] == 64
        assert run.config["lora_alpha"] == 128
        assert run.config["lora_target_modules"] == ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def test_custom_learning_rate(self, test_client, test_api_key, sample_batch):
        """Test using custom learning rate."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(
            base_model="meta-llama/Llama-3.2-3B",
            learning_rate=5e-4,
        )
        
        run.forward_backward(batch=sample_batch)
        
        # Use default learning rate
        result1 = run.optim_step()
        assert result1["learning_rate"] == 5e-4
        
        # Override learning rate for one step
        run.forward_backward(batch=sample_batch)
        result2 = run.optim_step(learning_rate=1e-4)
        assert result2["learning_rate"] == 1e-4
    
    def test_different_sampling_parameters(self, test_client, test_api_key):
        """Test sampling with different parameters."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Greedy sampling (temperature=0)
        result1 = run.sample(prompts=["Test"], temperature=0.0, max_tokens=50)
        assert len(result1["outputs"]) == 1
        
        # High temperature sampling
        result2 = run.sample(prompts=["Test"], temperature=1.5, top_p=0.95, max_tokens=100)
        assert len(result2["outputs"]) == 1
        
        # With logprobs
        result3 = run.sample(prompts=["Test"], return_logprobs=True)
        assert "logprobs" in result3


class TestModelSaving:
    """Tests for model saving workflows."""
    
    def test_save_adapter_and_merged(self, test_client, test_api_key, sample_batch):
        """Test saving both adapter and merged models."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Train for a few steps
        for _ in range(3):
            run.forward_backward(batch=sample_batch)
            run.optim_step()
        
        # Save adapter
        adapter_result = run.save_state(mode="adapter")
        assert "checkpoint_path" in adapter_result
        assert adapter_result["pushed_to_hub"] is False
        
        # Save merged model
        merged_result = run.save_state(mode="merged")
        assert "checkpoint_path" in merged_result
    
    def test_push_to_hub(self, test_client, test_api_key, sample_batch):
        """Test pushing model to HuggingFace Hub."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Train briefly
        run.forward_backward(batch=sample_batch)
        run.optim_step()
        
        # Save and push to hub
        result = run.save_state(
            mode="adapter",
            push_to_hub=True,
            hub_model_id="test-user/test-model",
        )
        
        assert result["pushed_to_hub"] is True
        assert result["hub_model_id"] == "test-user/test-model"


class TestMetricsTracking:
    """Tests for metrics tracking."""
    
    def test_metrics_are_tracked(self, test_client, test_api_key, sample_batch):
        """Test that training metrics are tracked."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Train for several steps
        for _ in range(5):
            run.forward_backward(batch=sample_batch)
            run.optim_step()
        
        # Get metrics
        metrics = run.get_metrics()
        
        assert "metrics" in metrics
        assert len(metrics["metrics"]) > 0
    
    def test_metrics_history(self, test_client, test_api_key, sample_batch):
        """Test that metrics history is preserved."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
        
        # Train for 3 steps, recording losses
        losses = []
        for _ in range(3):
            fb_result = run.forward_backward(batch=sample_batch)
            losses.append(fb_result["loss"])
            run.optim_step()
        
        # Get metrics and verify history
        metrics = run.get_metrics()
        assert len(metrics["metrics"]) == 3


class TestErrorScenarios:
    """Tests for error handling in workflows."""
    
    def test_invalid_model_name(self, test_client, test_api_key):
        """Test creating run with invalid model name."""
        client = SignalClient(api_key=test_api_key, base_url="http://testserver")
        client.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        with pytest.raises(Exception) as exc_info:
            client.create_run(base_model="nonexistent/model")
        
        assert "not supported" in str(exc_info.value).lower() or "400" in str(exc_info.value)
    
    def test_unauthorized_run_access(self, test_client, api_key_manager, run_registry, sample_batch):
        """Test that users can't access other users' runs."""
        # Create run for user1
        user1_key = api_key_manager.generate_key("user1", "Key")
        run_id = create_test_run(run_registry, "user1")
        
        # Try to access as user2
        user2_key = api_key_manager.generate_key("user2", "Key")
        client2 = SignalClient(api_key=user2_key, base_url="http://testserver")
        client2.session.request = lambda method, url, **kwargs: test_client.request(
            method, url.replace("http://testserver", ""), **kwargs
        )
        
        # Try to do forward-backward on user1's run
        response = test_client.post(
            f"/runs/{run_id}/forward_backward",
            headers={"Authorization": f"Bearer {user2_key}"},
            json={"batch_data": sample_batch},
        )
        
        assert response.status_code == 403

