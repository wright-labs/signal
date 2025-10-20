"""Tests for API endpoints."""

from tests.conftest import create_test_run


class TestPublicEndpoints:
    """Tests for public endpoints that don't require authentication."""

    def test_root_endpoint(self, test_client):
        """Test GET / endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Signal API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"

    def test_health_endpoint(self, test_client):
        """Test GET /health endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_list_models_endpoint(self, test_client):
        """Test GET /models endpoint."""
        response = test_client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0


class TestAuthentication:
    """Tests for authentication."""

    def test_missing_authorization_header(self, test_client):
        """Test that requests without auth are rejected."""
        response = test_client.post(
            "/runs",
            json={"base_model": "meta-llama/Llama-3.2-3B"},
        )

        assert response.status_code == 401
        assert "authorization" in response.json()["detail"].lower()

    def test_malformed_authorization_header(self, test_client, test_api_key):
        """Test that malformed auth headers are rejected."""
        # Missing "Bearer" prefix
        response = test_client.post(
            "/runs",
            headers={"Authorization": test_api_key},
            json={"base_model": "meta-llama/Llama-3.2-3B"},
        )

        assert response.status_code == 401

    def test_invalid_api_key(self, test_client):
        """Test that invalid API keys are rejected."""
        response = test_client.post(
            "/runs",
            headers={"Authorization": "Bearer sk-invalid-key-12345"},
            json={"base_model": "meta-llama/Llama-3.2-3B"},
        )

        assert response.status_code == 401
        assert "invalid api key" in response.json()["detail"].lower()

    def test_valid_api_key(self, test_client, test_api_key):
        """Test that valid API keys are accepted."""
        response = test_client.post(
            "/runs",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"base_model": "meta-llama/Llama-3.2-3B"},
        )

        # Should not be 401
        assert response.status_code != 401


class TestCreateRun:
    """Tests for POST /runs endpoint."""

    def test_create_run_success(self, test_client, test_api_key):
        """Test creating a run successfully."""
        response = test_client.post(
            "/runs",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={
                "base_model": "meta-llama/Llama-3.2-3B",
                "lora_r": 32,
                "lora_alpha": 64,
                "learning_rate": 3e-4,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "run_id" in data
        assert data["run_id"].startswith("run_")
        assert data["base_model"] == "meta-llama/Llama-3.2-3B"
        assert data["status"] == "active"
        assert "created_at" in data
        assert "config" in data

    def test_create_run_with_defaults(self, test_client, test_api_key):
        """Test creating a run with minimal config."""
        response = test_client.post(
            "/runs",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"base_model": "meta-llama/Llama-3.2-3B"},
        )

        assert response.status_code == 200
        data = response.json()

        # Check defaults are applied
        config = data["config"]
        assert config["lora_r"] == 32
        assert config["lora_alpha"] == 64
        assert config["learning_rate"] == 3e-4

    def test_create_run_unsupported_model(self, test_client, test_api_key):
        """Test creating a run with unsupported model."""
        response = test_client.post(
            "/runs",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"base_model": "unsupported/model"},
        )

        assert response.status_code == 400
        assert "not supported" in response.json()["detail"].lower()

    def test_create_run_custom_lora_config(self, test_client, test_api_key):
        """Test creating a run with custom LoRA config."""
        response = test_client.post(
            "/runs",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={
                "base_model": "meta-llama/Llama-3.2-3B",
                "lora_r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "v_proj"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        config = data["config"]
        assert config["lora_r"] == 64
        assert config["lora_alpha"] == 128
        assert config["lora_target_modules"] == ["q_proj", "v_proj"]


class TestListRuns:
    """Tests for GET /runs endpoint."""

    def test_list_runs(self, test_client, test_api_key, run_registry, test_user_id):
        """Test listing runs."""
        # Create a few runs
        create_test_run(run_registry, test_user_id)
        create_test_run(run_registry, test_user_id)

        response = test_client.get(
            "/runs",
            headers={"Authorization": f"Bearer {test_api_key}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert len(data["runs"]) == 2

    def test_list_runs_empty(self, test_client, test_api_key):
        """Test listing runs when user has none."""
        response = test_client.get(
            "/runs",
            headers={"Authorization": f"Bearer {test_api_key}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []

    def test_list_runs_isolation(self, test_client, api_key_manager, run_registry):
        """Test that users only see their own runs."""
        # Create runs for two different users
        user1_key = api_key_manager.generate_key("user1", "Key 1")
        user2_key = api_key_manager.generate_key("user2", "Key 2")

        create_test_run(run_registry, "user1")
        create_test_run(run_registry, "user2")

        # User 1 should only see their run
        response = test_client.get(
            "/runs",
            headers={"Authorization": f"Bearer {user1_key}"},
        )
        assert len(response.json()["runs"]) == 1
        assert response.json()["runs"][0]["user_id"] == "user1"

        # User 2 should only see their run
        response = test_client.get(
            "/runs",
            headers={"Authorization": f"Bearer {user2_key}"},
        )
        assert len(response.json()["runs"]) == 1
        assert response.json()["runs"][0]["user_id"] == "user2"


class TestGetRunStatus:
    """Tests for GET /runs/{run_id}/status endpoint."""

    def test_get_run_status(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test getting run status."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.get(
            f"/runs/{run_id}/status",
            headers={"Authorization": f"Bearer {test_api_key}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert data["status"] == "active"
        assert "current_step" in data
        assert "created_at" in data

    def test_get_run_status_not_found(self, test_client, test_api_key):
        """Test getting status of nonexistent run."""
        response = test_client.get(
            "/runs/nonexistent_run/status",
            headers={"Authorization": f"Bearer {test_api_key}"},
        )

        assert response.status_code == 404

    def test_get_run_status_unauthorized(
        self, test_client, api_key_manager, run_registry
    ):
        """Test that users can't see other users' runs."""
        # Create run for user1
        _ = api_key_manager.generate_key("user1", "Key")
        run_id = create_test_run(run_registry, "user1")

        # Try to access as user2
        user2_key = api_key_manager.generate_key("user2", "Key")
        response = test_client.get(
            f"/runs/{run_id}/status",
            headers={"Authorization": f"Bearer {user2_key}"},
        )

        assert response.status_code == 403


class TestGetRunMetrics:
    """Tests for GET /runs/{run_id}/metrics endpoint."""

    def test_get_run_metrics(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test getting run metrics."""
        run_id = create_test_run(run_registry, test_user_id)

        # Add some metrics
        run_registry.update_run(run_id, metrics={"loss": 2.5})
        run_registry.update_run(run_id, metrics={"loss": 2.3})

        response = test_client.get(
            f"/runs/{run_id}/metrics",
            headers={"Authorization": f"Bearer {test_api_key}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert len(data["metrics"]) == 2

    def test_get_run_metrics_empty(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test getting metrics for run with no metrics."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.get(
            f"/runs/{run_id}/metrics",
            headers={"Authorization": f"Bearer {test_api_key}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["metrics"] == []


class TestForwardBackward:
    """Tests for POST /runs/{run_id}/forward_backward endpoint."""

    def test_forward_backward_success(
        self, test_client, test_api_key, run_registry, test_user_id, sample_batch
    ):
        """Test forward-backward pass."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/forward_backward",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={
                "batch_data": sample_batch,
                "accumulate": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "loss" in data
        assert "step" in data
        assert "grad_norm" in data
        assert isinstance(data["loss"], float)

    def test_forward_backward_with_accumulate(
        self, test_client, test_api_key, run_registry, test_user_id, sample_batch
    ):
        """Test forward-backward with gradient accumulation."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/forward_backward",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={
                "batch_data": sample_batch,
                "accumulate": True,
            },
        )

        assert response.status_code == 200

    def test_forward_backward_updates_metrics(
        self, test_client, test_api_key, run_registry, test_user_id, sample_batch
    ):
        """Test that forward-backward updates run metrics."""
        run_id = create_test_run(run_registry, test_user_id)

        test_client.post(
            f"/runs/{run_id}/forward_backward",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"batch_data": sample_batch},
        )

        # Check that metrics were updated
        run = run_registry.get_run(run_id)
        assert len(run["metrics"]) > 0

    def test_forward_backward_unauthorized(
        self, test_client, api_key_manager, run_registry, sample_batch
    ):
        """Test forward-backward with unauthorized access."""
        _ = api_key_manager.generate_key("user1", "Key")
        run_id = create_test_run(run_registry, "user1")

        user2_key = api_key_manager.generate_key("user2", "Key")
        response = test_client.post(
            f"/runs/{run_id}/forward_backward",
            headers={"Authorization": f"Bearer {user2_key}"},
            json={"batch_data": sample_batch},
        )

        assert response.status_code == 403


class TestOptimStep:
    """Tests for POST /runs/{run_id}/optim_step endpoint."""

    def test_optim_step_success(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test optimizer step."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/optim_step",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={},
        )

        assert response.status_code == 200
        data = response.json()

        assert "step" in data
        assert "learning_rate" in data
        assert data["step"] > 0

    def test_optim_step_with_custom_lr(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test optimizer step with custom learning rate."""
        run_id = create_test_run(run_registry, test_user_id)

        custom_lr = 5e-4
        response = test_client.post(
            f"/runs/{run_id}/optim_step",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"learning_rate": custom_lr},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["learning_rate"] == custom_lr

    def test_optim_step_increments_step(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test that optim_step increments the step counter."""
        run_id = create_test_run(run_registry, test_user_id)

        # Get initial step
        run = run_registry.get_run(run_id)
        initial_step = run["current_step"]

        # Perform optim step
        test_client.post(
            f"/runs/{run_id}/optim_step",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={},
        )

        # Check step was incremented
        run = run_registry.get_run(run_id)
        assert run["current_step"] == initial_step + 1


class TestSample:
    """Tests for POST /runs/{run_id}/sample endpoint."""

    def test_sample_success(
        self, test_client, test_api_key, run_registry, test_user_id, sample_prompts
    ):
        """Test sampling from model."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/sample",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={
                "prompts": sample_prompts,
                "max_tokens": 256,
                "temperature": 0.8,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "outputs" in data
        assert len(data["outputs"]) == len(sample_prompts)
        assert all(isinstance(output, str) for output in data["outputs"])

    def test_sample_with_defaults(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test sampling with default parameters."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/sample",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"prompts": ["Test prompt"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["outputs"]) == 1

    def test_sample_with_logprobs(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test sampling with logprobs."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/sample",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={
                "prompts": ["Test"],
                "return_logprobs": True,
            },
        )

        assert response.status_code == 200
        # Note: Our mock doesn't actually return logprobs, but the field should exist
        data = response.json()
        assert "logprobs" in data


class TestSaveState:
    """Tests for POST /runs/{run_id}/save_state endpoint."""

    def test_save_state_adapter(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test saving LoRA adapter."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/save_state",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"mode": "adapter"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "artifact_uri" in data
        assert "checkpoint_path" in data
        assert data["pushed_to_hub"] is False

    def test_save_state_merged(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test saving merged model."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/save_state",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"mode": "merged"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "checkpoint_path" in data

    def test_save_state_push_to_hub(
        self, test_client, test_api_key, run_registry, test_user_id
    ):
        """Test saving and pushing to HuggingFace Hub."""
        run_id = create_test_run(run_registry, test_user_id)

        response = test_client.post(
            f"/runs/{run_id}/save_state",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={
                "mode": "adapter",
                "push_to_hub": True,
                "hub_model_id": "test-user/test-model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["pushed_to_hub"] is True
        assert data["hub_model_id"] == "test-user/test-model"


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_for_nonexistent_run(self, test_client, test_api_key):
        """Test that nonexistent runs return 404."""
        response = test_client.post(
            "/runs/nonexistent_run_123/forward_backward",
            headers={"Authorization": f"Bearer {test_api_key}"},
            json={"batch_data": [{"text": "test"}]},
        )

        assert response.status_code == 404

    def test_invalid_json_body(self, test_client, test_api_key):
        """Test that invalid JSON returns 422."""
        response = test_client.post(
            "/runs",
            headers={
                "Authorization": f"Bearer {test_api_key}",
                "Content-Type": "application/json",
            },
            content="invalid json {",
        )

        assert response.status_code == 422
