"""Tests for Python client SDK."""

import pytest
from unittest.mock import Mock, patch

from client.python_client import SignalClient, SignalRun


class TestSignalClientInitialization:
    """Tests for SignalClient initialization."""

    def test_client_initialization(self):
        """Test creating a SignalClient."""
        client = SignalClient(
            api_key="sk-test-key",
            base_url="http://localhost:8000",
        )

        assert client.api_key == "sk-test-key"
        assert client.base_url == "http://localhost:8000"
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == "Bearer sk-test-key"

    def test_client_strips_trailing_slash(self):
        """Test that base_url trailing slash is removed."""
        client = SignalClient(
            api_key="sk-test",
            base_url="http://localhost:8000/",
        )

        assert client.base_url == "http://localhost:8000"

    def test_client_sets_headers(self):
        """Test that client sets correct headers."""
        client = SignalClient(api_key="sk-test")

        assert client.session.headers["Authorization"] == "Bearer sk-test"
        assert client.session.headers["Content-Type"] == "application/json"


class TestSignalClientModels:
    """Tests for model operations."""

    def test_list_models(self, test_client, test_api_key):
        """Test listing available models."""
        # Use the test_client fixture to get a working server
        client = SignalClient(
            api_key=test_api_key,
            base_url="http://testserver",
        )

        # Override the session to use test_client
        client.session.get = lambda url, **kwargs: test_client.get(
            url.replace("http://testserver", "")
        )

        models = client.list_models()

        assert isinstance(models, list)
        assert len(models) > 0


class TestSignalClientCreateRun:
    """Tests for creating runs."""

    @patch("client.python_client.requests.Session.request")
    def test_create_run_minimal(self, mock_request):
        """Test creating a run with minimal parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "run_123",
            "user_id": "user_456",
            "base_model": "meta-llama/Llama-3.2-3B",
            "status": "active",
            "created_at": "2024-01-01T00:00:00",
            "config": {"lora_r": 32},
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = client.create_run(base_model="meta-llama/Llama-3.2-3B")

        assert isinstance(run, SignalRun)
        assert run.run_id == "run_123"
        assert run.client == client

    @patch("client.python_client.requests.Session.request")
    def test_create_run_with_custom_config(self, mock_request):
        """Test creating a run with custom configuration."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "run_123",
            "user_id": "user_456",
            "base_model": "meta-llama/Llama-3.2-3B",
            "status": "active",
            "created_at": "2024-01-01T00:00:00",
            "config": {"lora_r": 64, "lora_alpha": 128},
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = client.create_run(
            base_model="meta-llama/Llama-3.2-3B",
            lora_r=64,
            lora_alpha=128,
            learning_rate=5e-4,
        )

        assert isinstance(run, SignalRun)
        assert run.config["lora_r"] == 64


class TestSignalClientMethods:
    """Tests for SignalClient methods."""

    @patch("client.python_client.requests.Session.request")
    def test_forward_backward(self, mock_request):
        """Test forward_backward method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "loss": 2.5,
            "step": 0,
            "grad_norm": 1.2,
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        result = client.forward_backward(
            run_id="run_123",
            batch=[{"text": "test"}],
            accumulate=False,
        )

        assert result["loss"] == 2.5
        assert result["step"] == 0

    @patch("client.python_client.requests.Session.request")
    def test_optim_step(self, mock_request):
        """Test optim_step method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "step": 1,
            "learning_rate": 3e-4,
            "metrics": {},
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        result = client.optim_step(run_id="run_123")

        assert result["step"] == 1

    @patch("client.python_client.requests.Session.request")
    def test_sample(self, mock_request):
        """Test sample method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": ["generated text 1", "generated text 2"],
            "logprobs": None,
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        result = client.sample(
            run_id="run_123",
            prompts=["prompt 1", "prompt 2"],
            temperature=0.7,
        )

        assert len(result["outputs"]) == 2

    @patch("client.python_client.requests.Session.request")
    def test_save_state(self, mock_request):
        """Test save_state method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artifact_uri": "/path/to/artifact",
            "checkpoint_path": "/path/to/checkpoint",
            "pushed_to_hub": False,
            "hub_model_id": None,
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        result = client.save_state(
            run_id="run_123",
            mode="adapter",
        )

        assert "checkpoint_path" in result

    @patch("client.python_client.requests.Session.request")
    def test_get_run_status(self, mock_request):
        """Test get_run_status method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "run_123",
            "status": "active",
            "current_step": 5,
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        result = client.get_run_status("run_123")

        assert result["run_id"] == "run_123"
        assert result["status"] == "active"

    @patch("client.python_client.requests.Session.request")
    def test_list_runs(self, mock_request):
        """Test list_runs method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "runs": [
                {"run_id": "run_1", "status": "active"},
                {"run_id": "run_2", "status": "completed"},
            ]
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        result = client.list_runs()

        assert len(result) == 2


class TestSignalClientErrorHandling:
    """Tests for error handling in client."""

    @patch("client.python_client.requests.Session.request")
    def test_error_response(self, mock_request):
        """Test that client raises exception on error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Bad request"}
        mock_response.text = "Bad request"
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")

        with pytest.raises(Exception) as exc_info:
            client.forward_backward(run_id="run_123", batch=[])

        assert "400" in str(exc_info.value)

    @patch("client.python_client.requests.Session.request")
    def test_unauthorized_error(self, mock_request):
        """Test handling 401 unauthorized error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Invalid API key"}
        mock_response.text = "Unauthorized"
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-invalid")

        with pytest.raises(Exception) as exc_info:
            client.list_models()

        assert "401" in str(exc_info.value)


class TestSignalRun:
    """Tests for SignalRun class."""

    def test_signal_run_initialization(self):
        """Test creating a SignalRun."""
        client = SignalClient(api_key="sk-test")
        run = SignalRun(
            client=client,
            run_id="run_123",
            config={"lora_r": 32},
        )

        assert run.client == client
        assert run.run_id == "run_123"
        assert run.config == {"lora_r": 32}

    @patch("client.python_client.requests.Session.request")
    def test_run_forward_backward(self, mock_request):
        """Test SignalRun.forward_backward delegates to client."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"loss": 2.5, "step": 0}
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = SignalRun(client, "run_123", {})

        result = run.forward_backward(batch=[{"text": "test"}])

        assert result["loss"] == 2.5

    @patch("client.python_client.requests.Session.request")
    def test_run_optim_step(self, mock_request):
        """Test SignalRun.optim_step delegates to client."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "step": 1,
            "learning_rate": 3e-4,
            "metrics": {},
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = SignalRun(client, "run_123", {})

        result = run.optim_step()

        assert result["step"] == 1

    @patch("client.python_client.requests.Session.request")
    def test_run_sample(self, mock_request):
        """Test SignalRun.sample delegates to client."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": ["text"], "logprobs": None}
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = SignalRun(client, "run_123", {})

        result = run.sample(prompts=["test"])

        assert len(result["outputs"]) == 1

    @patch("client.python_client.requests.Session.request")
    def test_run_save_state(self, mock_request):
        """Test SignalRun.save_state delegates to client."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artifact_uri": "/path",
            "checkpoint_path": "/path",
            "pushed_to_hub": False,
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = SignalRun(client, "run_123", {})

        result = run.save_state(mode="adapter")

        assert "checkpoint_path" in result

    @patch("client.python_client.requests.Session.request")
    def test_run_get_status(self, mock_request):
        """Test SignalRun.get_status delegates to client."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "run_123",
            "status": "active",
            "current_step": 5,
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = SignalRun(client, "run_123", {})

        result = run.get_status()

        assert result["status"] == "active"

    @patch("client.python_client.requests.Session.request")
    def test_run_get_metrics(self, mock_request):
        """Test SignalRun.get_metrics delegates to client."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "run_123",
            "step": 5,
            "metrics": [{"loss": 2.5}],
        }
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = SignalRun(client, "run_123", {})

        result = run.get_metrics()

        assert len(result["metrics"]) == 1


class TestSignalRunParameters:
    """Tests for SignalRun method parameters."""

    @patch("client.python_client.requests.Session.request")
    def test_forward_backward_accumulate_parameter(self, mock_request):
        """Test that accumulate parameter is passed correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"loss": 2.5, "step": 0}
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = SignalRun(client, "run_123", {})

        run.forward_backward(batch=[{"text": "test"}], accumulate=True)

        # Verify the request was made with correct parameters
        call_args = mock_request.call_args
        assert call_args[1]["json"]["accumulate"] is True

    @patch("client.python_client.requests.Session.request")
    def test_sample_parameters(self, mock_request):
        """Test that sample parameters are passed correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": ["text"], "logprobs": None}
        mock_request.return_value = mock_response

        client = SignalClient(api_key="sk-test")
        run = SignalRun(client, "run_123", {})

        run.sample(
            prompts=["test"],
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            return_logprobs=True,
        )

        # Verify parameters
        call_args = mock_request.call_args
        json_data = call_args[1]["json"]
        assert json_data["max_tokens"] == 256
        assert json_data["temperature"] == 0.8
        assert json_data["top_p"] == 0.95
        assert json_data["return_logprobs"] is True
