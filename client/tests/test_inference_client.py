"""Tests for InferenceClient specialized client."""

import json
import pytest
from unittest.mock import Mock, patch

from requests import Response
from requests.structures import CaseInsensitiveDict
from rewardsignal import InferenceClient


def _make_response(status_code: int, payload: dict) -> Response:
    """Create a Response with a JSON payload for mocking."""

    response = Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")
    response.headers = CaseInsensitiveDict({"Content-Type": "application/json"})
    response.url = "https://api.frontier-signal.com/test"
    return response


class TestInferenceClientInit:
    """Test InferenceClient initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
        )

        assert client.run_id == "test_run"
        assert client.api_key == "sk-test"
        assert client.timeout == 30  # Fast inference default
        assert client.max_retries == 5  # More retries for transient failures
        assert client.batch_size == 1
        assert client.step is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            step=100,
            timeout=60,
            batch_size=32,
        )

        assert client.step == 100
        assert client.timeout == 60
        assert client.batch_size == 32

    def test_init_with_shared_session(self):
        """Test initialization with shared session."""
        mock_session = Mock()
        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            session=mock_session,
        )

        assert client.session == mock_session
        assert not client._owns_session


class TestInferenceClientSample:
    """Test sample method."""

    @patch.object(InferenceClient, "_request")
    def test_sample_basic(self, mock_request):
        """Test basic sample call."""
        mock_request.return_value = {
            "outputs": ["Hello, world!", "Goodbye, world!"],
        }

        client = InferenceClient(run_id="test_run", api_key="sk-test")
        outputs = client.sample(
            prompts=["Say hello", "Say goodbye"],
            max_tokens=50,
        )

        assert outputs == ["Hello, world!", "Goodbye, world!"]

        # Verify request
        call_args = mock_request.call_args
        assert call_args[0][1] == "/runs/test_run/sample"
        payload = call_args[1]["json"]
        assert payload["prompts"] == ["Say hello", "Say goodbye"]
        assert payload["max_tokens"] == 50

    @patch.object(InferenceClient, "_request")
    def test_sample_with_step_override(self, mock_request):
        """Test sample with step override."""
        mock_request.return_value = {"outputs": ["output"]}

        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            step=100,
        )

        # Override with different step
        client.sample(
            prompts=["test"],
            step=200,
        )

        # Verify step in request
        call_args = mock_request.call_args
        payload = call_args[1]["json"]
        assert payload["step"] == 200  # Override takes precedence

    @patch.object(InferenceClient, "_request")
    def test_sample_with_instance_step(self, mock_request):
        """Test sample with instance-level step."""
        mock_request.return_value = {"outputs": ["output"]}

        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            step=100,
        )

        client.sample(prompts=["test"])

        # Verify instance step in request
        call_args = mock_request.call_args
        payload = call_args[1]["json"]
        assert payload["step"] == 100


class TestInferenceClientBatchSample:
    """Test batch_sample method."""

    @patch.object(InferenceClient, "sample")
    def test_batch_sample_single_batch(self, mock_sample):
        """Test batch_sample with single batch."""
        mock_sample.return_value = ["out1", "out2"]

        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            batch_size=5,
        )

        outputs = client.batch_sample(
            prompts=["p1", "p2"],
            max_tokens=50,
        )

        # Should call sample once
        assert mock_sample.call_count == 1
        assert outputs == ["out1", "out2"]

    @patch.object(InferenceClient, "sample")
    def test_batch_sample_multiple_batches(self, mock_sample):
        """Test batch_sample with multiple batches."""
        # Return different outputs for each batch
        mock_sample.side_effect = [
            ["out1", "out2"],
            ["out3", "out4"],
            ["out5"],
        ]

        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            batch_size=2,
        )

        outputs = client.batch_sample(
            prompts=["p1", "p2", "p3", "p4", "p5"],
            max_tokens=50,
        )

        # Should call sample 3 times (2 + 2 + 1)
        assert mock_sample.call_count == 3
        assert outputs == ["out1", "out2", "out3", "out4", "out5"]


class TestInferenceClientCache:
    """Test caching functionality."""

    @patch.object(InferenceClient, "_request")
    def test_cache_disabled_by_default(self, mock_request):
        """Test cache is disabled by default."""
        client = InferenceClient(run_id="test_run", api_key="sk-test")
        assert not client._cache_enabled
        assert len(client._cache) == 0

    @patch.object(InferenceClient, "_request")
    def test_cache_hit(self, mock_request):
        """Test cache hit for repeated prompt."""
        mock_request.return_value = {"outputs": ["cached result"]}

        client = InferenceClient(run_id="test_run", api_key="sk-test")
        client.enable_cache()

        # First call hits API
        output1 = client.sample(["test prompt"], max_tokens=50)
        assert mock_request.call_count == 1

        # Second call hits cache
        output2 = client.sample(["test prompt"], max_tokens=50)
        assert mock_request.call_count == 1  # Still 1, no new request

        assert output1 == output2

    @patch.object(InferenceClient, "_request")
    def test_cache_miss_different_params(self, mock_request):
        """Test cache miss when parameters differ."""
        mock_request.return_value = {"outputs": ["result"]}

        client = InferenceClient(run_id="test_run", api_key="sk-test")
        client.enable_cache()

        # First call
        client.sample(["test"], max_tokens=50)
        assert mock_request.call_count == 1

        # Different max_tokens -> cache miss
        client.sample(["test"], max_tokens=100)
        assert mock_request.call_count == 2

    def test_cache_stats(self):
        """Test cache statistics."""
        client = InferenceClient(run_id="test_run", api_key="sk-test")
        client.enable_cache()

        # Manually add to cache
        client._cache["key1"] = "value1"
        client._cache["key2"] = "value2"

        stats = client.get_cache_stats()
        assert stats["cache_enabled"] is True
        assert stats["cache_size"] == 2
        assert len(stats["cache_keys"]) == 2

    def test_clear_cache(self):
        """Test clearing cache."""
        client = InferenceClient(run_id="test_run", api_key="sk-test")
        client.enable_cache()

        client._cache["key1"] = "value1"
        assert len(client._cache) == 1

        client.clear_cache()
        assert len(client._cache) == 0

    def test_disable_cache(self):
        """Test disabling cache."""
        client = InferenceClient(run_id="test_run", api_key="sk-test")
        client.enable_cache()

        client._cache["key1"] = "value1"
        assert client._cache_enabled is True

        client.disable_cache()
        assert client._cache_enabled is False
        assert len(client._cache) == 0


class TestInferenceClientStreamSample:
    """Test stream_sample method (future feature)."""

    @patch.object(InferenceClient, "sample")
    def test_stream_sample_placeholder(self, mock_sample):
        """Test stream_sample falls back to sample for now."""
        mock_sample.return_value = ["output"]

        client = InferenceClient(run_id="test_run", api_key="sk-test")

        # Stream should yield the output
        results = list(client.stream_sample("test prompt"))
        assert results == ["output"]


class TestInferenceClientEmbeddings:
    """Test embeddings method (future feature)."""

    def test_embeddings_not_implemented(self):
        """Test embeddings raises NotImplementedError."""
        client = InferenceClient(run_id="test_run", api_key="sk-test")

        with pytest.raises(NotImplementedError):
            client.embeddings(["text1", "text2"])


class TestInferenceClientContextManager:
    """Test context manager support."""

    def test_context_manager_creates_session(self):
        """Test context manager creates session."""
        client = InferenceClient(run_id="test_run", api_key="sk-test")

        with client as inference:
            assert inference.session is not None

    @patch("frontier_signal.inference_client.requests.Session")
    def test_context_manager_closes_owned_session(self, mock_session_class):
        """Test context manager closes owned session."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        client = InferenceClient(run_id="test_run", api_key="sk-test")

        with client:
            pass

        # Verify session was closed
        mock_session.close.assert_called_once()

    def test_context_manager_doesnt_close_shared_session(self):
        """Test context manager doesn't close shared session."""
        mock_session = Mock()
        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            session=mock_session,
        )

        with client:
            pass

        # Shared session should not be closed
        mock_session.close.assert_not_called()


class TestInferenceClientMigration:
    """Ensure migration handling for inference requests."""

    @patch("rewardsignal.migration.time.sleep", return_value=None)
    def test_request_waits_for_migration(self, _mock_sleep):
        """The client should wait for migration before returning a result."""

        session = Mock()
        session.request = Mock(
            side_effect=[
                _make_response(
                    425,
                    {
                        "error": {
                            "code": "run_migrating",
                            "message": "Run migrating",
                            "details": {
                                "status": "migrating",
                                "from": "L40S:2",
                                "to": "A100-80GB:1",
                                "checkpoint_step": 512,
                            },
                        }
                    },
                ),
                _make_response(200, {"outputs": ["done"]}),
            ]
        )
        session.get = Mock(
            side_effect=[
                _make_response(
                    200,
                    {
                        "status": "migrating",
                        "from": "L40S:2",
                        "to": "A100-80GB:1",
                        "checkpoint_step": 513,
                    },
                ),
                _make_response(
                    200,
                    {
                        "status": "running",
                        "from": "A100-80GB:1",
                        "to": "A100-80GB:1",
                        "checkpoint_step": 513,
                    },
                ),
            ]
        )
        callback = Mock()

        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            session=session,
            migration_callback=callback,
            migration_poll_interval=0.01,
        )

        result = client._request("POST", "/runs/test_run/sample", json={})

        assert result == {"outputs": ["done"]}
        assert session.request.call_count == 2
        assert session.get.call_count == 2
        assert callback.call_count >= 2
        last_payload = callback.call_args_list[-1].args[0]
        assert last_payload["phase"] == "running"

    def test_wait_for_migration_immediate_return(self):
        """wait_for_migration should return immediately when run already running."""

        session = Mock()
        session.get = Mock()

        client = InferenceClient(
            run_id="test_run",
            api_key="sk-test",
            session=session,
        )

        status = client.wait_for_migration(initial_status={"status": "running"})

        assert status["status"].lower() == "running"
        session.get.assert_not_called()
