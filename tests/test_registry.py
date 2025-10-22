"""Tests for run registry."""

from unittest.mock import MagicMock

import pytest


class TestRunCreation:
    """Tests for creating runs."""

    def test_create_run_basic(self, run_registry, test_user_id):
        """Test creating a basic run."""
        config = {"lora_r": 32, "learning_rate": 3e-4}

        run_id = run_registry.create_run(
            user_id=test_user_id,
            base_model="meta-llama/Llama-3.2-3B",
            config=config,
        )

        assert run_id is not None
        assert run_id.startswith("run_")
        assert len(run_id) > 10

    def test_create_run_unique_ids(self, run_registry, test_user_id):
        """Test that each run gets a unique ID."""
        config = {"lora_r": 32}

        run_id1 = run_registry.create_run(test_user_id, "model1", config)
        run_id2 = run_registry.create_run(test_user_id, "model2", config)

        assert run_id1 != run_id2

    def test_create_run_stores_config(self, run_registry, test_user_id):
        """Test that run configuration is stored correctly."""
        config = {
            "lora_r": 64,
            "lora_alpha": 128,
            "learning_rate": 5e-4,
        }

        run_id = run_registry.create_run(test_user_id, "test-model", config)
        run = run_registry.get_run(run_id)

        assert run["config"] == config

    def test_create_run_initial_status(self, run_registry, test_user_id):
        """Test that runs are created with correct initial status."""
        run_id = run_registry.create_run(test_user_id, "model", {})
        run = run_registry.get_run(run_id)

        assert run["status"] == "initialized"
        assert run["current_step"] == 0
        assert run["user_id"] == test_user_id
        assert run["base_model"] == "model"

    def test_create_run_timestamps(self, run_registry, test_user_id):
        """Test that runs have correct timestamps."""
        run_id = run_registry.create_run(test_user_id, "model", {})
        run = run_registry.get_run(run_id)

        assert "created_at" in run
        assert "updated_at" in run
        assert run["created_at"] is not None
        assert run["updated_at"] is not None


class TestRunQueries:
    """Tests for querying runs."""

    def test_get_run_by_id(self, run_registry, test_user_id):
        """Test getting a run by ID."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        run = run_registry.get_run(run_id)

        assert run is not None
        assert run["run_id"] == run_id
        assert run["user_id"] == test_user_id

    def test_get_nonexistent_run(self, run_registry):
        """Test getting a run that doesn't exist."""
        run = run_registry.get_run("nonexistent_run_id")
        assert run is None

    def test_list_runs_for_user(self, run_registry, test_user_id):
        """Test listing all runs for a user."""
        # Create multiple runs
        run_id1 = run_registry.create_run(test_user_id, "model1", {})
        run_id2 = run_registry.create_run(test_user_id, "model2", {})

        runs = run_registry.list_runs(test_user_id)

        assert len(runs) == 2
        run_ids = [run["run_id"] for run in runs]
        assert run_id1 in run_ids
        assert run_id2 in run_ids

    def test_list_runs_empty(self, run_registry):
        """Test listing runs for user with no runs."""
        runs = run_registry.list_runs("user_with_no_runs")
        assert runs == []

    def test_list_runs_isolation(self, run_registry):
        """Test that users only see their own runs."""
        user1_run = run_registry.create_run("user1", "model", {})
        user2_run = run_registry.create_run("user2", "model", {})

        user1_runs = run_registry.list_runs("user1")
        user2_runs = run_registry.list_runs("user2")

        assert len(user1_runs) == 1
        assert len(user2_runs) == 1
        assert user1_runs[0]["run_id"] == user1_run
        assert user2_runs[0]["run_id"] == user2_run


class TestRunUpdates:
    """Tests for updating run information."""

    def test_update_run_status(self, run_registry, test_user_id):
        """Test updating run status."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        result = run_registry.update_run(run_id, status="active")
        assert result is True

        run = run_registry.get_run(run_id)
        assert run["status"] == "active"

    def test_update_current_step(self, run_registry, test_user_id):
        """Test updating current step."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        run_registry.update_run(run_id, current_step=5)

        run = run_registry.get_run(run_id)
        assert run["current_step"] == 5

    def test_update_appends_metrics(self, run_registry, test_user_id):
        """Test that updating with metrics appends to list."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        # Add first metric
        run_registry.update_run(
            run_id, current_step=0, metrics={"loss": 2.5, "grad_norm": 1.2}
        )

        # Add second metric
        run_registry.update_run(
            run_id, current_step=1, metrics={"loss": 2.3, "grad_norm": 1.1}
        )

        run = run_registry.get_run(run_id)
        assert len(run["metrics"]) == 2
        assert run["metrics"][0]["loss"] == 2.5
        assert run["metrics"][1]["loss"] == 2.3

    def test_update_metrics_includes_timestamp(self, run_registry, test_user_id):
        """Test that metrics include timestamps."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        run_registry.update_run(run_id, metrics={"loss": 2.5})

        run = run_registry.get_run(run_id)
        assert len(run["metrics"]) == 1
        assert "timestamp" in run["metrics"][0]
        assert "step" in run["metrics"][0]
        assert "loss" in run["metrics"][0]

    def test_update_modifies_updated_at(self, run_registry, test_user_id):
        """Test that updates modify updated_at timestamp."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        run = run_registry.get_run(run_id)
        original_updated_at = run["updated_at"]

        import time

        time.sleep(0.01)

        run_registry.update_run(run_id, status="active")

        run = run_registry.get_run(run_id)
        assert run["updated_at"] != original_updated_at

    def test_update_nonexistent_run(self, run_registry):
        """Test updating a run that doesn't exist."""
        result = run_registry.update_run("nonexistent", status="active")
        assert result is False

    def test_update_run_records_migration(
        self, run_registry, test_user_id
    ):
        """Updating GPU config should append migration history and status message."""

        run_id = "run_test_migration"
        initial_run = {
            "id": run_id,
            "run_id": run_id,
            "user_id": test_user_id,
            "status": "running",
            "current_step": 0,
            "config": {"gpu_config": "L40S:1"},
            "current_gpu": "L40S:1",
            "migration_history": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }

        # Pretend the run exists in the registry
        original_get_run = run_registry.get_run
        run_registry.get_run = MagicMock(return_value=initial_run)

        result = run_registry.update_run(
            run_id,
            status="migrating",
            config_updates={"gpu_config": "H100:1"},
            status_message="Scaling to H100",
        )

        assert result is True

        update_call = run_registry.supabase.table.return_value.update
        updated_payload = update_call.call_args[0][0]

        assert updated_payload["status"] == "migrating"
        assert updated_payload["status_message"] == "Scaling to H100"
        assert updated_payload["config"]["gpu_config"] == "H100:1"
        assert updated_payload["current_gpu"] == "L40S:1"
        assert updated_payload["target_gpu"] == "H100:1"
        assert updated_payload["migration_history"]
        last_event = updated_payload["migration_history"][-1]
        assert last_event["from_gpu"] == "L40S:1"
        assert last_event["to_gpu"] == "H100:1"

        # Restore original method to avoid leaking state between tests
        run_registry.get_run = original_get_run


class TestRunDeletion:
    """Tests for deleting runs."""

    def test_delete_run(self, run_registry, test_user_id):
        """Test deleting a run."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        result = run_registry.delete_run(run_id, test_user_id)
        assert result is True

        # Verify run is deleted
        run = run_registry.get_run(run_id)
        assert run is None

    def test_delete_prevents_unauthorized(self, run_registry):
        """Test that users can't delete other users' runs."""
        run_id = run_registry.create_run("user1", "model", {})

        # Try to delete as different user
        result = run_registry.delete_run(run_id, "user2")
        assert result is False

        # Verify run still exists
        run = run_registry.get_run(run_id)
        assert run is not None

    def test_delete_nonexistent_run(self, run_registry, test_user_id):
        """Test deleting a run that doesn't exist."""
        result = run_registry.delete_run("nonexistent", test_user_id)
        assert result is False


class TestMetricsManagement:
    """Tests for metrics storage and retrieval."""

    def test_get_metrics(self, run_registry, test_user_id):
        """Test retrieving metrics for a run."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        # Add some metrics
        run_registry.update_run(run_id, metrics={"loss": 2.5})
        run_registry.update_run(run_id, metrics={"loss": 2.3})

        metrics = run_registry.get_metrics(run_id)
        assert len(metrics) == 2

    def test_get_metrics_empty(self, run_registry, test_user_id):
        """Test getting metrics for run with no metrics."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        metrics = run_registry.get_metrics(run_id)
        assert metrics == []

    def test_get_metrics_nonexistent_run(self, run_registry):
        """Test getting metrics for nonexistent run."""
        metrics = run_registry.get_metrics("nonexistent")
        assert metrics is None

    def test_metrics_preserve_order(self, run_registry, test_user_id):
        """Test that metrics are returned in order added."""
        run_id = run_registry.create_run(test_user_id, "model", {})

        # Add metrics in sequence
        for i in range(5):
            run_registry.update_run(
                run_id, current_step=i, metrics={"loss": 3.0 - i * 0.1}
            )

        metrics = run_registry.get_metrics(run_id)
        assert len(metrics) == 5

        # Verify order
        for i, metric in enumerate(metrics):
            assert metric["step"] == i
            assert metric["loss"] == pytest.approx(3.0 - i * 0.1)
