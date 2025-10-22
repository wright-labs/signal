"""Shared migration-handling utilities for Signal SDK clients."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import requests

from .exceptions import (
    ConnectionError as SignalConnectionError,
    MigrationInProgressError,
    SignalAPIError,
    TimeoutError as SignalTimeoutError,
)

# Type alias for migration progress callbacks.
MigrationCallback = Callable[[Dict[str, Any]], None]

logger = logging.getLogger(__name__)


class MigrationSupportMixin:
    """Mixin that adds migration polling and replay helpers."""

    _migration_poll_interval: float = 5.0
    _migration_timeout: Optional[float] = 600.0
    _migration_callback: Optional[MigrationCallback] = None

    _MIGRATION_STATUS_CODES = {425}
    _MIGRATION_STATUS_ENDPOINT = "/runs/{run_id}/migration_status"

    def _init_migration_support(
        self,
        *,
        migration_callback: Optional[MigrationCallback] = None,
        migration_poll_interval: float = 5.0,
        migration_timeout: Optional[float] = 600.0,
    ) -> None:
        """Initialize migration-related configuration."""

        self._migration_callback = migration_callback
        self._migration_poll_interval = migration_poll_interval
        self._migration_timeout = migration_timeout

    # -- Public helpers -------------------------------------------------
    def set_migration_callback(self, callback: Optional[MigrationCallback]) -> None:
        """Register a callback to observe migration progress events."""

        self._migration_callback = callback

    def wait_for_migration(
        self,
        *,
        poll_interval: Optional[float] = None,
        timeout: Optional[float] = None,
        initial_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Block until the current run finishes migrating."""

        poll_interval = poll_interval or self._migration_poll_interval
        timeout = self._migration_timeout if timeout is None else timeout
        start_time = time.time()

        status: Optional[Dict[str, Any]] = None
        if initial_status:
            status = dict(initial_status)
            self._emit_migration_progress(status, initial=True)
            if self._migration_state(status) == "running":
                return status

        while True:
            if timeout is not None and (time.time() - start_time) > timeout:
                raise SignalTimeoutError("Timed out waiting for run migration to complete.")

            status = self._fetch_migration_status()
            self._emit_migration_progress(status)

            if self._migration_state(status) == "running":
                return status

            time.sleep(poll_interval)

    def resume_after_migration(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        *,
        poll_interval: Optional[float] = None,
        timeout: Optional[float] = None,
        initial_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Wait for migration to finish and then replay the request."""

        poll_interval = poll_interval or self._migration_poll_interval
        timeout = self._migration_timeout if timeout is None else timeout
        pending_status = initial_status

        while True:
            self.wait_for_migration(
                poll_interval=poll_interval,
                timeout=timeout,
                initial_status=pending_status,
            )

            try:
                return self._perform_request(
                    method,
                    endpoint,
                    json=json,
                    allow_migration=False,
                )
            except MigrationInProgressError as exc:
                pending_status = exc.migration_status or {}
                continue

    # -- Internal helpers ----------------------------------------------
    def _perform_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        *,
        allow_migration: bool = True,
    ) -> Dict[str, Any]:
        """Execute an HTTP request with retry and migration handling."""

        url = f"{self.base_url}{endpoint}"
        last_exception: Optional[SignalAPIError] = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, json=json, timeout=self.timeout)

                migration_payload = self._extract_migration_payload(response)
                if migration_payload is not None:
                    if allow_migration:
                        return self.resume_after_migration(
                            method,
                            endpoint,
                            json=json,
                            initial_status=migration_payload,
                        )

                    raise MigrationInProgressError(
                        message=self._format_migration_message(migration_payload),
                        migration_status=migration_payload,
                        status_code=response.status_code,
                    )

                if response.status_code >= 400:
                    response.raise_for_status()

                return response.json()

            except requests.exceptions.Timeout:
                last_exception = SignalTimeoutError(
                    f"Request to {endpoint} timed out after {self.timeout}s"
                )
            except requests.exceptions.ConnectionError as exc:
                last_exception = SignalConnectionError(f"Failed to connect to {url}: {exc}")
            except requests.exceptions.RequestException as exc:
                last_exception = SignalAPIError(f"Request failed: {exc}")

            if attempt < self.max_retries - 1:
                time.sleep(self._get_retry_delay(attempt))

        if last_exception is None:
            last_exception = SignalAPIError("Request failed")
        raise last_exception

    # -- Migration detection utilities ---------------------------------
    def _extract_migration_payload(self, response: requests.Response) -> Optional[Dict[str, Any]]:
        """Return migration metadata when a response indicates migration."""

        payload: Optional[Dict[str, Any]] = None

        try:
            data = response.json()
        except ValueError:
            data = None

        if isinstance(data, dict):
            status_value = str(data.get("status", "")).lower()
            if status_value == "migrating":
                payload = dict(data)
            elif "migration" in data and isinstance(data["migration"], dict):
                payload = dict(data["migration"])
                payload.setdefault("status", data.get("status", "migrating"))

            if payload is None:
                error = data.get("error")
                if isinstance(error, dict):
                    code = str(error.get("code", "")).lower()
                    message = str(error.get("message", "")).lower()
                    if "migrat" in code or "migrat" in message:
                        details = error.get("details")
                        if isinstance(details, dict):
                            payload = dict(details)
                        else:
                            payload = {}
                        if error.get("status"):
                            payload.setdefault("status", error["status"])
                        if error.get("message"):
                            payload.setdefault("message", error["message"])
                        if error.get("code"):
                            payload.setdefault("code", error["code"])

            if payload is not None:
                payload.setdefault("status", data.get("status", payload.get("status", "migrating")))
                if "message" not in payload and isinstance(data, dict) and data.get("message"):
                    payload["message"] = data["message"]
                migration_data = data.get("migration")
                if isinstance(migration_data, dict):
                    for key, value in migration_data.items():
                        payload.setdefault(key, value)

        if payload is None and response.status_code in self._MIGRATION_STATUS_CODES:
            payload = {"status": "migrating"}

        return payload

    def _fetch_migration_status(self) -> Dict[str, Any]:
        """Fetch the latest migration status from the API."""

        endpoint = self._MIGRATION_STATUS_ENDPOINT.format(run_id=self.run_id)
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise SignalTimeoutError(
                f"Request to {endpoint} timed out after {self.timeout}s"
            ) from None
        except requests.exceptions.ConnectionError as exc:
            raise SignalConnectionError(f"Failed to connect to {url}: {exc}") from None
        except requests.exceptions.RequestException as exc:
            raise SignalAPIError(f"Request failed: {exc}") from None

        try:
            data = response.json()
        except ValueError as exc:
            raise SignalAPIError(
                "Invalid response from migration status endpoint.",
                status_code=response.status_code,
            ) from exc

        if response.status_code >= 400:
            message = self._extract_error_message(data)
            raise SignalAPIError(
                message or "Failed to fetch migration status.",
                status_code=response.status_code,
                response_data=data,
            )

        if isinstance(data, dict):
            data.setdefault("status", data.get("state", "migrating"))

        return data

    def _emit_migration_progress(self, status: Dict[str, Any], *, initial: bool = False) -> None:
        """Emit migration progress via callback or logging."""

        payload = dict(status) if isinstance(status, dict) else {}
        payload.setdefault("status", self._migration_state(payload))
        payload["phase"] = self._migration_state(payload)
        payload["initial"] = initial
        payload["message"] = self._format_migration_message(payload)

        if self._migration_callback:
            try:
                self._migration_callback(payload)
            except Exception:  # pragma: no cover - user callback errors shouldn't crash
                logger.exception("Migration callback raised an exception")
        else:
            message = payload.get("message")
            if message:
                logger.info(message)

    def _migration_state(self, status: Dict[str, Any]) -> str:
        """Return the normalized migration state."""

        if not isinstance(status, dict):
            return "unknown"
        value = status.get("status") or status.get("state")
        if isinstance(value, str):
            return value.lower()
        return "unknown"

    def _format_migration_message(self, status: Dict[str, Any]) -> str:
        """Build a user-friendly migration progress message."""

        if not isinstance(status, dict):
            return "Waiting for run migration to finish"

        phase = self._migration_state(status)
        source = status.get("from") or status.get("source") or status.get("source_config")
        target = status.get("to") or status.get("target") or status.get("target_config")
        checkpoint = (
            status.get("checkpoint_step")
            or status.get("checkpoint")
            or status.get("step")
        )
        base_message = status.get("message")

        parts = []
        if source or target:
            if source and target:
                parts.append(f"Upgrading from {source} → {target}")
            elif target:
                parts.append(f"Upgrading to {target}")
            else:
                parts.append(f"Migrating from {source}")

        if checkpoint is not None:
            parts.append(f"checkpoint step {checkpoint}")

        if phase and phase != "running":
            parts.append(f"status: {phase}")

        message = ", ".join(parts)
        if not message and base_message:
            message = base_message
        if not message:
            if phase == "running":
                message = "Migration complete: run is back to running"
            else:
                message = "Waiting for run migration to finish"

        return message

    def _extract_error_message(self, data: Any) -> Optional[str]:
        """Extract error message from a response payload."""

        if not isinstance(data, dict):
            return None

        error = data.get("error")
        if isinstance(error, dict):
            return error.get("message") or error.get("detail")

        return data.get("message")

    # -- Abstract hooks -------------------------------------------------
    def _get_retry_delay(self, attempt: int) -> float:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError
