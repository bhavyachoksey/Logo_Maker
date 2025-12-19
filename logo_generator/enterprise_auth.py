"""
Enterprise authentication bootstrap (ADC / embedded enterprise auth snippet).

Rules:
- No API keys
- No .env loading
- No secrets stored in code
- Initialize Vertex AI once
- Return nothing (sets global SDK context)
"""

from __future__ import annotations

from typing import Optional

import google.auth

_INITIALIZED = False
_PROJECT_ID: Optional[str] = None
_LOCATION: str = "us-central1"


def init_enterprise_auth(project_id: Optional[str] = None, location: str = "us-central1") -> None:
    """
    Initializes Google Enterprise authentication using Application Default Credentials (ADC)
    and configures Vertex AI context exactly once.

    You will later replace/extend the internals of this function with your embedded
    enterprise auth logic (still no secrets, no env loading).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    credentials, detected_project_id = google.auth.default()
    resolved_project_id = project_id or detected_project_id
    if not resolved_project_id:
        raise RuntimeError(
            "Google project id could not be resolved. Pass project_id to init_enterprise_auth() "
            "or ensure ADC provides a default project."
        )

    # We intentionally do NOT store secrets or load env vars.
    # This function's job is to ensure ADC is available and to set global context once.
    global _PROJECT_ID, _LOCATION
    _PROJECT_ID = resolved_project_id
    _LOCATION = location

    # Force credential resolution now so downstream client creation is deterministic.
    _ = credentials.token  # may be None until first request; still validates auth object is present
    _INITIALIZED = True


def get_enterprise_context() -> tuple[str, str]:
    if not _INITIALIZED or not _PROJECT_ID:
        raise RuntimeError("Enterprise auth not initialized. Call init_enterprise_auth() first.")
    return _PROJECT_ID, _LOCATION


