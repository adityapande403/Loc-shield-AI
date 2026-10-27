"""Security, authentication, privacy headers, and request auditing for FastAPI service."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
logger = logging.getLogger("localshield.api.security")


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> None:
    """Validate optional API key if LOCALSHIELD_AUTH_TOKEN is configured."""
    expected_token = os.getenv("LOCALSHIELD_AUTH_TOKEN")
    if not expected_token:
        # Authentication disabled for local loopback
        return

    if not api_key or api_key != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API authentication token.",
        )


async def privacy_audit_middleware(request: Request, call_next):
    """Audit request metadata WITHOUT logging sensitive body contents."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Never log request body or query parameters
    logger.info(
        f"API Request: {request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration_ms:.2f}ms"
    )
    return response
