"""FastAPI application factory, lifespan management, and global error handlers."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from localshield.api.routes import router, set_shield_instance
from localshield.api.security import privacy_audit_middleware
from localshield.config import LocalShieldConfig
from localshield.shield import LocalShield

logger = logging.getLogger("localshield.api")


def create_app(config: Optional[LocalShieldConfig] = None) -> FastAPI:
    """Instantiate and configure the FastAPI service."""
    cfg = config or LocalShieldConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: initialize engine
        logger.info("Initializing LocalShield engine for service...")
        shield = LocalShield(config=cfg)
        set_shield_instance(shield)
        logger.info("LocalShield service ready.")
        yield
        # Shutdown
        logger.info("Shutting down LocalShield service.")

    app = FastAPI(
        title="LocalShield AI Service",
        description="Privacy-First Local PII Detection & Redaction Engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Restrictive CORS by default (only local loopback)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Privacy audit middleware
    app.middleware("http")(privacy_audit_middleware)

    # Include routes
    app.include_router(router)

    # Global safe exception handler: never leak raw PII or stack traces in responses
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Internal error processing request: {type(exc).__name__}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An error occurred while processing the request securely.",
            },
        )

    return app
