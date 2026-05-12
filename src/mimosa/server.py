# -*- coding: utf-8 -*-
"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from loguru import logger

from .config import MimosaConfig
from .routes import create_ws_router
from .api_routes import create_api_router
from .service_context import ServiceContext


def create_app(config: MimosaConfig) -> FastAPI:
    """Create and configure the FastAPI application.

    :param config: Application configuration.
    :returns: Configured FastAPI app.
    """
    app = FastAPI(
        title="Mimosa",
        description="Virtual companion server",
        version="0.1.0",
    )

    # CORS middleware for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize service context
    context = ServiceContext(config)

    # Register WebSocket routes
    ws_router = create_ws_router(context)
    app.include_router(ws_router)

    # Register REST API routes
    api_router = create_api_router(context)
    app.include_router(api_router)

    # Serve Live2D model files
    app.mount(
        "/live2d-models",
        StaticFiles(directory="live2d-models"),
        name="live2d-models",
    )

    # Serve frontend static files
    app.mount(
        "/",
        StaticFiles(directory="frontend", html=True),
        name="frontend",
    )

    logger.info(
        f"Mimosa server configured - "
        f"Character: {config.character.name}, "
        f"LLM: {config.llm.provider}/{config.llm.model}"
    )

    return app
