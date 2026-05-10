#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mimosa server entry point."""

import uvicorn
from loguru import logger

from src.mimosa.config import load_config
from src.mimosa.server import create_app


def main():
    """Start the Mimosa server."""
    config = load_config()
    app = create_app(config)

    host = config.server.host
    port = config.server.port
    # Show a clickable URL with localhost instead of 0.0.0.0
    display_host = "localhost" if host == "0.0.0.0" else host
    logger.info(f"Mimosa is running at: http://{display_host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
