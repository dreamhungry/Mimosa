# -*- coding: utf-8 -*-
"""WebSocket route definitions."""

from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler


def create_ws_router(context: ServiceContext) -> APIRouter:
    """Create WebSocket router.

    :param context: Service context.
    :returns: APIRouter with WebSocket endpoint.
    """
    router = APIRouter()
    ws_handler = WebSocketHandler(context)

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint for client communication."""
        client_id = str(uuid4())

        try:
            await ws_handler.handle_connection(websocket, client_id)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
            await ws_handler.handle_disconnect(client_id)

    return router
