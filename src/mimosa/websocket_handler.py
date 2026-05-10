# -*- coding: utf-8 -*-
"""WebSocket message handler."""

import asyncio
import json
from typing import Dict

from fastapi import WebSocket
from loguru import logger

from .conversation.conversation_handler import ConversationHandler
from .service_context import ServiceContext


class WebSocketHandler:
    """Handles WebSocket communication with clients."""

    def __init__(self, context: ServiceContext):
        """Initialize WebSocket handler.

        :param context: Service context with all engines.
        """
        self.context = context
        self._connections: Dict[str, WebSocket] = {}
        self._handlers: Dict[str, ConversationHandler] = {}

    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """Handle a new WebSocket connection.

        :param websocket: The WebSocket connection.
        :param client_id: Unique client identifier.
        """
        await websocket.accept()
        self._connections[client_id] = websocket
        self._handlers[client_id] = ConversationHandler(self.context)

        logger.info(f"Client connected: {client_id}")

        # Send welcome message
        await self._send(websocket, {
            "type": "connected",
            "message": f"Welcome! I'm {self.context.config.character.name}.",
            "model_path": self.context.live2d.get_model_path(),
        })

        try:
            await self._listen(websocket, client_id)
        finally:
            await self.handle_disconnect(client_id)

    async def _listen(self, websocket: WebSocket, client_id: str):
        """Listen for messages from a client.

        :param websocket: The WebSocket connection.
        :param client_id: Client identifier.
        """
        handler = self._handlers[client_id]

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")

            if msg_type == "text-input":
                text = message.get("text", "")
                await handler.handle_text_input(
                    text,
                    send_fn=lambda msg: self._send(websocket, msg),
                )

            elif msg_type == "audio-data":
                audio = message.get("audio", [])
                logger.info(
                    f"[Audio] Received audio-data: {len(audio)} samples"
                )
                await handler.handle_audio_data(audio)

            elif msg_type == "audio-end":
                logger.info("[Audio] Received audio-end signal")
                await handler.handle_audio_end(
                    send_fn=lambda msg: self._send(websocket, msg),
                )

            elif msg_type == "realtime-start":
                logger.info("[Audio] Real-time voice mode started")
                handler.start_realtime_mode(
                    send_fn=lambda msg: self._send(websocket, msg)
                )
                await self._send(websocket, {
                    "type": "realtime-started",
                    "message": "Real-time voice mode active",
                })

            elif msg_type == "realtime-audio":
                audio = message.get("audio", [])
                if not hasattr(self, '_rt_audio_log_count'):
                    self._rt_audio_log_count = 0
                self._rt_audio_log_count += 1
                if self._rt_audio_log_count <= 5:
                    logger.debug(
                        f"[WS] realtime-audio received: "
                        f"{len(audio)} samples, "
                        f"range=[{min(audio) if audio else 0:.4f}, "
                        f"{max(audio) if audio else 0:.4f}]"
                    )
                elif self._rt_audio_log_count == 6:
                    logger.debug("[WS] (suppressing further realtime-audio logs)")
                await handler.handle_realtime_audio(
                    audio,
                    send_fn=lambda msg: self._send(websocket, msg),
                )

            elif msg_type == "realtime-stop":
                logger.info("[Audio] Real-time voice mode stopped")
                handler.stop_realtime_mode()
                await self._send(websocket, {
                    "type": "realtime-stopped",
                    "message": "Real-time voice mode ended",
                })

            elif msg_type == "interrupt":
                await handler.handle_interrupt()

            elif msg_type == "ping":
                await self._send(websocket, {"type": "pong"})

            else:
                logger.warning(f"Unknown message type: {msg_type}")

    async def handle_disconnect(self, client_id: str):
        """Clean up after client disconnects.

        :param client_id: Client identifier.
        """
        self._connections.pop(client_id, None)
        handler = self._handlers.pop(client_id, None)

        if handler:
            # Save conversation history on disconnect
            handler.ctx.chat_history.save(client_id)

            # Extract long-term memory in background (non-blocking)
            asyncio.create_task(self._extract_memory(handler))

        logger.info(f"Client disconnected: {client_id}")

    async def _extract_memory(self, handler: ConversationHandler):
        """Extract key facts from conversation and update long-term memory.

        :param handler: The conversation handler with history.
        """
        messages = handler.ctx.chat_history.messages
        if len(messages) < 2:
            return

        # Format conversation for the extraction prompt
        conversation_lines = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Mimosa"
            conversation_lines.append(f"{role}: {msg['content']}")
        conversation_text = "\n".join(conversation_lines)

        # Build extraction prompt
        extraction_prompt = handler.ctx.long_term_memory.build_extraction_prompt(
            conversation_text
        )

        # Call LLM to extract facts (collect full response)
        try:
            extraction_result = ""
            async for chunk in handler.ctx.llm.chat_completion(
                [{"role": "user", "content": extraction_prompt}],
                system="You are a precise fact extraction assistant. Follow instructions exactly.",
            ):
                extraction_result += chunk

            # Update memory file
            handler.ctx.long_term_memory.update_from_extraction(extraction_result)
            logger.info(f"Memory extraction complete: {extraction_result[:100]}...")

        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")

    async def _send(self, websocket: WebSocket, message: dict):
        """Send a JSON message to a WebSocket client.

        :param websocket: Target WebSocket connection.
        :param message: Message dict to send.
        """
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
