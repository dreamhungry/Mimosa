# -*- coding: utf-8 -*-
"""Chat history management for conversation memory."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class ChatHistory:
    """Simple chat history manager.

    Stores messages as a list and supports persistence to JSON files.
    """

    def __init__(self, max_messages: int = 50, storage_dir: str = "conversations"):
        """Initialize chat history.

        :param max_messages: Maximum messages to keep in memory.
        :param storage_dir: Directory for persisting conversations.
        """
        self.max_messages = max_messages
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._messages: List[Dict[str, str]] = []
        self._session_id: Optional[str] = None

    @property
    def messages(self) -> List[Dict[str, str]]:
        """Get current conversation messages (role + content only for LLM)."""
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def add_message(self, role: str, content: str):
        """Add a message to history.

        :param role: Message role ('user' or 'assistant').
        :param content: Message content.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self._messages.append(message)

        # Trim to max size (keep recent messages)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def clear(self):
        """Clear all messages."""
        self._messages = []

    def save(self, session_id: Optional[str] = None):
        """Save conversation to JSON file.

        :param session_id: Session identifier for the file name.
        """
        sid = session_id or self._session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_id = sid

        file_path = self.storage_dir / f"{sid}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._messages, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self._messages)} messages to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def load(self, session_id: str) -> bool:
        """Load conversation from JSON file.

        :param session_id: Session identifier.
        :returns: True if loaded successfully.
        """
        file_path = self.storage_dir / f"{session_id}.json"
        if not file_path.exists():
            logger.warning(f"Chat history file not found: {file_path}")
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self._messages = json.load(f)
            self._session_id = session_id
            logger.info(f"Loaded {len(self._messages)} messages from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            return False
