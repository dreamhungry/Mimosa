# -*- coding: utf-8 -*-
"""Abstract LLM interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional


class LLMInterface(ABC):
    """Abstract interface for all LLM implementations."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        :param messages: List of message dicts with 'role' and 'content' keys.
        :param system: Optional system prompt to prepend.
        :yields: Text chunks as they are generated.
        """
        ...
