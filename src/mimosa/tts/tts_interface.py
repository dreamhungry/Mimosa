# -*- coding: utf-8 -*-
"""Abstract TTS interface."""

from abc import ABC, abstractmethod


class TTSInterface(ABC):
    """Abstract interface for Text-to-Speech synthesis."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.

        :param text: Text to convert to speech.
        :returns: Audio data as bytes (WAV format).
        """
        ...
