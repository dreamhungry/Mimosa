# -*- coding: utf-8 -*-
"""Edge TTS implementation."""

import io

from loguru import logger

from .tts_interface import TTSInterface


class EdgeTTSEngine(TTSInterface):
    """TTS implementation using Microsoft Edge TTS (free, high quality)."""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%", volume: str = "+0%"):
        """Initialize Edge TTS engine.

        :param voice: Voice name (e.g., 'zh-CN-XiaoxiaoNeural', 'en-US-AriaNeural').
        :param rate: Speech rate adjustment (e.g., '+10%', '-5%').
        :param volume: Volume adjustment.
        """
        self.voice = voice
        self.rate = rate
        self.volume = volume
        logger.info(f"Initialized Edge TTS: voice={voice}, rate={rate}")

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio using Edge TTS.

        :param text: Text to convert to speech.
        :returns: Audio data as bytes (MP3 format).
        """
        import edge_tts

        if not text.strip():
            return b""

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
            )

            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

            audio_data = audio_buffer.getvalue()
            logger.debug(f"TTS synthesized {len(audio_data)} bytes for: {text[:50]}...")
            return audio_data

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return b""
