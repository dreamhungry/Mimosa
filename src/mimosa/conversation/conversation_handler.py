# -*- coding: utf-8 -*-
"""Conversation handler orchestrating the full dialogue flow."""

import asyncio
import base64
import random
from typing import Any, Callable, Coroutine, Dict, Optional

import numpy as np
from loguru import logger

from ..service_context import ServiceContext


class ConversationHandler:
    """Orchestrates the full conversation flow from input to response."""

    # VAD state machine states
    _STATE_IDLE = "idle"
    _STATE_LISTENING = "listening"
    _STATE_SPEAKING = "speaking"

    def __init__(self, context: ServiceContext):
        """Initialize conversation handler.

        :param context: Service context with all engines.
        """
        self.ctx = context
        self._audio_buffer: list = []
        self._is_speaking = False
        self._speech_detected = False

        # Real-time VAD state
        self._vad_state = self._STATE_IDLE
        self._silence_samples = 0
        self._speech_samples = 0
        self._realtime_mode = False
        self._send_fn: Optional[Callable] = None

        # VAD parameters (from config)
        sample_rate = self.ctx.config.asr.sample_rate
        vad_cfg = self.ctx.config.vad
        self._min_silence_samples = int(
            vad_cfg.min_silence_duration_ms * sample_rate / 1000
        )
        self._min_speech_samples = int(0.3 * sample_rate)  # 300ms min speech
        self._vad_chunk_size = int(0.032 * sample_rate)  # 32ms per VAD frame

    def start_realtime_mode(self, send_fn: Callable):
        """Enter real-time voice mode with server-side VAD.

        :param send_fn: Async function to send messages to client.
        """
        self._realtime_mode = True
        self._send_fn = send_fn
        self._vad_state = self._STATE_LISTENING
        self._audio_buffer = []
        self._silence_samples = 0
        self._speech_samples = 0
        self._speech_detected = False
        self.ctx.vad.reset()
        logger.info("[VAD] Real-time mode started, listening...")

    def stop_realtime_mode(self):
        """Exit real-time voice mode."""
        self._realtime_mode = False
        self._vad_state = self._STATE_IDLE
        self._audio_buffer = []
        self._send_fn = None
        logger.info("[VAD] Real-time mode stopped")

    async def handle_realtime_audio(
        self,
        audio_data: list,
        send_fn: Callable[[Dict[str, Any]], Coroutine],
    ) -> bool:
        """Process real-time audio with server-side VAD.

        Returns True if endpoint was detected and ASR triggered.

        :param audio_data: List of float audio samples.
        :param send_fn: Async function to send messages.
        :returns: True if speech endpoint detected.
        """
        if not self._realtime_mode:
            return False

        self._send_fn = send_fn
        self._audio_buffer.extend(audio_data)

        # Process VAD on incoming chunk
        audio_chunk = np.array(audio_data, dtype=np.float32)

        if not hasattr(self, '_vad_debug_count'):
            self._vad_debug_count = 0
        self._vad_debug_count += 1
        if self._vad_debug_count <= 3:
            rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0
            logger.debug(
                f"[VAD] Audio chunk: samples={len(audio_chunk)}, "
                f"rms={rms:.6f}, max={np.max(np.abs(audio_chunk)) if len(audio_chunk) > 0 else 0:.4f}"
            )

        is_speech = self._run_vad(audio_chunk)

        if is_speech:
            self._speech_samples += len(audio_data)
            self._silence_samples = 0

            if not self._speech_detected:
                self._speech_detected = True
                logger.info("[VAD] Speech detected, recording...")
                await send_fn({"type": "vad-status", "status": "speech_start"})

        else:
            if self._speech_detected:
                self._silence_samples += len(audio_data)

                # Check if silence exceeded threshold (endpoint)
                if self._silence_samples >= self._min_silence_samples:
                    logger.info(
                        f"[VAD] Endpoint detected: "
                        f"speech={self._speech_samples} samples, "
                        f"silence={self._silence_samples} samples"
                    )
                    await send_fn({"type": "vad-status", "status": "speech_end"})

                    # Only process if enough speech was detected
                    if self._speech_samples >= self._min_speech_samples:
                        await self._process_realtime_audio(send_fn)
                    else:
                        logger.info("[VAD] Speech too short, discarding")

                    # Reset for next utterance
                    self._audio_buffer = []
                    self._silence_samples = 0
                    self._speech_samples = 0
                    self._speech_detected = False
                    self.ctx.vad.reset()
                    return True

        return False

    def _run_vad(self, audio_chunk: np.ndarray) -> bool:
        """Run VAD on audio chunk, processing in frame-sized segments.

        :param audio_chunk: Audio samples as float32 array.
        :returns: True if speech detected in chunk.
        """
        # Process in VAD frame-sized chunks
        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio_chunk), self._vad_chunk_size):
            frame = audio_chunk[i:i + self._vad_chunk_size]
            if len(frame) < self._vad_chunk_size:
                break
            total_frames += 1
            if self.ctx.vad.is_speech(frame):
                speech_frames += 1

        if total_frames == 0:
            return False

        # If majority of frames contain speech
        return speech_frames > total_frames * 0.3

    async def _process_realtime_audio(
        self,
        send_fn: Callable[[Dict[str, Any]], Coroutine],
    ):
        """Process accumulated audio from real-time mode.

        :param send_fn: Async function to send messages.
        """
        # Trim trailing silence from buffer
        trim_samples = self._silence_samples
        if trim_samples > 0 and len(self._audio_buffer) > trim_samples:
            audio_data = self._audio_buffer[:-trim_samples]
        else:
            audio_data = self._audio_buffer

        audio_np = np.array(audio_data, dtype=np.float32)
        duration = len(audio_np) / self.ctx.config.asr.sample_rate
        max_val = float(np.max(np.abs(audio_np)))

        logger.info(
            f"[Audio] Realtime processing: {len(audio_np)} samples, "
            f"duration={duration:.2f}s, max_amplitude={max_val:.6f}"
        )

        # Transcribe
        text = await self.ctx.asr.transcribe(
            audio_np, self.ctx.config.asr.sample_rate
        )

        if not text.strip():
            logger.info("[Audio] ASR returned empty result")
            await send_fn({"type": "asr-result", "text": ""})
            return

        logger.info(f"[Audio] ASR result: \"{text}\"")
        await send_fn({"type": "asr-result", "text": text})

        # Add to history and generate response
        self.ctx.chat_history.add_message("user", text)
        await self._generate_response(send_fn)

    async def handle_text_input(
        self,
        text: str,
        send_fn: Callable[[Dict[str, Any]], Coroutine],
    ):
        """Handle text input from user.

        :param text: User's text message.
        :param send_fn: Async function to send response messages back.
        """
        if not text.strip():
            return

        logger.info(f"User text: {text}")

        # Add to history
        self.ctx.chat_history.add_message("user", text)

        # Get LLM response
        await self._generate_response(send_fn)

    async def handle_audio_data(self, audio_data: list):
        """Buffer incoming audio data (manual mode).

        :param audio_data: List of float audio samples.
        """
        self._audio_buffer.extend(audio_data)

    async def handle_audio_end(
        self,
        send_fn: Callable[[Dict[str, Any]], Coroutine],
    ):
        """Process buffered audio after recording ends (manual mode).

        :param send_fn: Async function to send response messages back.
        """
        if not self._audio_buffer:
            logger.warning("[Audio] Empty audio buffer, skipping")
            return

        # Convert to numpy array
        audio_np = np.array(self._audio_buffer, dtype=np.float32)
        buffer_len = len(self._audio_buffer)
        self._audio_buffer = []

        duration = len(audio_np) / self.ctx.config.asr.sample_rate
        max_val = float(np.max(np.abs(audio_np)))
        logger.info(
            f"[Audio] Processing: {buffer_len} samples, "
            f"duration={duration:.2f}s, max_amplitude={max_val:.6f}"
        )

        if max_val < 0.001:
            logger.warning(
                "[Audio] Audio amplitude too low (max < 0.001), "
                "microphone might not be capturing properly"
            )

        # Transcribe
        text = await self.ctx.asr.transcribe(audio_np, self.ctx.config.asr.sample_rate)

        if not text.strip():
            logger.info("[Audio] ASR returned empty result (no speech detected)")
            await send_fn({"type": "asr-result", "text": ""})
            return

        logger.info(f"[Audio] ASR result: \"{text}\"")

        # Send transcription to frontend
        await send_fn({"type": "asr-result", "text": text})

        # Add to history and generate response
        self.ctx.chat_history.add_message("user", text)
        await self._generate_response(send_fn)

    async def _generate_response(
        self,
        send_fn: Callable[[Dict[str, Any]], Coroutine],
    ):
        """Generate and send LLM response with TTS and emotion.

        Core text logic (LLM call, history, memory extraction, personality
        evolution) is delegated to AgentCore. This method adds the media
        layer: Live2D emotion extraction and TTS synthesis.

        :param send_fn: Async function to send messages.
        """
        agent = self.ctx.agent
        messages = agent.chat_history.messages

        # Stream LLM response (system prompt includes long-term memory)
        full_response = ""
        async for chunk in agent.llm.chat_completion(
            messages, system=agent.full_system_prompt
        ):
            full_response += chunk

        if not full_response:
            logger.warning("LLM returned empty response")
            return

        logger.info(f"LLM response: {full_response[:100]}...")

        # Extract emotion from response (media layer)
        clean_text, emotion = self.ctx.live2d.extract_emotion(full_response)
        expression = self.ctx.live2d.get_expression_name(emotion)

        # Send text response and emotion
        await send_fn({
            "type": "llm-response",
            "text": clean_text,
            "emotion": emotion,
            "expression": expression,
        })

        # Add assistant message to history (clean version without emotion tag)
        agent.chat_history.add_message("assistant", clean_text)

        # Generate TTS audio (media layer)
        audio_bytes = await self.ctx.tts.synthesize(clean_text)
        if audio_bytes:
            # Send audio as base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await send_fn({
                "type": "tts-audio",
                "audio": audio_b64,
                "format": "mp3",
            })

        # Save history
        agent.chat_history.save()

        # Periodic memory extraction and personality evolution (delegated to AgentCore)
        agent._turn_count += 1
        asyncio.create_task(agent._maybe_extract_memory())
        asyncio.create_task(agent._maybe_evolve_personality())

    async def handle_interrupt(self):
        """Handle user interruption signal."""
        self._is_speaking = False
        logger.info("Conversation interrupted by user")

    # Random variation pools for interaction prompts
    _INTERACTION_MOODS = [
        "sleepy", "energetic", "shy", "playful", "mischievous",
        "dreamy", "grumpy but cute", "excited", "clingy", "tsundere",
    ]
    _INTERACTION_SCENARIOS = [
        "you were daydreaming",
        "you were humming a tune",
        "you were reading something interesting",
        "you were about to doze off",
        "you were practicing magic spells",
        "you were stretching lazily",
        "you were lost in thought",
        "you were doodling",
        "you were counting stars",
        "you were playing with your hair",
    ]

    async def handle_interaction(
        self,
        trigger: str,
        send_fn: Callable[[Dict[str, Any]], Coroutine],
    ):
        """Handle a click interaction request.

        Generates a short, playful LLM response without adding to chat history.

        :param trigger: Interaction trigger type (e.g., 'click').
        :param send_fn: Async function to send messages.
        """
        mood = random.choice(self._INTERACTION_MOODS)
        scenario = random.choice(self._INTERACTION_SCENARIOS)

        interaction_prompt = (
            f"The user just poked/clicked on you. "
            f"Right now you are feeling {mood}, and {scenario}. "
            f"Respond with a very short, playful reaction (1 sentence max, under 30 words). "
            f"Be cute and expressive. Do NOT repeat phrases you've used before. "
            f"Include an emotion tag at the start like [joy], [surprise], [love], etc. "
            f"Do NOT greet. Just react naturally as if someone poked you."
        )

        try:
            full_response = ""
            async for chunk in self.ctx.llm.chat_completion(
                [{"role": "user", "content": interaction_prompt}],
                system=self.ctx.full_system_prompt,
            ):
                full_response += chunk

            if not full_response:
                return

            # Extract emotion from response
            clean_text, emotion = self.ctx.live2d.extract_emotion(full_response)
            expression = self.ctx.live2d.get_expression_name(emotion)

            # Cache phrase for future reuse
            self.ctx.interaction_phrases.add(clean_text)

            await send_fn({
                "type": "interaction-response",
                "text": clean_text,
                "emotion": emotion,
                "expression": expression,
            })

            logger.info(f"Interaction response: {clean_text[:80]}")

        except Exception as e:
            logger.error(f"Interaction LLM failed: {e}")

    async def _extract_memory_periodic(self):
        """Extract key facts from recent conversation (delegated to AgentCore).

        Kept for backward compatibility. New code should use
        agent._maybe_extract_memory() directly.
        """
        await self.ctx.agent._maybe_extract_memory()

    async def _evolve_personality_periodic(self):
        """Evolve personality based on recent conversation (delegated to AgentCore).

        Kept for backward compatibility. New code should use
        agent._maybe_evolve_personality() directly.
        """
        await self.ctx.agent._maybe_evolve_personality()
