# -*- coding: utf-8 -*-
"""AgentCore: pure-text agent wrapping LLM + history + memory + personality.

This module contains NO media dependencies (ASR, TTS, VAD, Live2D).
It can be used independently for experiments, testing, or headless runs.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from ..config import LLMConfig, MemoryConfig, PersonalityMetaConfig
from ..llm.llm_factory import create_llm
from ..llm.llm_interface import LLMInterface
from ..memory.chat_history import ChatHistory
from ..memory.long_term_memory import LongTermMemory
from ..personality import PersonalityEvolver, PersonalityManager


class AgentCore:
    """Pure-text agent: LLM + chat history + long-term memory + personality.

    This is the minimal core that powers both the full Mimosa demo
    and the persona-drift experiment harness.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        persona_prompt: str,
        personality_config_dir: str = "character",
        memory_extraction_interval: int = 10,
    ):
        """Initialize core agent services.

        :param llm_config: LLM provider configuration.
        :param persona_prompt: Base persona prompt text.
        :param personality_config_dir: Directory containing personality files.
        :param memory_extraction_interval: Extract memory every N turns.
        """
        # LLM
        self.llm: LLMInterface = create_llm(llm_config)
        self._llm_config = llm_config

        # Memory
        self.chat_history = ChatHistory()
        self.long_term_memory = LongTermMemory()

        # Personality
        self.personality = PersonalityManager(config_dir=personality_config_dir)

        # Persona prompt (loaded externally)
        self.persona_prompt = persona_prompt

        # Turn tracking for periodic tasks
        self._turn_count = 0
        self._extraction_interval = memory_extraction_interval
        self._evolution_interval = self.personality.evolution_config.interval
        self._evolution_enabled = self.personality.evolution_config.enabled

    @property
    def full_system_prompt(self) -> str:
        """Get full system prompt: persona + personality + long-term memory.

        :returns: Combined system prompt string.
        """
        personality_section = self.personality.get_prompt_section()
        memory_section = self.long_term_memory.get_prompt_injection()
        return self.persona_prompt + personality_section + memory_section

    async def chat(self, user_input: str) -> str:
        """Run one conversation turn (pure text, no media).

        Adds user message to history, calls LLM, adds assistant message,
        saves history, and triggers periodic memory extraction / personality
        evolution as needed.

        :param user_input: User's text message.
        :returns: Agent's text response (clean, no emotion tags).
        """
        if not user_input.strip():
            return ""

        self.chat_history.add_message("user", user_input)

        # Stream LLM response and collect full text
        full_response = ""
        async for chunk in self.llm.chat_completion(
            self.chat_history.messages,
            system=self.full_system_prompt,
        ):
            full_response += chunk

        if not full_response:
            logger.warning("LLM returned empty response")
            return ""

        # Add to history and save
        self.chat_history.add_message("assistant", full_response)
        self.chat_history.save()

        # Periodic tasks
        self._turn_count += 1
        await self._maybe_extract_memory()
        await self._maybe_evolve_personality()

        return full_response

    async def _maybe_extract_memory(self):
        """Extract memory from recent conversation if interval reached."""
        if self._turn_count % self._extraction_interval != 0:
            return

        messages = self.chat_history.messages
        if len(messages) < 2:
            return

        window = self._extraction_interval * 2
        recent = messages[-window:]

        conversation_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )

        extraction_prompt = self.long_term_memory.build_extraction_prompt(
            conversation_text
        )

        try:
            result = ""
            async for chunk in self.llm.chat_completion(
                [{"role": "user", "content": extraction_prompt}],
                system="You are a precise fact extraction assistant. Follow instructions exactly.",
            ):
                result += chunk

            self.long_term_memory.update_from_extraction(result)
            logger.info(
                f"Memory extraction (turn {self._turn_count}): {result[:100]}..."
            )
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")

    async def _maybe_evolve_personality(self):
        """Evolve personality based on recent conversation if interval reached."""
        if not self._evolution_enabled:
            return
        if self._turn_count % self._evolution_interval != 0:
            return

        messages = self.chat_history.messages
        if len(messages) < 4:
            return

        window = self._evolution_interval * 2
        recent = messages[-window:]

        conversation_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )

        try:
            evolver = PersonalityEvolver(self.personality, self.llm)
            changed = await evolver.evolve(conversation_text)
            if changed:
                logger.info(
                    f"Personality evolution (turn {self._turn_count}): updated"
                )
        except Exception as e:
            logger.error(f"Personality evolution failed: {e}")

    def update_llm(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        """Hot-reload LLM configuration.

        :param model: New model name.
        :param temperature: New temperature (0~2).
        :param max_tokens: New max tokens.
        :param api_key: New API key.
        """
        updates = {}
        if model is not None:
            updates["model"] = model
        if temperature is not None:
            if not (0 <= temperature <= 2):
                raise ValueError("temperature must be between 0 and 2")
            updates["temperature"] = temperature
        if max_tokens is not None:
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            updates["max_tokens"] = max_tokens
        if api_key is not None and api_key.strip():
            updates["api_key"] = api_key

        if not updates:
            return

        self._llm_config = self._llm_config.model_copy(update=updates)
        self.llm = create_llm(self._llm_config)
        logger.info(f"LLM config updated: {list(updates.keys())}")

    def load_latest_history(self):
        """Load the most recent conversation history for session continuity."""
        storage_dir = self.chat_history.storage_dir
        if not storage_dir.exists():
            return

        json_files = sorted(
            storage_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not json_files:
            return

        latest = json_files[0].stem
        if self.chat_history.load(latest):
            logger.info(f"Restored previous conversation: {latest}")

    def snapshot(self) -> dict:
        """Return full agent state for logging / experiment tracking.

        :returns: Dict with personality state, turn count, memory content.
        """
        return {
            "turn_count": self._turn_count,
            "personality": self.personality.get_state_dict(),
            "memory_length": len(self.long_term_memory.content),
            "history_length": len(self.chat_history.messages),
        }
