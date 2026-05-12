# -*- coding: utf-8 -*-
"""Personality evolution engine.

Uses LLM reflection to analyze conversation patterns and propose
bounded adjustments to Big Five traits and style parameters.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from ..llm.llm_interface import LLMInterface
from .personality_manager import (
    EvolutionConfig,
    PersonalityManager,
    PersonalityState,
)


# ---------------------------------------------------------------------------
# Reflection prompt template
# ---------------------------------------------------------------------------

EVOLUTION_REFLECTION_PROMPT = """You are a personality evolution analyst. Analyze the recent conversation and determine how the character's personality should subtly shift based on interaction patterns.

## Current Personality State
{current_state}

## Recent Conversation
{conversation}

## Task
Based on the conversation above, decide whether any Big Five traits or style parameters should be adjusted. Consider:
- Did the user prefer more formal or casual language?
- Did the user respond well to humor? What kind?
- Was the user seeking emotional support (higher agreeableness) or intellectual challenge (lower agreeableness)?
- Was the conversation exploratory/creative (higher openness) or practical (lower openness)?
- Did the user seem to prefer concise or elaborate responses?

## Output Format
Respond with ONLY a JSON object (no markdown, no extra text):
{{
    "adjustments": {{
        "openness": 0,
        "conscientiousness": 0,
        "extraversion": 0,
        "agreeableness": 0,
        "neuroticism": 0,
        "speech_formality": 0
    }},
    "reasoning": "Brief explanation of why these adjustments were made"
}}

Rules:
- Each adjustment must be an integer between -{max_delta} and +{max_delta}
- Use 0 for traits that should not change
- Only adjust traits where the conversation provides clear evidence
- Be conservative — small, gradual shifts are better than large jumps"""


class PersonalityEvolver:
    """Drives personality evolution through LLM reflection and constrained updates."""

    def __init__(
        self,
        personality_manager: PersonalityManager,
        llm: LLMInterface,
    ):
        """Initialize the evolver.

        :param personality_manager: Manager holding current state.
        :param llm: LLM interface for reflection calls.
        """
        self.manager = personality_manager
        self.llm = llm
        self._history_path = self.manager.config_dir / "evolution_history.jsonl"

    async def evolve(self, conversation_text: str) -> bool:
        """Run one evolution cycle.

        :param conversation_text: Formatted recent conversation.
        :returns: True if personality was updated, False otherwise.
        """
        evo_cfg = self.manager.evolution_config
        if not evo_cfg.enabled:
            logger.debug("Personality evolution is disabled")
            return False

        if not conversation_text.strip():
            return False

        # Build reflection prompt
        prompt = self._build_reflection_prompt(conversation_text, evo_cfg)

        # Call LLM
        try:
            raw_response = ""
            async for chunk in self.llm.chat_completion(
                [{"role": "user", "content": prompt}],
                system="You are a precise personality analysis assistant. Respond with valid JSON only.",
            ):
                raw_response += chunk

            adjustments = self._parse_response(raw_response)
            if adjustments is None:
                logger.warning("Failed to parse evolution response, skipping")
                return False

            # Apply constrained adjustments
            old_state = self.manager.state.model_copy(deep=True)
            changed = self._apply_adjustments(adjustments, evo_cfg)

            if changed:
                self.manager.save_state()
                self._log_history(old_state, adjustments, conversation_text)
                logger.info(f"Personality evolved: {adjustments}")

            return changed

        except Exception as e:
            logger.error(f"Personality evolution failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_reflection_prompt(
        self, conversation_text: str, evo_cfg: EvolutionConfig
    ) -> str:
        """Build the LLM reflection prompt.

        :param conversation_text: Recent conversation.
        :param evo_cfg: Evolution constraints.
        :returns: Formatted prompt string.
        """
        state = self.manager.state
        state_lines = []
        for trait in ["openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"]:
            state_lines.append(f"  {trait}: {getattr(state.big_five, trait)}")
        state_lines.append(f"  humor_style: {state.style.humor_style}")
        state_lines.append(f"  speech_formality: {state.style.speech_formality}")
        current_state_str = "\n".join(state_lines)

        return EVOLUTION_REFLECTION_PROMPT.format(
            current_state=current_state_str,
            conversation=conversation_text,
            max_delta=evo_cfg.max_delta,
        )

    def _parse_response(self, raw: str) -> Optional[dict]:
        """Parse LLM JSON response into adjustments dict.

        :param raw: Raw LLM output.
        :returns: Dict with 'adjustments' and 'reasoning', or None.
        """
        raw = raw.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        try:
            data = json.loads(raw)
            if "adjustments" not in data:
                logger.warning("Evolution response missing 'adjustments' key")
                return None
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse evolution JSON: {e}")
            logger.debug(f"Raw response: {raw[:500]}")
            return None

    def _apply_adjustments(self, parsed: dict, evo_cfg: EvolutionConfig) -> bool:
        """Apply constrained adjustments to current personality state.

        :param parsed: Parsed response with 'adjustments' dict.
        :param evo_cfg: Evolution constraints.
        :returns: True if any value actually changed.
        """
        adj = parsed.get("adjustments", {})
        state = self.manager.state
        changed = False

        # Big Five traits
        for trait in ["openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"]:
            delta = adj.get(trait, 0)
            if not isinstance(delta, (int, float)):
                continue
            delta = int(round(delta))
            delta = max(-evo_cfg.max_delta, min(evo_cfg.max_delta, delta))
            if delta == 0:
                continue

            old_val = getattr(state.big_five, trait)
            new_val = max(evo_cfg.min_value, min(evo_cfg.max_value, old_val + delta))
            if new_val != old_val:
                setattr(state.big_five, trait, new_val)
                changed = True

        # Speech formality
        formality_delta = adj.get("speech_formality", 0)
        if isinstance(formality_delta, (int, float)) and formality_delta != 0:
            formality_delta = int(round(formality_delta))
            formality_delta = max(-evo_cfg.max_delta, min(evo_cfg.max_delta, formality_delta))
            old_val = state.style.speech_formality
            new_val = max(0, min(100, old_val + formality_delta))
            if new_val != old_val:
                state.style.speech_formality = new_val
                changed = True

        return changed

    def _log_history(
        self,
        old_state: PersonalityState,
        parsed: dict,
        conversation_text: str,
    ):
        """Append an evolution event to the history log.

        :param old_state: State before evolution.
        :param parsed: Parsed LLM response.
        :param conversation_text: Conversation that triggered evolution.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_values": old_state.model_dump(),
            "new_values": self.manager.state.model_dump(),
            "adjustments": parsed.get("adjustments", {}),
            "reasoning": parsed.get("reasoning", ""),
        }

        try:
            with open(self._history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug("Appended evolution history entry")
        except Exception as e:
            logger.error(f"Failed to write evolution history: {e}")
