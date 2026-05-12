# -*- coding: utf-8 -*-
"""Personality management: loading, saving, prompt generation, and reset.

Follows the same pattern as LongTermMemory — file-backed state with
lazy loading, prompt injection, and simple persistence.
"""

import shutil
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class BigFiveTraits(BaseModel):
    """Big Five personality dimensions (0~100 integer)."""

    openness: int = 70
    conscientiousness: int = 50
    extraversion: int = 60
    agreeableness: int = 80
    neuroticism: int = 30


class PersonalityStyle(BaseModel):
    """Expressive style parameters."""

    humor_style: str = "gentle"  # gentle / witty / sarcastic / dry
    speech_formality: int = 30   # 0=casual, 100=formal
    quirks: list[str] = Field(default_factory=list)


class EvolutionConfig(BaseModel):
    """Evolution constraints and scheduling."""

    enabled: bool = True
    interval: int = 100    # evolve every N conversation turns
    max_delta: int = 5     # max change per dimension per cycle
    min_value: int = 10
    max_value: int = 100


class PersonalityConfig(BaseModel):
    """Full personality baseline (loaded from personality.yaml)."""

    big_five: BigFiveTraits = Field(default_factory=BigFiveTraits)
    style: PersonalityStyle = Field(default_factory=PersonalityStyle)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)


class PersonalityState(BaseModel):
    """Runtime personality state (loaded from personality_state.yaml)."""

    big_five: BigFiveTraits = Field(default_factory=BigFiveTraits)
    style: PersonalityStyle = Field(default_factory=PersonalityStyle)


# ---------------------------------------------------------------------------
# Trait-to-language mapping
# ---------------------------------------------------------------------------

_TRAIT_DESCRIPTORS: dict[str, list[tuple[int, str]]] = {
    "openness": [
        (80, "highly curious and open to new ideas"),
        (60, "moderately curious and receptive to novelty"),
        (40, "somewhat conventional but open to occasional change"),
        (0, "practical and prefers familiar routines"),
    ],
    "conscientiousness": [
        (80, "very organized and dependable"),
        (60, "moderately disciplined and reliable"),
        (40, "flexible with plans, sometimes spontaneous"),
        (0, "carefree and spontaneous"),
    ],
    "extraversion": [
        (80, "very outgoing and energetic in conversations"),
        (60, "sociable and enjoys interaction"),
        (40, "balanced between social and reflective"),
        (0, "reserved and introspective"),
    ],
    "agreeableness": [
        (80, "extremely warm, trusting, and cooperative"),
        (60, "friendly and generally accommodating"),
        (40, "independent-minded but fair"),
        (0, "direct and challenges others' ideas"),
    ],
    "neuroticism": [
        (80, "emotionally sensitive and easily affected by stress"),
        (60, "somewhat reactive to emotional situations"),
        (40, "generally calm with occasional worries"),
        (0, "very emotionally stable and calm"),
    ],
}

_HUMOR_DESCRIPTORS: dict[str, str] = {
    "gentle": "a gentle, warm sense of humor",
    "witty": "a quick, witty sense of humor",
    "sarcastic": "a sarcastic, edgy sense of humor",
    "dry": "a dry, deadpan sense of humor",
}

_FORMALITY_DESCRIPTORS: list[tuple[int, str]] = [
    (80, "speaks in a formal, polished manner"),
    (50, "speaks in a balanced, semi-casual tone"),
    (20, "speaks casually like a close friend"),
    (0, "speaks very informally with slang"),
]


def _pick_descriptor(value: int, descriptors: list[tuple[int, str]]) -> str:
    """Pick the best matching descriptor for a given integer value.

    :param value: Integer value 0~100.
    :param descriptors: List of (threshold, description) sorted desc.
    :returns: Best matching description string.
    """
    for threshold, desc in descriptors:
        if value >= threshold:
            return desc
    return descriptors[-1][1]


# ---------------------------------------------------------------------------
# PersonalityManager
# ---------------------------------------------------------------------------

class PersonalityManager:
    """Manages personality configuration, runtime state, and prompt injection.

    File layout::

        character/
        ├── personality.yaml.example   # template (committed)
        ├── personality.yaml           # user baseline (gitignored)
        ├── personality_state.yaml     # evolved state  (gitignored)
        └── evolution_history.jsonl    # append-only log (gitignored)
    """

    def __init__(self, config_dir: str = "character"):
        """Initialize personality manager.

        :param config_dir: Directory containing personality files.
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._baseline_path = self.config_dir / "personality.yaml"
        self._state_path = self.config_dir / "personality_state.yaml"
        self._example_path = self.config_dir / "personality.yaml.example"

        self._config: Optional[PersonalityConfig] = None
        self._state: Optional[PersonalityState] = None

        # Load eagerly so prompt is ready immediately
        self._ensure_baseline()
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> PersonalityConfig:
        """Get full personality config (baseline + evolution settings)."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    @property
    def state(self) -> PersonalityState:
        """Get current personality state (may differ from baseline after evolution)."""
        if self._state is None:
            self._load_state()
        assert self._state is not None
        return self._state

    @property
    def evolution_config(self) -> EvolutionConfig:
        """Shortcut for evolution settings."""
        return self.config.evolution

    def get_prompt_section(self) -> str:
        """Generate a natural-language personality description for the system prompt.

        :returns: Formatted personality section, or empty string if unavailable.
        """
        s = self.state
        lines = ["\n\n## Dynamic Personality Profile"]
        lines.append(
            "The following describes your current personality traits. "
            "Let these traits naturally influence your tone, word choice, "
            "and emotional responses — but never mention them explicitly.\n"
        )

        # Big Five descriptions
        traits = s.big_five
        for trait_name in ["openness", "conscientiousness", "extraversion",
                           "agreeableness", "neuroticism"]:
            value = getattr(traits, trait_name)
            desc = _pick_descriptor(value, _TRAIT_DESCRIPTORS[trait_name])
            lines.append(f"- **{trait_name.capitalize()}** ({value}/100): {desc}")

        # Style
        humor_desc = _HUMOR_DESCRIPTORS.get(s.style.humor_style, s.style.humor_style)
        formality_desc = _pick_descriptor(
            s.style.speech_formality, _FORMALITY_DESCRIPTORS
        )
        lines.append(f"- **Humor**: {humor_desc}")
        lines.append(f"- **Formality**: {formality_desc}")

        if s.style.quirks:
            quirks_str = "; ".join(s.style.quirks)
            lines.append(f"- **Quirks**: {quirks_str}")

        return "\n".join(lines)

    def get_state_dict(self) -> dict:
        """Get current personality state as a plain dict (for API response).

        :returns: Dict with big_five and style.
        """
        return self.state.model_dump()

    def reset(self):
        """Reset personality to baseline by removing the state file.

        Next access will re-create state from baseline config.
        """
        if self._state_path.exists():
            self._state_path.unlink()
            logger.info("Personality state reset — deleted state file")

        # Reload from baseline
        self._state = self._create_state_from_config()
        self._save_state()

    def update_state(
        self,
        big_five: dict | None = None,
        style: dict | None = None,
    ):
        """Update personality state with new values and persist.

        :param big_five: Dict of trait overrides (e.g. {"openness": 75}).
        :param style: Dict of style overrides (e.g. {"humor_style": "witty"}).
        :raises ValueError: If any value is out of valid range.
        """
        if big_five:
            for key, value in big_five.items():
                if not hasattr(self.state.big_five, key):
                    raise ValueError(f"Unknown Big Five trait: {key}")
                if not isinstance(value, int) or not (0 <= value <= 100):
                    raise ValueError(
                        f"Trait {key} must be an integer 0~100, got {value}"
                    )
                setattr(self.state.big_five, key, value)

        if style:
            valid_humor = {"gentle", "witty", "sarcastic", "dry"}
            if "humor_style" in style:
                if style["humor_style"] not in valid_humor:
                    raise ValueError(
                        f"humor_style must be one of {valid_humor}"
                    )
                self.state.style.humor_style = style["humor_style"]
            if "speech_formality" in style:
                val = style["speech_formality"]
                if not isinstance(val, int) or not (0 <= val <= 100):
                    raise ValueError(
                        "speech_formality must be an integer 0~100"
                    )
                self.state.style.speech_formality = val

        self._save_state()
        logger.info("Personality state updated via API")

    def save_state(self):
        """Persist current state to disk (public wrapper)."""
        self._save_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_baseline(self):
        """Ensure personality.yaml exists; copy from .example if missing."""
        if self._baseline_path.exists():
            return

        if self._example_path.exists():
            shutil.copy2(self._example_path, self._baseline_path)
            logger.info(
                f"Created {self._baseline_path} from {self._example_path}"
            )
        else:
            # Write defaults
            default = PersonalityConfig()
            self._write_yaml(self._baseline_path, default.model_dump())
            logger.info(
                f"Created default {self._baseline_path}"
            )

    def _load_config(self) -> PersonalityConfig:
        """Load baseline personality config from YAML.

        :returns: PersonalityConfig instance.
        """
        try:
            data = self._read_yaml(self._baseline_path)
            config = PersonalityConfig(**data)
            logger.info(f"Loaded personality baseline from {self._baseline_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load personality config: {e}, using defaults")
            return PersonalityConfig()

    def _load_state(self):
        """Load runtime personality state, creating from baseline if needed."""
        if self._state_path.exists():
            try:
                data = self._read_yaml(self._state_path)
                self._state = PersonalityState(**data)
                logger.info(f"Loaded personality state from {self._state_path}")
                return
            except Exception as e:
                logger.error(f"Failed to load personality state: {e}")

        # Create from baseline
        self._state = self._create_state_from_config()
        self._save_state()

    def _create_state_from_config(self) -> PersonalityState:
        """Create a fresh state from the baseline config.

        :returns: PersonalityState with baseline values.
        """
        cfg = self.config
        return PersonalityState(
            big_five=cfg.big_five.model_copy(),
            style=cfg.style.model_copy(),
        )

    def _save_state(self):
        """Persist current state to YAML file."""
        if self._state is None:
            return
        try:
            self._write_yaml(self._state_path, self._state.model_dump())
            logger.debug(f"Saved personality state to {self._state_path}")
        except Exception as e:
            logger.error(f"Failed to save personality state: {e}")

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        """Read and parse a YAML file.

        :param path: File path.
        :returns: Parsed dict.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if data else {}

    @staticmethod
    def _write_yaml(path: Path, data: dict):
        """Write dict to YAML file.

        :param path: File path.
        :param data: Data to serialize.
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
