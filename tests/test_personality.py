# -*- coding: utf-8 -*-
"""Unit tests for the personality system."""

import json
import shutil
from pathlib import Path

import pytest

from src.mimosa.personality.personality_manager import (
    BigFiveTraits,
    EvolutionConfig,
    PersonalityConfig,
    PersonalityManager,
    PersonalityState,
    PersonalityStyle,
    _pick_descriptor,
)


@pytest.fixture
def tmp_character_dir(tmp_path):
    """Create a temporary character directory with baseline config."""
    char_dir = tmp_path / "character"
    char_dir.mkdir()

    # Write a baseline personality.yaml
    import yaml

    baseline = {
        "big_five": {
            "openness": 70,
            "conscientiousness": 50,
            "extraversion": 60,
            "agreeableness": 80,
            "neuroticism": 30,
        },
        "style": {
            "humor_style": "gentle",
            "speech_formality": 30,
            "quirks": ["uses plant metaphors occasionally"],
        },
        "evolution": {
            "enabled": True,
            "interval": 100,
            "max_delta": 5,
            "min_value": 10,
            "max_value": 100,
        },
    }
    with open(char_dir / "personality.yaml", "w") as f:
        yaml.dump(baseline, f)

    return char_dir


class TestBigFiveTraits:
    """Tests for BigFiveTraits model."""

    def test_defaults(self):
        traits = BigFiveTraits()
        assert traits.openness == 70
        assert traits.conscientiousness == 50
        assert traits.extraversion == 60
        assert traits.agreeableness == 80
        assert traits.neuroticism == 30

    def test_custom_values(self):
        traits = BigFiveTraits(openness=90, neuroticism=10)
        assert traits.openness == 90
        assert traits.neuroticism == 10


class TestPersonalityStyle:
    """Tests for PersonalityStyle model."""

    def test_defaults(self):
        style = PersonalityStyle()
        assert style.humor_style == "gentle"
        assert style.speech_formality == 30
        assert style.quirks == []

    def test_custom_quirks(self):
        style = PersonalityStyle(quirks=["says hmm a lot"])
        assert len(style.quirks) == 1


class TestPersonalityManager:
    """Tests for PersonalityManager."""

    def test_load_baseline(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        assert mgr.state.big_five.openness == 70
        assert mgr.state.style.humor_style == "gentle"

    def test_state_created_from_baseline(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        # State file should be created
        assert (tmp_character_dir / "personality_state.yaml").exists()

    def test_evolution_config_loaded(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        assert mgr.evolution_config.enabled is True
        assert mgr.evolution_config.max_delta == 5
        assert mgr.evolution_config.interval == 100

    def test_reset(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))

        # Modify state
        mgr.state.big_five.openness = 90
        mgr.save_state()
        assert mgr.state.big_five.openness == 90

        # Reset
        mgr.reset()
        assert mgr.state.big_five.openness == 70

    def test_get_prompt_section(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        section = mgr.get_prompt_section()

        assert "Dynamic Personality Profile" in section
        assert "Openness" in section
        assert "70/100" in section
        assert "gentle" in section.lower()

    def test_get_state_dict(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        data = mgr.get_state_dict()

        assert "big_five" in data
        assert "style" in data
        assert data["big_five"]["openness"] == 70

    def test_fallback_from_example(self, tmp_path):
        """Test auto-copy from .example when personality.yaml is missing."""
        char_dir = tmp_path / "character"
        char_dir.mkdir()

        # Only provide .example file
        import yaml

        example = {
            "big_five": {"openness": 85},
            "style": {"humor_style": "witty"},
            "evolution": {"enabled": False},
        }
        with open(char_dir / "personality.yaml.example", "w") as f:
            yaml.dump(example, f)

        mgr = PersonalityManager(config_dir=str(char_dir))
        assert mgr.state.big_five.openness == 85
        assert mgr.state.style.humor_style == "witty"
        assert mgr.evolution_config.enabled is False

    def test_fallback_defaults(self, tmp_path):
        """Test defaults when no files exist at all."""
        char_dir = tmp_path / "character"
        # Don't create dir — manager should create it
        mgr = PersonalityManager(config_dir=str(char_dir))
        assert mgr.state.big_five.openness == 70  # default
        assert char_dir.exists()


class TestPickDescriptor:
    """Tests for _pick_descriptor helper."""

    def test_high_value(self):
        descriptors = [(80, "high"), (50, "mid"), (0, "low")]
        assert _pick_descriptor(90, descriptors) == "high"

    def test_mid_value(self):
        descriptors = [(80, "high"), (50, "mid"), (0, "low")]
        assert _pick_descriptor(60, descriptors) == "mid"

    def test_low_value(self):
        descriptors = [(80, "high"), (50, "mid"), (0, "low")]
        assert _pick_descriptor(10, descriptors) == "low"

    def test_exact_threshold(self):
        descriptors = [(80, "high"), (50, "mid"), (0, "low")]
        assert _pick_descriptor(80, descriptors) == "high"


class TestPersonalityEvolver:
    """Tests for PersonalityEvolver constraint logic."""

    def test_apply_adjustments_within_bounds(self, tmp_character_dir):
        from src.mimosa.personality.personality_evolver import PersonalityEvolver

        mgr = PersonalityManager(config_dir=str(tmp_character_dir))

        # Create evolver with a mock LLM (we'll test _apply_adjustments directly)
        evolver = PersonalityEvolver(mgr, llm=None)

        parsed = {
            "adjustments": {
                "openness": 3,
                "neuroticism": -2,
                "speech_formality": 5,
            },
            "reasoning": "test",
        }

        changed = evolver._apply_adjustments(parsed, mgr.evolution_config)
        assert changed is True
        assert mgr.state.big_five.openness == 73
        assert mgr.state.big_five.neuroticism == 28
        assert mgr.state.style.speech_formality == 35

    def test_apply_adjustments_clamped(self, tmp_character_dir):
        from src.mimosa.personality.personality_evolver import PersonalityEvolver

        mgr = PersonalityManager(config_dir=str(tmp_character_dir))

        evolver = PersonalityEvolver(mgr, llm=None)

        # Try to exceed max_delta (5)
        parsed = {
            "adjustments": {"openness": 20},
            "reasoning": "test",
        }

        evolver._apply_adjustments(parsed, mgr.evolution_config)
        # Should be clamped to +5
        assert mgr.state.big_five.openness == 75

    def test_apply_adjustments_respects_bounds(self, tmp_character_dir):
        from src.mimosa.personality.personality_evolver import PersonalityEvolver

        mgr = PersonalityManager(config_dir=str(tmp_character_dir))

        # Set neuroticism to minimum boundary
        mgr.state.big_five.neuroticism = 12

        evolver = PersonalityEvolver(mgr, llm=None)

        parsed = {
            "adjustments": {"neuroticism": -5},
            "reasoning": "test",
        }

        evolver._apply_adjustments(parsed, mgr.evolution_config)
        # Should be clamped to min_value=10
        assert mgr.state.big_five.neuroticism == 10

    def test_no_change_returns_false(self, tmp_character_dir):
        from src.mimosa.personality.personality_evolver import PersonalityEvolver

        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        evolver = PersonalityEvolver(mgr, llm=None)

        parsed = {
            "adjustments": {"openness": 0, "neuroticism": 0},
            "reasoning": "no change needed",
        }

        changed = evolver._apply_adjustments(parsed, mgr.evolution_config)
        assert changed is False

    def test_parse_response_valid(self, tmp_character_dir):
        from src.mimosa.personality.personality_evolver import PersonalityEvolver

        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        evolver = PersonalityEvolver(mgr, llm=None)

        raw = json.dumps({
            "adjustments": {"openness": 3},
            "reasoning": "test",
        })
        result = evolver._parse_response(raw)
        assert result is not None
        assert result["adjustments"]["openness"] == 3

    def test_parse_response_with_markdown_fences(self, tmp_character_dir):
        from src.mimosa.personality.personality_evolver import PersonalityEvolver

        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        evolver = PersonalityEvolver(mgr, llm=None)

        raw = '```json\n{"adjustments": {"openness": 2}, "reasoning": "test"}\n```'
        result = evolver._parse_response(raw)
        assert result is not None
        assert result["adjustments"]["openness"] == 2

    def test_parse_response_invalid(self, tmp_character_dir):
        from src.mimosa.personality.personality_evolver import PersonalityEvolver

        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        evolver = PersonalityEvolver(mgr, llm=None)

        result = evolver._parse_response("not json at all")
        assert result is None
