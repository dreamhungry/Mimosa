# -*- coding: utf-8 -*-
"""Tests for REST API endpoints (personality + LLM config)."""

import pytest
from fastapi.testclient import TestClient

from src.mimosa.api_routes import create_api_router
from src.mimosa.config import LLMConfig, MimosaConfig
from src.mimosa.personality.personality_manager import PersonalityManager
from src.mimosa.service_context import ServiceContext


@pytest.fixture
def tmp_character_dir(tmp_path):
    """Create a temporary character directory with baseline config."""
    import yaml

    char_dir = tmp_path / "character"
    char_dir.mkdir()

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
            "quirks": [],
        },
        "evolution": {
            "enabled": False,
            "interval": 100,
            "max_delta": 5,
        },
    }
    with open(char_dir / "personality.yaml", "w") as f:
        yaml.dump(baseline, f)

    return char_dir


@pytest.fixture
def client(tmp_character_dir):
    """Create a test client with mocked ServiceContext."""
    from unittest.mock import MagicMock
    from fastapi import FastAPI

    config = MimosaConfig(
        llm=LLMConfig(
            provider="openai_compatible",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="sk-test-1234567890",
            temperature=0.7,
            max_tokens=1024,
        ),
    )

    ctx = MagicMock(spec=ServiceContext)
    ctx.config = config
    ctx.personality = PersonalityManager(config_dir=str(tmp_character_dir))

    # Wire up update_llm_config to actually modify config
    def mock_update_llm(model=None, temperature=None, max_tokens=None, api_key=None):
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
        if updates:
            ctx.config = ctx.config.model_copy(
                update={"llm": ctx.config.llm.model_copy(update=updates)}
            )

    ctx.update_llm_config = mock_update_llm

    app = FastAPI()
    app.include_router(create_api_router(ctx))

    return TestClient(app)


# ---------------------------------------------------------------
# Personality endpoints
# ---------------------------------------------------------------


class TestPersonalityAPI:
    """Tests for /api/personality endpoints."""

    def test_get_personality(self, client):
        res = client.get("/api/personality")
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["big_five"]["openness"] == 70
        assert data["style"]["humor_style"] == "gentle"

    def test_update_personality_big_five(self, client):
        res = client.put(
            "/api/personality",
            json={"big_five": {"openness": 85, "neuroticism": 20}},
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["big_five"]["openness"] == 85
        assert data["big_five"]["neuroticism"] == 20
        # Unchanged traits stay the same
        assert data["big_five"]["extraversion"] == 60

    def test_update_personality_style(self, client):
        res = client.put(
            "/api/personality",
            json={"style": {"humor_style": "witty", "speech_formality": 60}},
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["style"]["humor_style"] == "witty"
        assert data["style"]["speech_formality"] == 60

    def test_update_personality_invalid_trait(self, client):
        res = client.put(
            "/api/personality",
            json={"big_five": {"openness": 150}},
        )
        assert res.status_code == 400

    def test_update_personality_unknown_trait(self, client):
        res = client.put(
            "/api/personality",
            json={"big_five": {"happiness": 50}},
        )
        assert res.status_code == 400

    def test_update_personality_invalid_humor(self, client):
        res = client.put(
            "/api/personality",
            json={"style": {"humor_style": "silly"}},
        )
        assert res.status_code == 400

    def test_reset_personality(self, client):
        # First modify
        client.put("/api/personality", json={"big_five": {"openness": 99}})
        # Then reset
        res = client.post("/api/personality/reset")
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["big_five"]["openness"] == 70


# ---------------------------------------------------------------
# LLM config endpoints
# ---------------------------------------------------------------


class TestLLMConfigAPI:
    """Tests for /api/config/llm endpoints."""

    def test_get_llm_config(self, client):
        res = client.get("/api/config/llm")
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["model"] == "gpt-4o-mini"
        assert data["temperature"] == 0.7
        # API key should be masked
        assert data["api_key"].startswith("****")
        assert data["api_key"].endswith("7890")

    def test_update_llm_model(self, client):
        res = client.put("/api/config/llm", json={"model": "gpt-4o"})
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["model"] == "gpt-4o"

    def test_update_llm_temperature(self, client):
        res = client.put("/api/config/llm", json={"temperature": 1.2})
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["temperature"] == 1.2

    def test_update_llm_invalid_temperature(self, client):
        res = client.put("/api/config/llm", json={"temperature": 3.0})
        assert res.status_code == 400

    def test_update_llm_max_tokens(self, client):
        res = client.put("/api/config/llm", json={"max_tokens": 2048})
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["max_tokens"] == 2048


class TestPersonalityManagerUpdate:
    """Tests for PersonalityManager.update_state method."""

    def test_update_big_five(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        mgr.update_state(big_five={"openness": 90})
        assert mgr.state.big_five.openness == 90

    def test_update_style(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        mgr.update_state(style={"humor_style": "dry", "speech_formality": 80})
        assert mgr.state.style.humor_style == "dry"
        assert mgr.state.style.speech_formality == 80

    def test_invalid_trait_value(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        with pytest.raises(ValueError, match="0~100"):
            mgr.update_state(big_five={"openness": -5})

    def test_unknown_trait(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        with pytest.raises(ValueError, match="Unknown Big Five trait"):
            mgr.update_state(big_five={"happiness": 50})

    def test_invalid_humor_style(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        with pytest.raises(ValueError, match="humor_style"):
            mgr.update_state(style={"humor_style": "silly"})

    def test_persisted_to_disk(self, tmp_character_dir):
        mgr = PersonalityManager(config_dir=str(tmp_character_dir))
        mgr.update_state(big_five={"openness": 42})

        # Reload from disk
        mgr2 = PersonalityManager(config_dir=str(tmp_character_dir))
        assert mgr2.state.big_five.openness == 42
