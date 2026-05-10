# -*- coding: utf-8 -*-
"""Live2D model information manager."""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger


class Live2DModel:
    """Manages Live2D model info and emotion extraction."""

    def __init__(self, model_dict_path: str = "model_dict.json"):
        """Initialize Live2D model manager.

        :param model_dict_path: Path to the model dictionary JSON file.
        """
        self._models: Dict = {}
        self._current_model: Optional[str] = None
        self._load_model_dict(model_dict_path)

    def _load_model_dict(self, path: str):
        """Load model dictionary from JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            logger.warning(f"Model dict not found: {path}")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self._models = json.load(f)
            logger.info(f"Loaded {len(self._models)} Live2D model(s)")
        except Exception as e:
            logger.error(f"Failed to load model dict: {e}")

    def set_model(self, model_name: str) -> bool:
        """Set the current active model.

        :param model_name: Model name key from model_dict.
        :returns: True if model exists and was set.
        """
        if model_name in self._models:
            self._current_model = model_name
            logger.info(f"Active Live2D model: {model_name}")
            return True

        logger.warning(f"Model not found: {model_name}")
        return False

    def get_model_path(self) -> Optional[str]:
        """Get the current model's file path.

        :returns: Model path relative to project root.
        """
        if self._current_model and self._current_model in self._models:
            return self._models[self._current_model].get("model_path")
        return None

    def get_emotion_map(self) -> Dict[str, str]:
        """Get the emotion-to-expression mapping for current model.

        :returns: Dict mapping emotion names to expression file names.
        """
        if self._current_model and self._current_model in self._models:
            return self._models[self._current_model].get("emotion_map", {})
        return {}

    def extract_emotion(self, text: str) -> Tuple[str, str]:
        """Extract emotion tag from LLM response text.

        :param text: LLM response text containing emotion tag like [joy].
        :returns: Tuple of (clean_text, emotion_name). Defaults to 'neutral'.
        """
        # Pattern: [emotion] at the beginning of text
        pattern = r"^\[(\w+)\]\s*"
        match = re.match(pattern, text)

        if match:
            emotion = match.group(1).lower()
            clean_text = text[match.end():]

            # Validate emotion exists in our map
            emotion_map = self.get_emotion_map()
            if emotion in emotion_map:
                return clean_text, emotion

            logger.debug(f"Unknown emotion tag: {emotion}, using neutral")
            return clean_text, "neutral"

        return text, "neutral"

    def get_expression_name(self, emotion: str) -> Optional[str]:
        """Get the expression file name for an emotion.

        :param emotion: Emotion name (e.g., 'joy', 'sadness').
        :returns: Expression name (e.g., 'exp_02') or None.
        """
        emotion_map = self.get_emotion_map()
        return emotion_map.get(emotion)
