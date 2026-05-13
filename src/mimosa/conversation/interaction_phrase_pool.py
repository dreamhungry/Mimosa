# -*- coding: utf-8 -*-
"""Interaction phrase pool for caching LLM-generated click responses."""

import json
import random
from pathlib import Path
from typing import List, Optional

from loguru import logger


class InteractionPhrasePool:
    """Manages a pool of LLM-generated interaction phrases with file persistence.

    Phrases are stored in a JSON file and loaded into memory on init.
    New phrases are deduplicated and subject to FIFO eviction at max capacity.
    """

    def __init__(
        self,
        storage_path: str = "character/interaction_phrases.json",
        max_phrases: int = 100,
    ):
        """Initialize phrase pool.

        :param storage_path: Path to the JSON persistence file.
        :param max_phrases: Maximum number of cached phrases (FIFO eviction).
        """
        self._path = Path(storage_path)
        self._max_phrases = max_phrases
        self._phrases: List[str] = []

        self._load()

    @property
    def phrases(self) -> List[str]:
        """Get all cached phrases."""
        return list(self._phrases)

    @property
    def count(self) -> int:
        """Get current pool size."""
        return len(self._phrases)

    def get_random(self) -> Optional[str]:
        """Get a random phrase from the pool.

        :returns: A random phrase, or None if pool is empty.
        """
        if not self._phrases:
            return None
        return random.choice(self._phrases)

    def add(self, phrase: str) -> bool:
        """Add a new phrase to the pool.

        Deduplicates and applies FIFO eviction when over capacity.

        :param phrase: The phrase to add.
        :returns: True if the phrase was actually added (not duplicate).
        """
        if not phrase or not isinstance(phrase, str):
            return False

        phrase = phrase.strip()
        if not phrase:
            return False

        # Deduplicate
        if phrase in self._phrases:
            return False

        self._phrases.append(phrase)

        # FIFO eviction
        while len(self._phrases) > self._max_phrases:
            self._phrases.pop(0)

        self._save()
        logger.debug(f"Phrase pool: added new phrase, total={len(self._phrases)}")
        return True

    def _load(self):
        """Load phrases from JSON file."""
        if not self._path.exists():
            self._phrases = []
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                self._phrases = [p for p in data if isinstance(p, str) and p.strip()]
            else:
                self._phrases = []

            logger.info(
                f"Loaded {len(self._phrases)} interaction phrases from {self._path}"
            )
        except Exception as e:
            logger.error(f"Failed to load interaction phrases: {e}")
            self._phrases = []

    def _save(self):
        """Save phrases to JSON file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._phrases, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save interaction phrases: {e}")
