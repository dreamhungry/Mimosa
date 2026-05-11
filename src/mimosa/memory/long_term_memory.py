# -*- coding: utf-8 -*-
"""Long-term memory management using Markdown file storage.

Extracts and persists key facts from conversations using LLM,
then injects them into the system prompt for future sessions.
"""

from pathlib import Path
from typing import Optional

from loguru import logger


# Prompt for LLM to extract memorable facts from a conversation
MEMORY_EXTRACTION_PROMPT = """You are a memory extraction assistant. Analyze the following conversation and extract important facts worth remembering long-term.

Focus on:
- Personal information about the user (name, preferences, family, pets, hobbies, work, etc.)
- Important events or dates mentioned
- User preferences and habits
- Relationship context (how the user addresses you, tone preferences)

Output format: Return ONLY the new facts as a markdown list. Each fact should be a single concise line starting with "- ".
If there are no new facts worth remembering, respond with exactly "NONE".

Do NOT include:
- Transient conversation topics (weather chat, greetings)
- Facts that are already in the existing memory
- Opinions or speculation — only concrete facts stated by the user

Existing memory (do not repeat these):
{existing_memory}

Conversation to analyze:
{conversation}"""


class LongTermMemory:
    """Manages persistent long-term memory stored as Markdown.

    Memory file structure:
    ```
    # User Facts
    - fact 1
    - fact 2

    # Preferences
    - preference 1

    # Important Events
    - YYYY-MM-DD: event description
    ```
    """

    def __init__(self, memory_dir: str = "memory"):
        """Initialize long-term memory manager.

        :param memory_dir: Directory for storing memory files.
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory_file = self.memory_dir / "long_term_memory.md"
        self._content: Optional[str] = None

    @property
    def content(self) -> str:
        """Get current memory content, loading from disk if needed."""
        if self._content is None:
            self._content = self._load()
        return self._content

    def _load(self) -> str:
        """Load memory from Markdown file.

        :returns: Memory content as string, empty string if file not found.
        """
        if not self._memory_file.exists():
            return ""

        try:
            text = self._memory_file.read_text(encoding="utf-8").strip()
            logger.info(f"Loaded long-term memory: {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"Failed to load long-term memory: {e}")
            return ""

    def _save(self, content: str):
        """Save memory content to Markdown file.

        :param content: Full memory content to write.
        """
        try:
            self._memory_file.write_text(content, encoding="utf-8")
            self._content = content
            logger.info(f"Saved long-term memory: {len(content)} chars")
        except Exception as e:
            logger.error(f"Failed to save long-term memory: {e}")

    def get_prompt_injection(self) -> str:
        """Get memory content formatted for injection into system prompt.

        :returns: Formatted memory string, or empty string if no memory.
        """
        mem = self.content
        if not mem:
            return ""
        return (
            "\n\n## Long-Term Memory\n"
            "The following are facts you remember about the user from "
            "previous conversations. Use them naturally when relevant, "
            "but don't explicitly mention that you're recalling from memory.\n\n"
            f"{mem}"
        )

    def build_extraction_prompt(self, conversation_text: str) -> str:
        """Build the prompt for LLM to extract new facts.

        :param conversation_text: Formatted conversation to analyze.
        :returns: Complete extraction prompt.
        """
        return MEMORY_EXTRACTION_PROMPT.format(
            existing_memory=self.content or "(empty)",
            conversation=conversation_text,
        )

    def update_from_extraction(self, extracted: str):
        """Update memory with newly extracted facts from LLM.

        :param extracted: Raw LLM output containing new facts.
        """
        extracted = extracted.strip()
        if not extracted or extracted.upper() == "NONE":
            logger.debug("No new facts to remember")
            return

        # Parse the extracted facts (lines starting with "- ")
        new_facts = []
        for line in extracted.splitlines():
            line = line.strip()
            if line.startswith("- "):
                new_facts.append(line)

        if not new_facts:
            logger.debug("No valid fact lines found in extraction")
            return

        current = self.content
        if not current:
            # Initialize with default structure
            current = "# User Facts\n"

        # Append new facts under the appropriate section
        # Simple approach: append at end
        updated = current.rstrip() + "\n" + "\n".join(new_facts) + "\n"
        self._save(updated)
        logger.info(f"Added {len(new_facts)} new facts to long-term memory")
