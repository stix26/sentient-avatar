from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SentientAgent:
    """Simplified agent used for testing."""

    def __init__(self, llm_service, memory_window: int = 10) -> None:
        self.llm_service = llm_service
        self.memory_window = memory_window
        self.conversation_history: List[Dict[str, Any]] = []

    async def process_input(self, text: str) -> str:
        """Process user text and return the LLM response."""
        response = await self.llm_service.generate(text)
        resp_text = response["text"] if isinstance(response, dict) else response
        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append({"role": "assistant", "content": resp_text})
        if len(self.conversation_history) > self.memory_window * 2:
            self.conversation_history = self.conversation_history[
                -self.memory_window * 2 :
            ]
        return resp_text

    async def execute_task(self, task: str) -> str:
        """Execute a task via the LLM service."""
        response = await self.llm_service.chat(task)
        return response["text"] if isinstance(response, dict) else response

    async def get_memory_context(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Placeholder for compatibility with older API."""
        logger.debug("get_memory_context called with query=%s", query)
        return []
