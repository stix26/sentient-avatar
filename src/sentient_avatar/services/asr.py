import base64
import logging
from typing import Any, Dict, Optional

from .base import BaseService

logger = logging.getLogger(__name__)


class ASRService(BaseService):
    """Service connection for ASR (whisper.cpp)"""

    async def health_check(self) -> bool:
        """Check if ASR service is healthy"""
        try:
            response = await self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"ASR health check failed: {str(e)}")
            return False

    async def initialize(self) -> None:
        """Initialize ASR service connection"""
        if not await self.health_check():
            raise RuntimeError("ASR service is not healthy")

    async def cleanup(self) -> None:
        """Cleanup ASR service resources"""

    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text

        Args:
            audio_data: Raw audio data in bytes
            language: Language code (e.g., 'en', 'es')
            task: Task type ('transcribe' or 'translate')
            initial_prompt: Optional initial prompt for better accuracy
            word_timestamps: Whether to include word-level timestamps

        Returns:
            Dict containing transcription and metadata
        """
        # Encode audio data as base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        payload = {"audio": audio_b64, "task": task, "word_timestamps": word_timestamps}

        if language:
            payload["language"] = language
        if initial_prompt:
            payload["initial_prompt"] = initial_prompt

        try:
            response = await self._request("POST", "/v1/transcribe", json=payload)
            return response
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise

    async def transcribe_stream(
        self, audio_chunk: bytes, is_final: bool = False, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stream audio chunks for real-time transcription

        Args:
            audio_chunk: Chunk of audio data
            is_final: Whether this is the final chunk
            language: Language code

        Returns:
            Dict containing partial transcription and metadata
        """
        audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")

        payload = {"audio": audio_b64, "is_final": is_final}

        if language:
            payload["language"] = language

        try:
            response = await self._request(
                "POST", "/v1/transcribe/stream", json=payload
            )
            return response
        except Exception as e:
            logger.error(f"Error streaming audio transcription: {str(e)}")
            raise
