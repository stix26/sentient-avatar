import base64
import logging
from typing import Any, Dict, Optional, Tuple

from .base import BaseService

logger = logging.getLogger(__name__)


class TTSService(BaseService):
    """Service connection for TTS (Bark + XTTS)"""

    async def health_check(self) -> bool:
        """Check if TTS service is healthy"""
        try:
            response = await self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"TTS health check failed: {str(e)}")
            return False

    async def initialize(self) -> None:
        """Initialize TTS service connection"""
        if not await self.health_check():
            raise RuntimeError("TTS service is not healthy")

    async def cleanup(self) -> None:
        """Cleanup TTS service resources"""
        pass

    async def synthesize(
        self,
        text: str,
        voice: str = "en_speaker_6",
        model: str = "bark",
        temperature: float = 0.7,
        speed: float = 1.0,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Synthesize text to speech

        Args:
            text: Text to synthesize
            voice: Voice ID or name
            model: TTS model to use ('bark' or 'xtts')
            temperature: Sampling temperature
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio_data, metadata)
        """
        payload = {
            "text": text,
            "voice": voice,
            "model": model,
            "temperature": temperature,
            "speed": speed,
        }

        try:
            response = await self._request("POST", "/v1/synthesize", json=payload)

            # Decode base64 audio data
            audio_b64 = response.pop("audio")
            audio_data = base64.b64decode(audio_b64)

            return audio_data, response
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            raise

    async def clone_voice(
        self, reference_audio: bytes, text: str, language: str = "en"
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Clone voice from reference audio

        Args:
            reference_audio: Reference audio data in bytes
            text: Text to synthesize
            language: Language code

        Returns:
            Tuple of (audio_data, metadata)
        """
        # Encode reference audio as base64
        ref_audio_b64 = base64.b64encode(reference_audio).decode("utf-8")

        payload = {"reference_audio": ref_audio_b64, "text": text, "language": language}

        try:
            response = await self._request("POST", "/v1/clone", json=payload)

            # Decode base64 audio data
            audio_b64 = response.pop("audio")
            audio_data = base64.b64decode(audio_b64)

            return audio_data, response
        except Exception as e:
            logger.error(f"Error cloning voice: {str(e)}")
            raise

    async def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available voices

        Returns:
            Dict containing available voices and metadata
        """
        try:
            response = await self._request("GET", "/v1/voices")
            return response
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}")
            raise
