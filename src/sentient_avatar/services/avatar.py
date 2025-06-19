import base64
import logging
from typing import Any, Dict, Optional, Tuple

from .base import BaseService

logger = logging.getLogger(__name__)


class AvatarService(BaseService):
    """Service connection for Avatar (SadTalker)"""

    async def health_check(self) -> bool:
        """Check if Avatar service is healthy"""
        try:
            response = await self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Avatar health check failed: {str(e)}")
            return False

    async def initialize(self) -> None:
        """Initialize Avatar service connection"""
        if not await self.health_check():
            raise RuntimeError("Avatar service is not healthy")

    async def cleanup(self) -> None:
        """Cleanup Avatar service resources"""
        pass

    async def generate_video(
        self,
        audio_data: bytes,
        reference_image: Optional[bytes] = None,
        style: str = "default",
        motion_scale: float = 1.0,
        still: bool = False,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate talking head video from audio

        Args:
            audio_data: Audio data in bytes
            reference_image: Optional reference face image
            style: Animation style ('default', 'happy', 'serious', etc.)
            motion_scale: Motion intensity multiplier
            still: Whether to keep head still

        Returns:
            Tuple of (video_data, metadata)
        """
        # Encode audio and image as base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        payload = {
            "audio": audio_b64,
            "style": style,
            "motion_scale": motion_scale,
            "still": still,
        }

        if reference_image:
            image_b64 = base64.b64encode(reference_image).decode("utf-8")
            payload["reference_image"] = image_b64

        try:
            response = await self._request("POST", "/v1/generate", json=payload)

            # Decode base64 video data
            video_b64 = response.pop("video")
            video_data = base64.b64decode(video_b64)

            return video_data, response
        except Exception as e:
            logger.error(f"Error generating avatar video: {str(e)}")
            raise

    async def generate_stream(
        self,
        audio_chunk: bytes,
        is_final: bool = False,
        reference_image: Optional[bytes] = None,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Stream video generation for real-time avatar

        Args:
            audio_chunk: Chunk of audio data
            is_final: Whether this is the final chunk
            reference_image: Optional reference face image

        Returns:
            Tuple of (video_data, metadata)
        """
        audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
        payload = {"audio": audio_b64, "is_final": is_final}

        if reference_image:
            image_b64 = base64.b64encode(reference_image).decode("utf-8")
            payload["reference_image"] = image_b64

        try:
            response = await self._request("POST", "/v1/stream", json=payload)

            # Decode base64 video data
            video_b64 = response.pop("video")
            video_data = base64.b64decode(video_b64)

            return video_data, response
        except Exception as e:
            logger.error(f"Error streaming avatar video: {str(e)}")
            raise

    async def get_available_styles(self) -> Dict[str, Any]:
        """
        Get list of available animation styles

        Returns:
            Dict containing available styles and metadata
        """
        try:
            response = await self._request("GET", "/v1/styles")
            return response
        except Exception as e:
            logger.error(f"Error getting available styles: {str(e)}")
            raise
