import base64
import logging
from typing import Any, Dict

from .base import BaseService

logger = logging.getLogger(__name__)


class VisionService(BaseService):
    """Service connection for Vision (LLaVA-NeXT)"""

    async def health_check(self) -> bool:
        """Check if Vision service is healthy"""
        try:
            response = await self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Vision health check failed: {str(e)}")
            return False

    async def initialize(self) -> None:
        """Initialize Vision service connection"""
        if not await self.health_check():
            raise RuntimeError("Vision service is not healthy")

    async def cleanup(self) -> None:
        """Cleanup Vision service resources"""

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        detail_level: str = "high",
    ) -> Dict[str, Any]:
        """
        Analyze image with visual question answering

        Args:
            image_data: Image data in bytes
            prompt: Question or instruction about the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            detail_level: Analysis detail level ('low', 'medium', 'high')

        Returns:
            Dict containing analysis and metadata
        """
        # Encode image as base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        payload = {
            "image": image_b64,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "detail_level": detail_level,
        }

        try:
            response = await self._request("POST", "/v1/analyze", json=payload)
            return response
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise

    async def describe_image(
        self,
        image_data: bytes,
        style: str = "natural",
        include_objects: bool = True,
        include_actions: bool = True,
        include_attributes: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate detailed image description

        Args:
            image_data: Image data in bytes
            style: Description style ('natural', 'technical', 'poetic')
            include_objects: Whether to include object detection
            include_actions: Whether to include action detection
            include_attributes: Whether to include attribute detection

        Returns:
            Dict containing description and metadata
        """
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        payload = {
            "image": image_b64,
            "style": style,
            "include_objects": include_objects,
            "include_actions": include_actions,
            "include_attributes": include_attributes,
        }

        try:
            response = await self._request("POST", "/v1/describe", json=payload)
            return response
        except Exception as e:
            logger.error(f"Error describing image: {str(e)}")
            raise

    async def detect_objects(
        self,
        image_data: bytes,
        confidence_threshold: float = 0.5,
        max_objects: int = 20,
    ) -> Dict[str, Any]:
        """
        Detect objects in image

        Args:
            image_data: Image data in bytes
            confidence_threshold: Minimum confidence score
            max_objects: Maximum number of objects to detect

        Returns:
            Dict containing detected objects and metadata
        """
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        payload = {
            "image": image_b64,
            "confidence_threshold": confidence_threshold,
            "max_objects": max_objects,
        }

        try:
            response = await self._request("POST", "/v1/detect", json=payload)
            return response
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            raise

    async def get_available_styles(self) -> Dict[str, Any]:
        """
        Get list of available description styles

        Returns:
            Dict containing available styles and metadata
        """
        try:
            response = await self._request("GET", "/v1/styles")
            return response
        except Exception as e:
            logger.error(f"Error getting available styles: {str(e)}")
            raise
