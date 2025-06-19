from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image

logger = logging.getLogger(__name__)


class Vision:
    """Minimal vision wrapper used for tests."""

    def __init__(self, vision_service) -> None:
        self.vision_service = vision_service

    async def analyze_image(
        self, image_data: bytes, prompt: str | None = None
    ) -> Dict[str, Any]:
        return await self.vision_service.analyze_image(image_data)

    async def describe_image(self, image_data: bytes) -> Dict[str, Any]:
        return await self.vision_service.describe_image(image_data)

    async def detect_objects(self, image_data: bytes) -> Dict[str, Any]:
        return await self.vision_service.detect_objects(image_data)

    async def get_available_styles(self) -> List[Dict[str, Any]]:
        return await self.vision_service.get_available_styles()

    @staticmethod
    def validate_image(image_data: bytes) -> bool:
        try:
            Image.open(BytesIO(image_data))
            return True
        except Exception:
            return False
