from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

class Vision:
    """Handles image analysis and processing"""
    
    def __init__(self, vision_service):
        self.vision_service = vision_service
    
    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        detail_level: str = "high"
    ) -> Dict[str, Any]:
        """
        Analyze image with a specific prompt
        
        Args:
            image_data: Image data in bytes
            prompt: Analysis prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            detail_level: Level of detail in analysis
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Analyze image
            result = await self.vision_service.analyze_image(
                image_data=image_data,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                detail_level=detail_level
            )
            
            return {
                "analysis": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    async def describe_image(
        self,
        image_data: bytes,
        style: str = "natural",
        include_objects: bool = True,
        include_actions: bool = True,
        include_attributes: bool = True
    ) -> Dict[str, Any]:
        """
        Generate detailed image description
        
        Args:
            image_data: Image data in bytes
            style: Description style
            include_objects: Whether to include object detection
            include_actions: Whether to include action detection
            include_attributes: Whether to include attribute detection
            
        Returns:
            Dict containing image description
        """
        try:
            # Describe image
            result = await self.vision_service.describe_image(
                image_data=image_data,
                style=style,
                include_objects=include_objects,
                include_actions=include_actions,
                include_attributes=include_attributes
            )
            
            return {
                "description": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error describing image: {e}")
            raise
    
    async def detect_objects(
        self,
        image_data: bytes,
        confidence_threshold: float = 0.5,
        max_objects: int = 10
    ) -> Dict[str, Any]:
        """
        Detect objects in image
        
        Args:
            image_data: Image data in bytes
            confidence_threshold: Minimum confidence score
            max_objects: Maximum number of objects to detect
            
        Returns:
            Dict containing detected objects
        """
        try:
            # Detect objects
            result = await self.vision_service.detect_objects(
                image_data=image_data,
                confidence_threshold=confidence_threshold,
                max_objects=max_objects
            )
            
            return {
                "objects": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            raise
    
    async def get_available_styles(self) -> List[Dict[str, Any]]:
        """
        Get available description styles
        
        Returns:
            List of available styles and their metadata
        """
        try:
            # Get styles
            styles = await self.vision_service.get_available_styles()
            
            return styles
            
        except Exception as e:
            logger.error(f"Error getting available styles: {e}")
            raise
    
    @staticmethod
    def validate_image(image_data: bytes) -> bool:
        """
        Validate image data
        
        Args:
            image_data: Image data in bytes
            
        Returns:
            Whether image is valid
        """
        try:
            # Try to open image
            Image.open(BytesIO(image_data))
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def resize_image(
        image_data: bytes,
        max_size: int = 1024,
        quality: int = 85
    ) -> bytes:
        """
        Resize image if needed
        
        Args:
            image_data: Image data in bytes
            max_size: Maximum dimension
            quality: JPEG quality
            
        Returns:
            Resized image data
        """
        try:
            # Open image
            image = Image.open(BytesIO(image_data))
            
            # Check if resize needed
            if max(image.size) <= max_size:
                return image_data
            
            # Calculate new size
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            
            # Resize image
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise 