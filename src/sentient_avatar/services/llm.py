from typing import Dict, Any, List, Optional
from .base import BaseService
import logging

logger = logging.getLogger(__name__)


class LLMService(BaseService):
    """Service connection for LLM (vLLM or llama.cpp)"""

    async def health_check(self) -> bool:
        """Check if LLM service is healthy"""
        try:
            response = await self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return False

    async def initialize(self) -> None:
        """Initialize LLM service connection"""
        if not await self.health_check():
            raise RuntimeError("LLM service is not healthy")

    async def cleanup(self) -> None:
        """Cleanup LLM service resources"""
        pass

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text completion from LLM

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: List of stop sequences
            stream: Whether to stream the response

        Returns:
            Dict containing generated text and metadata
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop

        try:
            response = await self._request("POST", "/v1/completions", json=payload)
            return response
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate chat completion from LLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stream: Whether to stream the response

        Returns:
            Dict containing generated message and metadata
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        try:
            response = await self._request("POST", "/v1/chat/completions", json=payload)
            return response
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise
