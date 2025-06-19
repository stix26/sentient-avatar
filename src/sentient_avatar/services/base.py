import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Base class for all service connections"""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        if not self._session:
            raise RuntimeError(
                "Service not initialized. Use 'async with' context manager."
            )

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            async with self._session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize service connection"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        pass
