from typing import Any, Dict, Optional, Type

from ..cache.redis_cache import RedisCache
from ..config.config import Config
from ..logging.logger import get_logger
from ..monitoring.metrics import MetricsCollector
from ..rate_limit.rate_limiter import RateLimiter
from .asr import ASRService
from .avatar import AvatarService
from .llm import LLMService
from .tts import TTSService
from .vector_store import VectorStoreService
from .vision import VisionService

logger = get_logger(__name__)


class ServiceFactory:
    """Factory for creating and managing service instances."""

    def __init__(self, config: Config):
        """Initialize service factory.

        Args:
            config: Configuration instance
        """
        self.config = config
        self._services: Dict[str, Any] = {}
        self._cache = None
        self._rate_limiter = None
        self._metrics = None

    @property
    def cache(self) -> RedisCache:
        """Get cache instance.

        Returns:
            Cache instance
        """
        if self._cache is None:
            self._cache = RedisCache(
                url=self.config.cache.url,
                ttl=self.config.cache.ttl,
                max_size=self.config.cache.max_size,
            )
        return self._cache

    @property
    def rate_limiter(self) -> RateLimiter:
        """Get rate limiter instance.

        Returns:
            Rate limiter instance
        """
        if self._rate_limiter is None:
            self._rate_limiter = RateLimiter(
                url=self.config.rate_limit.url,
                window=self.config.rate_limit.window,
                max_requests=self.config.rate_limit.max_requests,
            )
        return self._rate_limiter

    @property
    def metrics(self) -> MetricsCollector:
        """Get metrics collector instance.

        Returns:
            Metrics collector instance
        """
        if self._metrics is None:
            self._metrics = MetricsCollector(port=self.config.metrics.port)
        return self._metrics

    def get_service(self, service_type: str) -> Any:
        """Get service instance.

        Args:
            service_type: Service type

        Returns:
            Service instance
        """
        if service_type not in self._services:
            self._services[service_type] = self._create_service(service_type)
        return self._services[service_type]

    def _create_service(self, service_type: str) -> Any:
        """Create service instance.

        Args:
            service_type: Service type

        Returns:
            Service instance
        """
        service_map = {
            "llm": LLMService,
            "asr": ASRService,
            "tts": TTSService,
            "avatar": AvatarService,
            "vision": VisionService,
            "vector_store": VectorStoreService,
        }

        if service_type not in service_map:
            raise ValueError(f"Unknown service type: {service_type}")

        service_class = service_map[service_type]
        service_config = getattr(self.config, service_type)

        return service_class(
            url=service_config.url,
            timeout=service_config.timeout,
            retries=service_config.retries,
            retry_delay=service_config.retry_delay,
            cache=self.cache,
            rate_limiter=self.rate_limiter,
            metrics=self.metrics,
        )

    async def initialize(self):
        """Initialize all services."""
        for service_type in self._services:
            service = self._services[service_type]
            if hasattr(service, "initialize"):
                await service.initialize()

    async def cleanup(self):
        """Cleanup all services."""
        for service_type in self._services:
            service = self._services[service_type]
            if hasattr(service, "cleanup"):
                await service.cleanup()

    def get_llm(self) -> LLMService:
        """Get LLM service instance.

        Returns:
            LLM service instance
        """
        return self.get_service("llm")

    def get_asr(self) -> ASRService:
        """Get ASR service instance.

        Returns:
            ASR service instance
        """
        return self.get_service("asr")

    def get_tts(self) -> TTSService:
        """Get TTS service instance.

        Returns:
            TTS service instance
        """
        return self.get_service("tts")

    def get_avatar(self) -> AvatarService:
        """Get avatar service instance.

        Returns:
            Avatar service instance
        """
        return self.get_service("avatar")

    def get_vision(self) -> VisionService:
        """Get vision service instance.

        Returns:
            Vision service instance
        """
        return self.get_service("vision")

    def get_vector_store(self) -> VectorStoreService:
        """Get vector store service instance.

        Returns:
            Vector store service instance
        """
        return self.get_service("vector_store")

    @classmethod
    def create(cls, config_path: Optional[str] = None) -> "ServiceFactory":
        """Create service factory.

        Args:
            config_path: Path to config file

        Returns:
            Service factory instance
        """
        from ..config.config import load_config

        config = load_config(config_path)
        return cls(config)
