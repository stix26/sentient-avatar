import json
import os
from typing import Any, Dict, Optional, Type, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base configuration class with validation."""

    class Config:
        """Pydantic config."""

        env_prefix = "SENTIENT_"
        case_sensitive = False

    @classmethod
    def from_env(cls: Type[T]) -> T:
        """Create config from environment variables.

        Returns:
            Config instance
        """
        return cls.parse_obj(
            {k: v for k, v in os.environ.items() if k.startswith(cls.Config.env_prefix)}
        )

    @classmethod
    def from_file(cls: Type[T], path: str) -> T:
        """Create config from file.

        Args:
            path: Path to config file

        Returns:
            Config instance
        """
        with open(path) as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                data = yaml.safe_load(f)
            elif path.endswith(".json"):
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path}")

        return cls.parse_obj(data)


class ServiceConfig(BaseConfig):
    """Service configuration."""

    url: str = Field(..., description="Service URL")
    timeout: int = Field(30, description="Request timeout in seconds")
    retries: int = Field(3, description="Number of retries")
    retry_delay: int = Field(1, description="Delay between retries in seconds")


class LLMConfig(ServiceConfig):
    """LLM service configuration."""

    model: str = Field(..., description="Model name")
    temperature: float = Field(0.7, description="Temperature")
    max_tokens: int = Field(2048, description="Maximum tokens")
    top_p: float = Field(1.0, description="Top P")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, description="Presence penalty")


class ASRConfig(ServiceConfig):
    """ASR service configuration."""

    model: str = Field(..., description="Model name")
    language: str = Field("en", description="Language")
    sample_rate: int = Field(16000, description="Sample rate")
    chunk_size: int = Field(1024, description="Chunk size")


class TTSConfig(ServiceConfig):
    """TTS service configuration."""

    model: str = Field(..., description="Model name")
    voice: str = Field(..., description="Voice ID")
    sample_rate: int = Field(24000, description="Sample rate")
    speed: float = Field(1.0, description="Speech speed")


class AvatarConfig(ServiceConfig):
    """Avatar service configuration."""

    model: str = Field(..., description="Model name")
    style: str = Field("default", description="Animation style")
    motion_scale: float = Field(1.0, description="Motion scale")
    stillness: float = Field(0.0, description="Stillness")


class VisionConfig(ServiceConfig):
    """Vision service configuration."""

    model: str = Field(..., description="Model name")
    max_size: int = Field(1024, description="Maximum image size")
    confidence_threshold: float = Field(0.5, description="Confidence threshold")


class VectorStoreConfig(ServiceConfig):
    """Vector store configuration."""

    collection: str = Field(..., description="Collection name")
    vector_size: int = Field(1536, description="Vector size")
    distance: str = Field("Cosine", description="Distance metric")


class CacheConfig(BaseConfig):
    """Cache configuration."""

    url: str = Field(..., description="Redis URL")
    ttl: int = Field(3600, description="Time to live in seconds")
    max_size: int = Field(1000, description="Maximum cache size")


class RateLimitConfig(BaseConfig):
    """Rate limit configuration."""

    url: str = Field(..., description="Redis URL")
    window: int = Field(60, description="Time window in seconds")
    max_requests: int = Field(100, description="Maximum requests per window")


class MetricsConfig(BaseConfig):
    """Metrics configuration."""

    port: int = Field(8000, description="Metrics server port")
    path: str = Field("/metrics", description="Metrics endpoint path")


class LoggingConfig(BaseConfig):
    """Logging configuration."""

    level: str = Field("INFO", description="Logging level")
    dir: str = Field("logs", description="Log directory")
    max_bytes: int = Field(10 * 1024 * 1024, description="Maximum log file size")
    backup_count: int = Field(5, description="Number of backup files")


class Config(BaseConfig):
    """Main configuration."""

    # Service configurations
    llm: LLMConfig
    asr: ASRConfig
    tts: TTSConfig
    avatar: AvatarConfig
    vision: VisionConfig
    vector_store: VectorStoreConfig

    # System configurations
    cache: CacheConfig
    rate_limit: RateLimitConfig
    metrics: MetricsConfig
    logging: LoggingConfig

    # Application settings
    debug: bool = Field(False, description="Debug mode")
    host: str = Field("0.0.0.0", description="Host")
    port: int = Field(8000, description="Port")
    workers: int = Field(1, description="Number of workers")

    @validator("port")
    def validate_port(cls, v: int) -> int:
        """Validate port number.

        Args:
            v: Port number

        Returns:
            Validated port number
        """
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @validator("workers")
    def validate_workers(cls, v: int) -> int:
        """Validate number of workers.

        Args:
            v: Number of workers

        Returns:
            Validated number of workers
        """
        if v < 1:
            raise ValueError("Number of workers must be at least 1")
        return v


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment.

    Args:
        config_path: Path to config file

    Returns:
        Config instance
    """
    # Load environment variables
    load_dotenv()

    if config_path:
        return Config.from_file(config_path)
    else:
        return Config.from_env()


def save_config(config: Config, path: str):
    """Save configuration to file.

    Args:
        config: Config instance
        path: Path to save config
    """
    with open(path, "w") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            yaml.dump(config.dict(), f)
        elif path.endswith(".json"):
            json.dump(config.dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path}")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration.

    Returns:
        Default configuration
    """
    return {
        "llm": {
            "url": "http://localhost:8001",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "asr": {
            "url": "http://localhost:8002",
            "model": "whisper-large-v3",
            "language": "en",
            "sample_rate": 16000,
        },
        "tts": {
            "url": "http://localhost:8003",
            "model": "tts-1",
            "voice": "alloy",
            "sample_rate": 24000,
        },
        "avatar": {
            "url": "http://localhost:8004",
            "model": "sadtalker",
            "style": "default",
            "motion_scale": 1.0,
        },
        "vision": {
            "url": "http://localhost:8005",
            "model": "llava-next",
            "max_size": 1024,
        },
        "vector_store": {
            "url": "http://localhost:8006",
            "collection": "memories",
            "vector_size": 1536,
        },
        "cache": {"url": "redis://localhost:6379/0", "ttl": 3600},
        "rate_limit": {
            "url": "redis://localhost:6379/1",
            "window": 60,
            "max_requests": 100,
        },
        "metrics": {"port": 8000, "path": "/metrics"},
        "logging": {
            "level": "INFO",
            "dir": "logs",
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 5,
        },
        "debug": False,
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
    }
