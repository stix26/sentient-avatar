import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import aioredis

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using Redis for distributed rate limiting."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
        """
        self.redis = aioredis.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )

    async def is_rate_limited(
        self, key: str, max_requests: int, window: int, cost: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited.

        Args:
            key: Rate limit key (e.g., IP address or user ID)
            max_requests: Maximum number of requests allowed in window
            window: Time window in seconds
            cost: Cost of the request (default: 1)

        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        try:
            now = int(time.time())
            window_key = f"{key}:{now // window}"

            # Get current count and window info
            pipe = self.redis.pipeline()
            pipe.get(window_key)
            pipe.ttl(window_key)
            current_count, ttl = await pipe.execute()

            current_count = int(current_count) if current_count else 0

            # Check if rate limited
            if current_count + cost > max_requests:
                reset_time = now + (ttl if ttl > 0 else window)
                return True, {
                    "limited": True,
                    "remaining": 0,
                    "reset": reset_time,
                    "retry_after": reset_time - now,
                }

            # Increment counter
            if not current_count:
                await self.redis.setex(window_key, window, cost)
            else:
                await self.redis.incrby(window_key, cost)

            return False, {
                "limited": False,
                "remaining": max_requests - (current_count + cost),
                "reset": now + window,
                "retry_after": 0,
            }

        except Exception as e:
            logger.error(f"Error checking rate limit for {key}: {e}")
            return False, {
                "limited": False,
                "remaining": max_requests,
                "reset": now + window,
                "retry_after": 0,
            }

    async def get_rate_limit_info(
        self, key: str, max_requests: int, window: int
    ) -> Dict[str, Any]:
        """Get rate limit information.

        Args:
            key: Rate limit key
            max_requests: Maximum number of requests allowed in window
            window: Time window in seconds

        Returns:
            Rate limit information
        """
        try:
            now = int(time.time())
            window_key = f"{key}:{now // window}"

            # Get current count and window info
            pipe = self.redis.pipeline()
            pipe.get(window_key)
            pipe.ttl(window_key)
            current_count, ttl = await pipe.execute()

            current_count = int(current_count) if current_count else 0
            reset_time = now + (ttl if ttl > 0 else window)

            return {
                "limited": current_count >= max_requests,
                "remaining": max(0, max_requests - current_count),
                "reset": reset_time,
                "retry_after": max(0, reset_time - now),
            }

        except Exception as e:
            logger.error(f"Error getting rate limit info for {key}: {e}")
            return {
                "limited": False,
                "remaining": max_requests,
                "reset": now + window,
                "retry_after": 0,
            }

    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a key.

        Args:
            key: Rate limit key

        Returns:
            True if successful
        """
        try:
            # Delete all rate limit keys for this key
            pattern = f"{key}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Error resetting rate limit for {key}: {e}")
            return False

    async def get_all_rate_limits(self) -> Dict[str, Dict[str, Any]]:
        """Get all rate limits.

        Returns:
            Dictionary of rate limit information
        """
        try:
            # Get all rate limit keys
            keys = await self.redis.keys("*:*")
            if not keys:
                return {}

            # Get values for all keys
            pipe = self.redis.pipeline()
            for key in keys:
                pipe.get(key)
                pipe.ttl(key)
            results = await pipe.execute()

            # Process results
            rate_limits = {}
            for i in range(0, len(results), 2):
                key = keys[i // 2]
                count = int(results[i]) if results[i] else 0
                ttl = results[i + 1]

                base_key = key.split(":")[0]
                if base_key not in rate_limits:
                    rate_limits[base_key] = {"count": 0, "windows": {}}

                rate_limits[base_key]["count"] += count
                rate_limits[base_key]["windows"][key] = {"count": count, "ttl": ttl}

            return rate_limits

        except Exception as e:
            logger.error(f"Error getting all rate limits: {e}")
            return {}

    async def clear_all_rate_limits(self) -> bool:
        """Clear all rate limits.

        Returns:
            True if successful
        """
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing all rate limits: {e}")
            return False

    def rate_limit(
        self,
        max_requests: int,
        window: int,
        key_func: Optional[callable] = None,
        cost: int = 1,
    ):
        """Decorator for rate limiting functions.

        Args:
            max_requests: Maximum number of requests allowed in window
            window: Time window in seconds
            key_func: Function to generate rate limit key
            cost: Cost of the request

        Returns:
            Decorated function
        """

        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate rate limit key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    key = f"{func.__module__}:{func.__name__}"

                # Check rate limit
                is_limited, info = await self.is_rate_limited(
                    key, max_requests, window, cost
                )

                if is_limited:
                    raise Exception(
                        f"Rate limit exceeded. Try again in {info['retry_after']} seconds."
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator
