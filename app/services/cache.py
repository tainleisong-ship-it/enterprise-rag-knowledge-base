import hashlib
import json
import logging
from typing import Any, Optional

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis 缓存封装，连接失败时自动降级为无缓存模式。"""

    def __init__(self) -> None:
        self._client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            logger.info("Redis connected: %s", settings.redis_url)
        except Exception as e:
            logger.warning("Redis unavailable, cache disabled: %s", e)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    @staticmethod
    def make_key(question: str) -> str:
        digest = hashlib.md5(question.encode("utf-8")).hexdigest()
        return f"rag:query:{digest}"

    def get(self, key: str) -> Optional[Any]:
        if not self._client:
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning("Cache get failed: %s", e)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self._client:
            return
        try:
            self._client.set(
                key,
                json.dumps(value, ensure_ascii=False),
                ex=ttl or settings.redis_ttl,
            )
        except Exception as e:
            logger.warning("Cache set failed: %s", e)


cache = RedisCache()
