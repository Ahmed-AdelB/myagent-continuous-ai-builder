"""
Cache Eviction Policies for Memory Management

Implements LRU, LFU, and TTL-based eviction strategies to prevent
memory leaks in long-running autonomous operations.

Prevents the system from accumulating unbounded memory entries
during 24/7 continuous operation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict
from loguru import logger
import heapq


class EvictionPolicy(Enum):
    """Available cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    HYBRID = "hybrid"  # Combination of LRU + TTL


@dataclass
class CacheEntry:
    """Cache entry with access tracking"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None

    def access(self):
        """Record an access to this entry"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def get_lru_priority(self) -> float:
        """Get LRU priority (older access = higher priority for eviction)"""
        return (datetime.now() - self.last_accessed).total_seconds()

    def get_lfu_priority(self) -> int:
        """Get LFU priority (lower access count = higher priority for eviction)"""
        return -self.access_count  # Negative for min-heap


class CacheEvictionManager:
    """
    Manages cache eviction using configurable policies

    Prevents memory leaks by enforcing limits on:
    - Total number of entries
    - Total memory usage
    - Entry age (TTL)
    """

    def __init__(
        self,
        max_entries: int = 10000,
        max_size_bytes: int = 500 * 1024 * 1024,  # 500 MB default
        policy: EvictionPolicy = EvictionPolicy.HYBRID,
        default_ttl_seconds: int = 3600 * 24 * 7,  # 7 days default
        eviction_threshold: float = 0.9  # Evict when 90% full
    ):
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self.policy = policy
        self.default_ttl_seconds = default_ttl_seconds
        self.eviction_threshold = eviction_threshold

        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}

        # LRU tracking (OrderedDict for O(1) reordering)
        self.lru_tracker: OrderedDict[str, None] = OrderedDict()

        # LFU tracking (min-heap by access count)
        self.lfu_heap: List[tuple] = []

        # Statistics
        self.stats = {
            'total_accesses': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'ttl_evictions': 0,
            'policy_evictions': 0,
            'current_size_bytes': 0
        }

        # Eviction callbacks
        self.eviction_callbacks: List[Callable] = []

        logger.info(f"Cache eviction manager initialized: policy={policy.value}, "
                   f"max_entries={max_entries}, max_size={max_size_bytes // 1024 // 1024}MB")

    def set(
        self,
        key: str,
        value: Any,
        size_bytes: Optional[int] = None,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set a cache entry with automatic eviction if needed

        Returns:
            bool: True if entry was added, False if eviction failed
        """
        # Calculate size if not provided
        if size_bytes is None:
            try:
                import sys
                size_bytes = sys.getsizeof(value)
            except:
                size_bytes = 1024  # Default 1KB estimate

        # Use default TTL if not specified
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        # Check if we need to evict
        if len(self.cache) >= self.max_entries * self.eviction_threshold:
            self._evict_entries(target_count=int(self.max_entries * 0.1))  # Evict 10%

        if self.stats['current_size_bytes'] + size_bytes >= self.max_size_bytes * self.eviction_threshold:
            self._evict_by_size(size_bytes)

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds
        )

        # Remove old entry if exists
        if key in self.cache:
            old_entry = self.cache[key]
            self.stats['current_size_bytes'] -= old_entry.size_bytes

        # Add to cache
        self.cache[key] = entry
        self.stats['current_size_bytes'] += size_bytes

        # Update trackers based on policy
        if self.policy in [EvictionPolicy.LRU, EvictionPolicy.HYBRID]:
            self.lru_tracker[key] = None
            self.lru_tracker.move_to_end(key)

        if self.policy == EvictionPolicy.LFU:
            heapq.heappush(self.lfu_heap, (0, key))  # New entry, access_count=0

        logger.debug(f"Cache SET: {key} ({size_bytes} bytes, TTL={ttl_seconds}s)")
        return True

    def get(self, key: str) -> Optional[Any]:
        """
        Get a cache entry and update access tracking

        Returns:
            The cached value or None if not found/expired
        """
        self.stats['total_accesses'] += 1

        if key not in self.cache:
            self.stats['misses'] += 1
            return None

        entry = self.cache[key]

        # Check TTL expiration
        if entry.is_expired():
            logger.debug(f"Cache entry expired: {key}")
            self._remove_entry(key)
            self.stats['ttl_evictions'] += 1
            self.stats['misses'] += 1
            return None

        # Update access tracking
        entry.access()
        self.stats['hits'] += 1

        # Update policy trackers
        if self.policy in [EvictionPolicy.LRU, EvictionPolicy.HYBRID]:
            self.lru_tracker.move_to_end(key)

        if self.policy == EvictionPolicy.LFU:
            heapq.heappush(self.lfu_heap, (entry.access_count, key))

        return entry.value

    def delete(self, key: str) -> bool:
        """Delete a cache entry manually"""
        if key in self.cache:
            self._remove_entry(key)
            return True
        return False

    def _remove_entry(self, key: str):
        """Remove entry from cache and all trackers"""
        if key not in self.cache:
            return

        entry = self.cache[key]
        self.stats['current_size_bytes'] -= entry.size_bytes

        del self.cache[key]

        if key in self.lru_tracker:
            del self.lru_tracker[key]

        # Notify callbacks
        for callback in self.eviction_callbacks:
            try:
                callback(key, entry)
            except Exception as e:
                logger.error(f"Eviction callback failed: {e}")

    def _evict_entries(self, target_count: int):
        """Evict target_count entries based on policy"""
        evicted = 0

        if self.policy == EvictionPolicy.LRU:
            # Evict least recently used
            while evicted < target_count and len(self.lru_tracker) > 0:
                key, _ = self.lru_tracker.popitem(last=False)  # FIFO order
                if key in self.cache:
                    self._remove_entry(key)
                    evicted += 1

        elif self.policy == EvictionPolicy.LFU:
            # Evict least frequently used
            while evicted < target_count and len(self.lfu_heap) > 0:
                _, key = heapq.heappop(self.lfu_heap)
                if key in self.cache:
                    self._remove_entry(key)
                    evicted += 1

        elif self.policy == EvictionPolicy.TTL:
            # Evict expired entries first
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys[:target_count]:
                self._remove_entry(key)
                evicted += 1

        elif self.policy == EvictionPolicy.HYBRID:
            # First evict expired, then LRU
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                if evicted >= target_count:
                    break
                self._remove_entry(key)
                evicted += 1

            # If not enough, use LRU
            while evicted < target_count and len(self.lru_tracker) > 0:
                key, _ = self.lru_tracker.popitem(last=False)
                if key in self.cache:
                    self._remove_entry(key)
                    evicted += 1

        self.stats['evictions'] += evicted
        self.stats['policy_evictions'] += evicted

        logger.info(f"Evicted {evicted} entries using {self.policy.value} policy")

    def _evict_by_size(self, needed_bytes: int):
        """Evict entries until needed_bytes is available"""
        freed_bytes = 0
        target_bytes = needed_bytes + int(self.max_size_bytes * 0.1)  # Free extra 10%

        # Sort by policy priority
        if self.policy in [EvictionPolicy.LRU, EvictionPolicy.HYBRID]:
            candidates = sorted(
                self.cache.items(),
                key=lambda x: x[1].get_lru_priority(),
                reverse=True
            )
        elif self.policy == EvictionPolicy.LFU:
            candidates = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
        else:  # TTL
            candidates = sorted(
                self.cache.items(),
                key=lambda x: x[1].created_at
            )

        for key, entry in candidates:
            if freed_bytes >= target_bytes:
                break
            freed_bytes += entry.size_bytes
            self._remove_entry(key)

        logger.info(f"Freed {freed_bytes // 1024}KB by evicting {len(candidates)} entries")

    async def cleanup_expired(self):
        """Async task to periodically cleanup expired entries"""
        while True:
            try:
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if entry.is_expired()
                ]

                for key in expired_keys:
                    self._remove_entry(key)

                if expired_keys:
                    self.stats['ttl_evictions'] += len(expired_keys)
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

                # Run cleanup every 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)

    def add_eviction_callback(self, callback: Callable[[str, CacheEntry], None]):
        """Add callback to be notified on evictions"""
        self.eviction_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (
            self.stats['hits'] / self.stats['total_accesses']
            if self.stats['total_accesses'] > 0
            else 0.0
        )

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'entry_count': len(self.cache),
            'size_mb': self.stats['current_size_bytes'] / 1024 / 1024,
            'utilization': len(self.cache) / self.max_entries if self.max_entries > 0 else 0
        }

    def clear(self):
        """Clear all cache entries"""
        count = len(self.cache)
        self.cache.clear()
        self.lru_tracker.clear()
        self.lfu_heap.clear()
        self.stats['current_size_bytes'] = 0
        logger.info(f"Cache cleared: {count} entries removed")
