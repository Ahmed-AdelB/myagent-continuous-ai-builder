"""
CodeEmbedder - Generates embeddings using OpenAI API.

Implements embedding generation with:
- OpenAI text-embedding-3-small (1536 dimensions)
- Batch processing for efficiency
- L2 normalization for cosine similarity
- Local caching to avoid redundant API calls
- Error handling with exponential backoff

Security (Gemini recommendations - FULLY IMPLEMENTED):
- ✅ PII validation before embedding (Priority 1)
- ✅ API key from environment (config/settings.py)
- ✅ File permissions chmod 750 on cache directory (Priority 2)
- ✅ Audit logging for API calls and PII detections (Priority 3)

Based on: docs/architecture/rag_specification.md (v1.1)
Issue: #3
Implementation: Claude (Sonnet 4.5) - Based on Codex patterns
Security review: Gemini (2.5 Pro) - Conditional approval granted
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# PII scanning (Gemini Priority 1 requirement)
from core.utils.pii_scanner import scan_for_pii

# Audit logging (Gemini Priority 3 requirement)
from core.utils.audit_logger import get_embedder_audit_logger


class SecurityError(Exception):
    """Raised when security validation fails (e.g., PII detected)."""
    pass


# OpenAI imports
try:
    from openai import AsyncOpenAI
    _openai_available = True
except ImportError:
    AsyncOpenAI = None
    _openai_available = False
    logger.warning("OpenAI library not available - embeddings will not work")


class CodeEmbedder:
    """
    Generates embeddings using OpenAI text-embedding-3-small API.

    Features:
    - Batch processing (up to 100 texts)
    - L2 normalization for cosine similarity
    - Local file-based cache
    - Exponential backoff for rate limits
    - Error handling and fallbacks

    Security (Gemini requirements):
    - ✅ PII validation before embedding (Priority 1)
    - ✅ API key from environment (config/settings.py)
    - ✅ SecurityError raised if PII detected
    - TODO: Audit logging (Priority 3)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        batch_size: int = 100,
    ):
        """
        Initialize embedder.

        Args:
            model: OpenAI embedding model name
            cache_enabled: Enable local caching
            cache_dir: Cache directory (default: persistence/cache/embeddings/)
            max_retries: Maximum API retry attempts
            batch_size: Maximum texts per batch (OpenAI limit: 100)
        """
        self.model = model
        self.cache_enabled = cache_enabled
        self.max_retries = max_retries
        self.batch_size = min(batch_size, 100)  # OpenAI hard limit

        # Cache setup
        if cache_dir is None:
            cache_dir = Path("persistence/cache/embeddings")
        self.cache_dir = cache_dir
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Security (Gemini Priority 2): Set file permissions chmod 750
            # Restricts access to owner (rwx) and group (r-x), no access for others
            try:
                os.chmod(self.cache_dir, 0o750)
                logger.info(f"Set permissions 750 on {self.cache_dir}")
            except Exception as exc:
                logger.warning(f"Failed to set permissions on {self.cache_dir}: {exc}")

        # Initialize OpenAI client
        self.client = self._init_client()

        # Audit logger (Gemini Priority 3)
        self.audit_logger = get_embedder_audit_logger()

        # Stats
        self.stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "errors": 0,
        }

    def _init_client(self) -> Optional[AsyncOpenAI]:
        """Initialize OpenAI async client with API key from environment."""
        if not _openai_available:
            logger.error("OpenAI library not installed")
            return None

        try:
            # Import settings to get API key from environment
            from config.settings import settings

            if not settings.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY not set in environment")
                return None

            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(f"Initialized OpenAI client for model: {self.model}")
            return client

        except Exception as exc:
            logger.error(f"Failed to initialize OpenAI client: {exc}")
            return None

    async def embed(self, text: str, chunk_id: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Text to embed
            chunk_id: Optional chunk identifier for audit logging

        Returns:
            Normalized embedding vector (1536 dimensions)

        Raises:
            SecurityError: If PII is detected in text
        """
        embeddings = await self.embed_batch([text], chunk_ids=[chunk_id] if chunk_id else None)
        return embeddings[0]

    async def embed_batch(
        self,
        texts: List[str],
        chunk_ids: Optional[List[Optional[str]]] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed (max 100)
            chunk_ids: Optional list of chunk identifiers for audit logging

        Returns:
            List of normalized embedding vectors

        Raises:
            SecurityError: If PII is detected in any text
        """
        if not texts:
            return []

        # Ensure chunk_ids matches texts length
        if chunk_ids is None:
            chunk_ids = [None] * len(texts)
        elif len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids must have same length as texts")

        # PII VALIDATION (Gemini Priority 1 - MUST happen before embedding)
        for i, text in enumerate(texts):
            pii_result = scan_for_pii(text, chunk_id=chunk_ids[i])
            if pii_result.detected:
                # Audit log PII detection (Gemini Priority 3)
                finding_types = [f['type'] for f in pii_result.findings]
                self.audit_logger.log_pii_detection(
                    chunk_id=chunk_ids[i],
                    num_findings=len(pii_result.findings),
                    finding_types=finding_types,
                )

                error_msg = (
                    f"PII detected in chunk {chunk_ids[i] or i}: "
                    f"{len(pii_result.findings)} findings - "
                    f"types: {finding_types}"
                )
                logger.error(error_msg)
                self.stats["errors"] += 1
                raise SecurityError(error_msg)

        if len(texts) > self.batch_size:
            logger.warning(
                f"Batch size {len(texts)} exceeds limit {self.batch_size}, "
                f"splitting into multiple batches"
            )
            # Split into multiple batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_ids = chunk_ids[i:i + self.batch_size]
                batch_embeddings = await self.embed_batch(batch, chunk_ids=batch_ids)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

        # Check cache first
        if self.cache_enabled:
            cached_embeddings = self._check_cache(texts, chunk_ids)
            if cached_embeddings:
                return cached_embeddings

        # Generate embeddings via API
        embeddings = await self._generate_embeddings(texts, chunk_ids)

        # Cache results
        if self.cache_enabled and embeddings:
            self._save_to_cache(texts, embeddings)

        return embeddings

    async def _generate_embeddings(
        self,
        texts: List[str],
        chunk_ids: List[Optional[str]],
    ) -> List[np.ndarray]:
        """Generate embeddings via OpenAI API with retry logic."""
        if not self.client:
            logger.error("OpenAI client not initialized")
            self.stats["errors"] += 1

            # Audit log failure
            self.audit_logger.log_embedding_call(
                chunk_id=chunk_ids[0] if chunk_ids else None,
                status="failure",
                num_texts=len(texts),
                error="OpenAI client not initialized",
            )

            # Return zero vectors as fallback
            return [np.zeros(1536) for _ in texts]

        for attempt in range(self.max_retries):
            try:
                # Track API call duration for audit logging
                start_time = time.time()

                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )

                duration_ms = (time.time() - start_time) * 1000

                # Extract embeddings
                raw_embeddings = [item.embedding for item in response.data]

                # L2 normalization for cosine similarity
                embeddings = [self._normalize(emb) for emb in raw_embeddings]

                # Update stats
                self.stats["api_calls"] += 1
                self.stats["cache_misses"] += len(texts)
                self.stats["total_tokens"] += response.usage.total_tokens

                # Audit log successful API call (Gemini Priority 3)
                self.audit_logger.log_embedding_call(
                    chunk_id=chunk_ids[0] if chunk_ids else None,
                    status="success",
                    duration_ms=duration_ms,
                    num_texts=len(texts),
                )

                logger.debug(
                    f"Generated {len(embeddings)} embeddings "
                    f"({response.usage.total_tokens} tokens) in {duration_ms:.1f}ms"
                )

                return embeddings

            except Exception as exc:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Embedding API error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{exc}. Retrying in {wait_time}s..."
                )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
                    self.stats["errors"] += 1

                    # Audit log failure after all retries exhausted
                    self.audit_logger.log_embedding_call(
                        chunk_id=chunk_ids[0] if chunk_ids else None,
                        status="failure",
                        num_texts=len(texts),
                        error=str(exc),
                    )

                    # Return zero vectors as fallback
                    return [np.zeros(1536) for _ in texts]

    def _normalize(self, embedding: List[float]) -> np.ndarray:
        """Apply L2 normalization for cosine similarity."""
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _check_cache(
        self,
        texts: List[str],
        chunk_ids: List[Optional[str]],
    ) -> Optional[List[np.ndarray]]:
        """Check if all texts are in cache."""
        embeddings = []

        for text in texts:
            cache_key = self._cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.npy"

            if not cache_file.exists():
                return None  # Cache miss - need to fetch all

            try:
                embedding = np.load(cache_file)
                embeddings.append(embedding)
                self.stats["cache_hits"] += 1
            except Exception as exc:
                logger.warning(f"Failed to load cache file {cache_file}: {exc}")
                return None

        # Audit log cache hit (Gemini Priority 3)
        self.audit_logger.log_cache_hit(
            chunk_id=chunk_ids[0] if chunk_ids else None,
            num_texts=len(texts),
        )

        logger.debug(f"Cache hit for {len(texts)} texts")
        return embeddings

    def _save_to_cache(self, texts: List[str], embeddings: List[np.ndarray]):
        """Save embeddings to cache."""
        for text, embedding in zip(texts, embeddings):
            cache_key = self._cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.npy"

            try:
                np.save(cache_file, embedding)
            except Exception as exc:
                logger.warning(f"Failed to save to cache: {exc}")

    def _cache_key(self, text: str) -> str:
        """Generate cache key from text (SHA256 hash)."""
        # Include model name in hash to avoid collisions
        content = f"{self.model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests
            if total_requests > 0
            else 0
        )

        return {
            **self.stats,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "model": self.model,
            "cache_enabled": self.cache_enabled,
        }

    async def clear_cache(self):
        """Clear embedding cache."""
        if not self.cache_enabled:
            return

        try:
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()
            logger.info("Embedding cache cleared")
        except Exception as exc:
            logger.error(f"Failed to clear cache: {exc}")
