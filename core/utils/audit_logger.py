"""
Audit Logger - Security audit trail for RAG operations.

Implements audit logging for:
- CodeEmbedder: API calls, PII detections, errors
- VectorStore: Add/delete operations, query counts
- Append-only logging for compliance (SOC 2, ISO 27001)

Security requirement (Gemini Priority 3):
- Immutable audit trail
- Timestamps for all operations
- Operation metadata (chunk_ids, status, duration)
- Separate audit log file from application logs

Based on: Gemini security review (Issue #3)
Implementation: Claude (Sonnet 4.5)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum


class AuditEventType(Enum):
    """Types of auditable events."""
    EMBEDDING_API_CALL = "embedding_api_call"
    EMBEDDING_CACHE_HIT = "embedding_cache_hit"
    PII_DETECTED = "pii_detected"
    VECTOR_ADD = "vector_add"
    VECTOR_DELETE = "vector_delete"
    VECTOR_QUERY = "vector_query"
    SECURITY_ERROR = "security_error"


class AuditLogger:
    """
    Audit logger for security-sensitive RAG operations.

    Features:
    - Append-only JSON log file
    - Structured event logging
    - Automatic timestamp addition
    - Safe file handling (creates parent dirs)

    Log format:
    {
        "timestamp": "2025-11-20T12:00:00.000Z",
        "event_type": "embedding_api_call",
        "component": "CodeEmbedder",
        "status": "success",
        "metadata": {...}
    }
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        component_name: str = "RAG",
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Path to audit log file (default: persistence/audit/rag_audit.log)
            component_name: Name of component for log entries
        """
        if log_file is None:
            log_file = Path("persistence/audit/rag_audit.log")

        self.log_file = log_file
        self.component_name = component_name

        # Create directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Set file permissions to 640 (owner read/write, group read)
        if not self.log_file.exists():
            self.log_file.touch()
            import os
            try:
                os.chmod(self.log_file, 0o640)
            except Exception:
                pass  # Best effort

        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_type": {},
        }

    def log_event(
        self,
        event_type: AuditEventType,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an audit event.

        Args:
            event_type: Type of event
            status: Status (success, failure, warning)
            metadata: Additional event metadata
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type.value,
            "component": self.component_name,
            "status": status,
            "metadata": metadata or {},
        }

        # Write to file (append-only)
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            # Fallback to standard logging if audit log fails
            logging.error(f"Failed to write audit log: {exc}")

        # Update stats
        self.stats["total_events"] += 1
        event_type_str = event_type.value
        self.stats["events_by_type"][event_type_str] = \
            self.stats["events_by_type"].get(event_type_str, 0) + 1

    def log_embedding_call(
        self,
        chunk_id: Optional[str],
        status: str,
        duration_ms: Optional[float] = None,
        num_texts: int = 1,
        error: Optional[str] = None,
    ):
        """
        Log an embedding API call.

        Args:
            chunk_id: Chunk identifier
            status: success or failure
            duration_ms: API call duration in milliseconds
            num_texts: Number of texts embedded
            error: Error message if failed
        """
        metadata = {
            "chunk_id": chunk_id,
            "num_texts": num_texts,
        }

        if duration_ms is not None:
            metadata["duration_ms"] = round(duration_ms, 2)

        if error:
            metadata["error"] = error

        self.log_event(
            event_type=AuditEventType.EMBEDDING_API_CALL,
            status=status,
            metadata=metadata,
        )

    def log_cache_hit(self, chunk_id: Optional[str], num_texts: int = 1):
        """Log an embedding cache hit."""
        self.log_event(
            event_type=AuditEventType.EMBEDDING_CACHE_HIT,
            status="success",
            metadata={"chunk_id": chunk_id, "num_texts": num_texts},
        )

    def log_pii_detection(
        self,
        chunk_id: Optional[str],
        num_findings: int,
        finding_types: list,
    ):
        """
        Log PII detection event.

        Args:
            chunk_id: Chunk identifier
            num_findings: Number of PII findings
            finding_types: List of PII types detected
        """
        self.log_event(
            event_type=AuditEventType.PII_DETECTED,
            status="blocked",
            metadata={
                "chunk_id": chunk_id,
                "num_findings": num_findings,
                "finding_types": finding_types,
            },
        )

    def log_vector_operation(
        self,
        operation: str,
        num_chunks: int,
        status: str,
        collection_name: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """
        Log vector store operation.

        Args:
            operation: add, delete, or query
            num_chunks: Number of chunks affected
            status: success or failure
            collection_name: Collection name
            error: Error message if failed
        """
        event_type_map = {
            "add": AuditEventType.VECTOR_ADD,
            "delete": AuditEventType.VECTOR_DELETE,
            "query": AuditEventType.VECTOR_QUERY,
        }

        metadata = {
            "operation": operation,
            "num_chunks": num_chunks,
        }

        if collection_name:
            metadata["collection"] = collection_name

        if error:
            metadata["error"] = error

        self.log_event(
            event_type=event_type_map.get(operation, AuditEventType.VECTOR_ADD),
            status=status,
            metadata=metadata,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        return {
            **self.stats,
            "log_file": str(self.log_file),
            "component": self.component_name,
        }


# Global audit logger instances (lazy initialization)
_embedder_audit_logger = None
_vector_store_audit_logger = None


def get_embedder_audit_logger() -> AuditLogger:
    """Get audit logger for CodeEmbedder."""
    global _embedder_audit_logger
    if _embedder_audit_logger is None:
        _embedder_audit_logger = AuditLogger(
            component_name="CodeEmbedder",
        )
    return _embedder_audit_logger


def get_vector_store_audit_logger() -> AuditLogger:
    """Get audit logger for VectorStore."""
    global _vector_store_audit_logger
    if _vector_store_audit_logger is None:
        _vector_store_audit_logger = AuditLogger(
            component_name="VectorStore",
        )
    return _vector_store_audit_logger
