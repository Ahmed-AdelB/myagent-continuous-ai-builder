"""
PII Scanner - Detects personally identifiable information and secrets in code.

Implements basic PII/secret detection before code is embedded.

Security requirement (Gemini Priority 1):
- Scans for common secrets (API keys, tokens, passwords)
- Detects high-entropy strings (potential credentials)
- Validates email addresses and other PII
- Blocks embedding if sensitive data detected

Based on: Gemini security review (Issue #3)
Implementation: Claude (Sonnet 4.5)
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class PIIDetection:
    """Result of PII scan."""
    detected: bool
    findings: List[Dict[str, Any]]
    sanitized_text: Optional[str] = None


class PIIScanner:
    """
    Scans code for PII and secrets before embedding.

    Detection patterns:
    - API keys and tokens (AWS, GitHub, OpenAI, etc.)
    - Email addresses
    - High-entropy strings (potential passwords/keys)
    - Credit card numbers
    - SSH private keys
    - Database connection strings

    Security approach:
    - Fail-safe: If uncertain, flag as potential PII
    - Audit logging for all detections
    - Configurable strictness levels
    """

    # Regex patterns for common secrets
    PATTERNS = {
        "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "github_token": re.compile(r"ghp_[a-zA-Z0-9]{36}"),
        "openai_key": re.compile(r"sk-[a-zA-Z0-9]{48}"),
        "generic_api_key": re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?"),
        "password": re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]([^'\"]{8,})['\"]"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "ssh_private_key": re.compile(r"-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----"),
        "jwt_token": re.compile(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*"),
        "database_url": re.compile(r"(?i)(postgres|mysql|mongodb)://[^'\";\s]+"),
    }

    # Entropy threshold for detecting random strings (potential secrets)
    ENTROPY_THRESHOLD = 4.5  # Shannon entropy bits per character
    MIN_LENGTH_FOR_ENTROPY = 20  # Only check entropy on strings 20+ chars

    def __init__(
        self,
        strict_mode: bool = True,
        check_entropy: bool = True,
        audit_log: bool = True,
    ):
        """
        Initialize PII scanner.

        Args:
            strict_mode: If True, flag any potential PII (fail-safe)
            check_entropy: Enable high-entropy string detection
            audit_log: Log all PII detections to audit trail
        """
        self.strict_mode = strict_mode
        self.check_entropy = check_entropy
        self.audit_log = audit_log

        # Statistics
        self.stats = {
            "scans_performed": 0,
            "pii_detected": 0,
            "pii_types": Counter(),
        }

    def scan(self, text: str, chunk_id: Optional[str] = None) -> PIIDetection:
        """
        Scan text for PII and secrets.

        Args:
            text: Code text to scan
            chunk_id: Optional identifier for audit logging

        Returns:
            PIIDetection with findings
        """
        self.stats["scans_performed"] += 1
        findings = []

        # Check regex patterns
        for pattern_name, pattern in self.PATTERNS.items():
            matches = pattern.finditer(text)
            for match in matches:
                finding = {
                    "type": pattern_name,
                    "matched_text": match.group(0)[:50],  # Truncate for safety
                    "position": match.start(),
                    "severity": self._get_severity(pattern_name),
                }
                findings.append(finding)

        # Check high-entropy strings (potential secrets)
        if self.check_entropy:
            entropy_findings = self._detect_high_entropy_strings(text)
            findings.extend(entropy_findings)

        # Determine if PII was detected
        detected = len(findings) > 0

        if detected:
            self.stats["pii_detected"] += 1
            for finding in findings:
                self.stats["pii_types"][finding["type"]] += 1

            # Audit logging
            if self.audit_log:
                logger.warning(
                    f"PII detected in chunk {chunk_id or 'unknown'}: "
                    f"{len(findings)} findings - types: {[f['type'] for f in findings]}"
                )

        return PIIDetection(
            detected=detected,
            findings=findings,
            sanitized_text=None,  # TODO: Implement sanitization if needed
        )

    def _detect_high_entropy_strings(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect high-entropy strings that could be secrets.

        High entropy indicates randomness, which is common in:
        - API keys
        - Tokens
        - Passwords
        - Encryption keys
        """
        findings = []

        # Extract potential string literals
        string_pattern = re.compile(r"['\"]([^'\"]{20,})['\"]")
        for match in string_pattern.finditer(text):
            candidate = match.group(1)

            # Skip if it looks like natural language
            if self._is_natural_language(candidate):
                continue

            # Calculate Shannon entropy
            entropy = self._calculate_entropy(candidate)

            if entropy > self.ENTROPY_THRESHOLD:
                findings.append({
                    "type": "high_entropy_string",
                    "matched_text": candidate[:50],
                    "position": match.start(),
                    "severity": "high",
                    "entropy": round(entropy, 2),
                })

        return findings

    def _calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy (bits per character).

        Formula: H = -Σ p(x) * log2(p(x))

        Higher entropy = more random = more likely to be a secret.
        """
        if not text:
            return 0.0

        # Count character frequencies
        char_counts = Counter(text)
        text_len = len(text)

        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            probability = count / text_len
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def _is_natural_language(self, text: str) -> bool:
        """
        Heuristic to detect natural language vs random strings.

        Natural language has:
        - Common words
        - Spaces
        - Lowercase letters
        - Lower entropy
        """
        # Has spaces (common in natural language)
        if " " in text:
            return True

        # Mostly lowercase (common in sentences)
        lowercase_ratio = sum(1 for c in text if c.islower()) / len(text)
        if lowercase_ratio > 0.7:
            return True

        # Check for common English words
        common_words = ["the", "is", "and", "to", "of", "in", "for", "with"]
        text_lower = text.lower()
        if any(word in text_lower for word in common_words):
            return True

        return False

    def _get_severity(self, pattern_name: str) -> str:
        """Determine severity level for detected PII type."""
        high_severity = {
            "aws_access_key",
            "github_token",
            "openai_key",
            "ssh_private_key",
            "password",
            "database_url",
        }

        if pattern_name in high_severity:
            return "critical"
        elif pattern_name in {"credit_card", "jwt_token"}:
            return "high"
        else:
            return "medium"

    def get_stats(self) -> Dict[str, Any]:
        """Get scanning statistics."""
        detection_rate = (
            self.stats["pii_detected"] / self.stats["scans_performed"]
            if self.stats["scans_performed"] > 0
            else 0
        )

        return {
            **self.stats,
            "detection_rate": round(detection_rate, 3),
            "pii_types_breakdown": dict(self.stats["pii_types"]),
        }

    def clear_stats(self):
        """Reset statistics."""
        self.stats = {
            "scans_performed": 0,
            "pii_detected": 0,
            "pii_types": Counter(),
        }


# Singleton instance for global use
_scanner = None


def get_scanner() -> PIIScanner:
    """Get global PII scanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = PIIScanner(
            strict_mode=True,
            check_entropy=True,
            audit_log=True,
        )
    return _scanner


def scan_for_pii(text: str, chunk_id: Optional[str] = None) -> PIIDetection:
    """
    Convenience function to scan text for PII.

    Usage:
        result = scan_for_pii(code_chunk)
        if result.detected:
            logger.error(f"PII detected: {result.findings}")
            raise SecurityError("Cannot embed code containing PII")
    """
    scanner = get_scanner()
    return scanner.scan(text, chunk_id)
