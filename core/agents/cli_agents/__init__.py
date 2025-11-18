"""
CLI Agent Wrappers for Tri-Agent Collaboration System
"""

from .aider_codex_agent import AiderCodexAgent
from .gemini_cli_agent import GeminiCLIAgent
from .claude_code_agent import ClaudeCodeSelfAgent

__all__ = ['AiderCodexAgent', 'GeminiCLIAgent', 'ClaudeCodeSelfAgent']
