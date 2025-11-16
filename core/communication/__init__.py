"""
Communication Module

GPT-5 Recommended Agent Communication System
Implements centralized message bus for structured agent coordination
"""

from .agent_message_bus import (
    AgentMessageBus,
    AgentMessage,
    MessageSubscription,
    MessageType,
    MessagePriority
)

__all__ = [
    'AgentMessageBus',
    'AgentMessage',
    'MessageSubscription',
    'MessageType',
    'MessagePriority'
]