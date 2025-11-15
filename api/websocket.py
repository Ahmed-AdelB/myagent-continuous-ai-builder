"""
WebSocket Handler - Real-time communication module
Manages WebSocket connections for live project updates
"""

# WebSocket functionality is implemented in api/main.py
# This module exists for structure consistency

from .main import websocket_manager, handle_websocket

# WebSocket features available:
# - Real-time project status updates
# - Agent execution progress
# - Error notifications
# - Quality metrics updates
# - Live logs streaming

__all__ = ['websocket_manager', 'handle_websocket']
