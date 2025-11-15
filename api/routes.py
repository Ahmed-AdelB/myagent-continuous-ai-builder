"""
API Routes - Main routing module
Centralizes all routes defined in main.py for easier maintenance
"""

# This module exists for structure consistency
# All routes are currently defined in api/main.py
# Import main FastAPI app for access to routes

from .main import app

# Routes are defined in main.py:
# - GET /health - Health check
# - GET /api/status - System status  
# - GET /api/projects - Project list
# - POST /api/projects - Create project
# - GET /api/projects/{project_id} - Project details
# - POST /api/projects/{project_id}/start - Start project
# - POST /api/projects/{project_id}/stop - Stop project
# - GET /api/agents - Agent status
# - WebSocket /ws/{project_id} - Real-time updates

__all__ = ['app']
