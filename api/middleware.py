"""
API Middleware - Request/Response processing middleware
Handles CORS, authentication, logging, and error handling
"""

# Middleware is configured in api/main.py
# This module exists for structure consistency

from .main import app

# Middleware configured in main.py:
# - CORS middleware for frontend access
# - Custom error handling middleware
# - Request logging middleware
# - Authentication middleware for protected routes

# Middleware can be accessed through the main FastAPI app instance

__all__ = ['app']
