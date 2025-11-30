
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json
import uuid
import os
from datetime import datetime

# Set dummy API key to avoid validation errors during import/init
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"

# Mock aiosqlite BEFORE importing modules that use it
import sys
mock_aiosqlite = MagicMock()
mock_conn = AsyncMock()
mock_conn.__aenter__.return_value = mock_conn
mock_conn.__aexit__.return_value = None
mock_aiosqlite.connect.return_value = mock_conn
sys.modules["aiosqlite"] = mock_aiosqlite

from api.main import app
from core.orchestrator.continuous_director import ProjectState

client = TestClient(app)

@pytest.fixture
def mock_db():
    """Mock database manager to avoid Postgres connection"""
    with patch("api.main.db_manager") as mock_db_manager:
        # Mock execute to return success
        mock_db_manager.execute = AsyncMock(return_value=None)
        # Mock fetch/fetchrow/fetchval
        mock_db_manager.fetch = AsyncMock(return_value=[])
        mock_db_manager.fetchrow = AsyncMock(return_value=None)
        mock_db_manager.fetchval = AsyncMock(return_value=None)
        yield mock_db_manager

@pytest.fixture
def mock_init_db():
    """Mock database initialization"""
    with patch("api.main.init_database", new_callable=AsyncMock) as mock_init:
        yield mock_init

@pytest.fixture
def mock_llm():
    """Mock LLM to avoid API calls"""
    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_instance = MagicMock()
        # Mock response for "create idea" or planning
        mock_instance.invoke.return_value.content = json.dumps({
            "plan": ["Create calculator.py", "Add add function", "Add subtract function"],
            "files": [{"path": "calculator.py", "content": "def add(a,b): return a+b"}]
        })
        # Also mock apredict for async calls
        mock_instance.apredict = AsyncMock(return_value=json.dumps({
            "read": [],
            "modify": [],
            "create": ["calculator.py"]
        }))
        mock_chat.return_value = mock_instance
        yield mock_chat

@pytest.mark.asyncio
async def test_end_to_end_project_creation(mock_db, mock_init_db, mock_llm):
    """
    Simulate full user flow:
    1. Create Project (Calculator)
    2. Verify Orchestrator starts
    3. Check Agents are assigned
    4. Check Metrics are initialized
    """
    
    # User requirements (from env vars or default)
    project_name = os.getenv("TEST_PROJECT_NAME", "Calculator App")
    project_desc = os.getenv("TEST_PROJECT_DESC", "A simple calculator with basic arithmetic operations")
    
    initial_requirements = [
        f"Build a {project_name}",
        project_desc,
        "Include unit tests",
        "Create documentation"
    ]
    
    project_payload = {
        "name": project_name,
        "description": project_desc,
        "requirements": initial_requirements,
        "target_metrics": {"test_coverage": 90.0},
        "max_iterations": 10
    }
    
    response = client.post("/projects", json=project_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["name"] == project_name
    assert "id" in data
    project_id = data["id"]
    assert data["state"] == "initializing"
    
    # 2. Verify Project Status
    response = client.get(f"/projects/{project_id}")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["id"] == project_id
    assert status_data["name"] == project_name
    
    # 3. Verify Agents are Initialized
    # The orchestrator runs in background, so agents might take a moment.
    import time
    agents = []
    for _ in range(20):  # Wait up to 10 seconds
        response = client.get(f"/projects/{project_id}/agents")
        if response.status_code == 200:
            agents = response.json()
            if agents:
                break
        time.sleep(0.5)
    
    assert len(agents) > 0, "Agents were not initialized in time"
    
    # Should have core agents: Coder, Tester, Debugger, Architect
    agent_roles = [a["role"] for a in agents]
    assert "System Architect" in agent_roles
    assert "Code Generator" in agent_roles
    assert "Test Engineer" in agent_roles
    
    # 4. Verify Metrics
    response = client.get(f"/projects/{project_id}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "quality_metrics" in metrics
    
    print(f"\n✅ Successfully simulated project creation: {project_id}")
    print(f"✅ Agents initialized: {agent_roles}")
    print(f"✅ Metrics initialized: {metrics['quality_metrics']}")

