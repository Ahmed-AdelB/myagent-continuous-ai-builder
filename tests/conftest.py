"""
Global pytest configuration and fixtures for MyAgent testing suite
"""

import asyncio
import os
import tempfile
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from datetime import datetime
import json
import sqlite3
from pathlib import Path

# Core imports
from core.orchestrator.continuous_director import ContinuousDirector
from core.agents.base_agent import PersistentAgent
from core.agents.coder_agent import CoderAgent
from core.agents.tester_agent import TesterAgent
from core.agents.debugger_agent import DebuggerAgent
from core.agents.architect_agent import ArchitectAgent
from core.agents.analyzer_agent import AnalyzerAgent
from core.agents.ui_refiner_agent import UIRefinerAgent
from core.memory.project_ledger import ProjectLedger
from core.memory.vector_memory import VectorMemory
from core.memory.error_knowledge_graph import ErrorKnowledgeGraph

# GPT-5 Priority imports
from core.memory_pyramid.hierarchical_memory_pyramid import HierarchicalMemoryPyramid
from core.security.security_compliance_scanner import SecurityComplianceScanner
from core.self_healing.self_healing_orchestrator import SelfHealingOrchestrator
from core.knowledge.knowledge_graph_manager import KnowledgeGraphManager
from core.deployment.deployment_orchestrator import DeploymentOrchestrator
from core.causal_analytics.causal_graph_analytics import CausalGraphAnalytics

# API imports
from api.main import app as fastapi_app
from fastapi.testclient import TestClient

# Test utilities
import httpx
import pandas as pd
import numpy as np


# ===================== SCOPE CONFIGURATION =====================




# ===================== TEMPORARY RESOURCES =====================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_database():
    """Provide a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


# ===================== MOCK CONFIGURATIONS =====================

@pytest.fixture
def mock_llm():
    """Mock LLM client for deterministic testing."""
    mock = AsyncMock()

    # Default responses for different agent types
    responses = {
        "coder": "def test_function():\n    return 'Hello, World!'",
        "tester": "import pytest\n\ndef test_example():\n    assert True",
        "debugger": "Fixed issue by updating line 42",
        "architect": "Recommended using observer pattern",
        "analyzer": "Performance metrics within acceptable range",
        "ui_refiner": "Improved accessibility with ARIA labels"
    }

    def side_effect(*args, **kwargs):
        # Extract agent type from context
        if hasattr(kwargs.get('input', {}), 'get'):
            agent_type = kwargs['input'].get('agent_type', 'coder')
        else:
            agent_type = 'coder'

        response_mock = Mock()
        response_mock.content = responses.get(agent_type, responses['coder'])
        return response_mock

    mock.ainvoke.side_effect = side_effect
    mock.invoke.side_effect = side_effect

    return mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for API testing."""
    with patch('openai.AsyncOpenAI') as mock:
        client = AsyncMock()

        # Mock chat completions
        completion_response = Mock()
        completion_response.choices = [Mock()]
        completion_response.choices[0].message.content = "Mocked AI response"

        client.chat.completions.create.return_value = completion_response
        mock.return_value = client

        yield client


# ===================== AGENT FIXTURES =====================

@pytest.fixture
def base_agent(temp_dir, mock_llm):
    """Create a base agent for testing."""
    agent = PersistentAgent(
        name="test_agent",
        role="Test Agent",
        capabilities=["test_capability"]
    )
    agent.checkpoint_dir = temp_dir / "checkpoints"
    agent.checkpoint_dir.mkdir(exist_ok=True)
    agent.llm = mock_llm
    return agent


@pytest.fixture
def coder_agent(temp_dir, mock_llm):
    """Create a coder agent for testing."""
    agent = CoderAgent()
    agent.checkpoint_dir = temp_dir / "checkpoints"
    agent.checkpoint_dir.mkdir(exist_ok=True)
    agent.llm = mock_llm
    return agent


@pytest.fixture
def tester_agent(temp_dir, mock_llm):
    """Create a tester agent for testing."""
    agent = TesterAgent()
    agent.checkpoint_dir = temp_dir / "checkpoints"
    agent.checkpoint_dir.mkdir(exist_ok=True)
    agent.llm = mock_llm
    return agent


@pytest.fixture
def debugger_agent(temp_dir, mock_llm):
    """Create a debugger agent for testing."""
    agent = DebuggerAgent()
    agent.checkpoint_dir = temp_dir / "checkpoints"
    agent.checkpoint_dir.mkdir(exist_ok=True)
    agent.llm = mock_llm
    return agent


@pytest.fixture
def all_agents(temp_dir, mock_llm):
    """Create all specialized agents for testing."""
    agents = {
        'coder': CoderAgent(),
        'tester': TesterAgent(),
        'debugger': DebuggerAgent(),
        'architect': ArchitectAgent(),
        'analyzer': AnalyzerAgent(),
        'ui_refiner': UIRefinerAgent()
    }

    for agent in agents.values():
        agent.checkpoint_dir = temp_dir / "checkpoints"
        agent.checkpoint_dir.mkdir(exist_ok=True)
        agent.llm = mock_llm

    return agents


# ===================== MEMORY SYSTEM FIXTURES =====================

@pytest.fixture
async def project_ledger(temp_database):
    """Create a project ledger for testing."""
    ledger = ProjectLedger(database_path=temp_database)
    await ledger.initialize()
    yield ledger
    await ledger.close()


@pytest.fixture
async def vector_memory(temp_dir):
    """Create a vector memory system for testing."""
    memory = VectorMemory(
        collection_name="test_collection",
        persist_directory=str(temp_dir / "vector_memory")
    )
    await memory.initialize()
    yield memory
    await memory.cleanup()


@pytest.fixture
async def error_knowledge_graph(temp_database):
    """Create an error knowledge graph for testing."""
    graph = ErrorKnowledgeGraph(database_path=temp_database)
    await graph.initialize()
    yield graph
    await graph.cleanup()


# ===================== GPT-5 PRIORITY FIXTURES =====================

@pytest_asyncio.fixture(scope="function")
async def memory_pyramid_instance(temp_database):
    """Create hierarchical memory pyramid for testing."""
    pyramid = HierarchicalMemoryPyramid(database_path=temp_database)
    await pyramid.initialize()
    yield pyramid
    await pyramid.cleanup()


@pytest.fixture
async def security_scanner():
    """Create security compliance scanner for testing."""
    scanner = SecurityComplianceScanner()
    await scanner.initialize()
    yield scanner
    await scanner.cleanup()


@pytest.fixture
async def self_healing_orchestrator(temp_database):
    """Create self-healing orchestrator for testing."""
    orchestrator = SelfHealingOrchestrator(database_path=temp_database)
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.cleanup()


@pytest.fixture
async def knowledge_graph_manager(temp_database):
    """Create knowledge graph manager for testing."""
    manager = KnowledgeGraphManager(database_path=temp_database)
    await manager.initialize()
    yield manager
    await manager.cleanup()


@pytest.fixture
async def deployment_orchestrator(temp_database):
    """Create deployment orchestrator for testing."""
    orchestrator = DeploymentOrchestrator(database_path=temp_database)
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.cleanup()


@pytest.fixture
async def causal_analytics(temp_database):
    """Create causal graph analytics for testing."""
    analytics = CausalGraphAnalytics(db_path=temp_database)
    yield analytics


# ===================== ORCHESTRATOR FIXTURES =====================

@pytest.fixture
async def continuous_director(temp_dir, all_agents, project_ledger, vector_memory, error_knowledge_graph):
    """Create a continuous director for testing."""
    project_config = {
        'name': 'Test Project',
        'description': 'Test project for unit testing',
        'requirements': ['test requirement'],
        'target_metrics': {
            'test_coverage': 95.0,
            'code_quality': 85.0
        }
    }

    director = ContinuousDirector(
        project_id="test_project",
        project_config=project_config
    )

    # Mock the initialization to avoid external dependencies
    director.agents = all_agents
    director.project_ledger = project_ledger
    director.vector_memory = vector_memory
    director.error_knowledge_graph = error_knowledge_graph
    director.workspace_dir = temp_dir / "workspace"
    director.workspace_dir.mkdir(exist_ok=True)

    yield director


# ===================== API FIXTURES =====================

@pytest.fixture
def fastapi_client():
    """Create FastAPI test client."""
    return TestClient(fastapi_app)


@pytest.fixture
async def async_client():
    """Create async HTTP client for testing."""
    async with httpx.AsyncClient(app=fastapi_app, base_url="http://test") as client:
        yield client


# ===================== TEST DATA FIXTURES =====================

@pytest.fixture
def sample_project_data():
    """Sample project data for testing."""
    return {
        'name': 'Sample Project',
        'description': 'A sample project for testing purposes',
        'requirements': [
            'Create a web application',
            'Include user authentication',
            'Add data visualization'
        ],
        'target_metrics': {
            'test_coverage': 95.0,
            'code_quality': 85.0,
            'performance_score': 90.0,
            'documentation_coverage': 90.0,
            'security_score': 95.0,
            'user_satisfaction': 90.0
        },
        'max_iterations': 100
    }


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        'type': 'generate_function',
        'description': 'Create a function to calculate fibonacci numbers',
        'priority': 5,
        'agent': 'coder',
        'data': {
            'language': 'python',
            'function_name': 'fibonacci',
            'parameters': ['n']
        }
    }


@pytest.fixture
def sample_code_data():
    """Sample code data for testing."""
    return {
        'python_function': '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
''',
        'test_code': '''
import pytest

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55
''',
        'buggy_code': '''
def divide(a, b):
    return a / b  # No zero division check
'''
    }


@pytest.fixture
def sample_causal_data():
    """Sample data for causal analytics testing."""
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic causal data
    treatment = np.random.binomial(1, 0.5, n_samples)
    confounder = np.random.normal(0, 1, n_samples)
    noise = np.random.normal(0, 0.5, n_samples)

    # Outcome affected by treatment and confounder
    outcome = 2.0 * treatment + 1.5 * confounder + noise

    return pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome,
        'confounder': confounder,
        'mediator': treatment * 0.8 + np.random.normal(0, 0.3, n_samples)
    })


# ===================== PERFORMANCE FIXTURES =====================

@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        'agent_response_time': 5.0,  # seconds
        'api_response_time': 1.0,    # seconds
        'memory_usage_mb': 500,      # megabytes
        'database_query_time': 0.1,  # seconds
        'websocket_latency': 0.05    # seconds
    }


# ===================== UTILITY FIXTURES =====================

@pytest.fixture
def mock_file_system(temp_dir):
    """Mock file system operations."""
    def create_file(path: str, content: str = ""):
        file_path = temp_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def create_directory(path: str):
        dir_path = temp_dir / path
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    return {
        'create_file': create_file,
        'create_directory': create_directory,
        'temp_dir': temp_dir
    }


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    env_vars = {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'DATABASE_URL': 'sqlite:///test.db',
        'REDIS_URL': 'redis://localhost:6379/1',
        'LOG_LEVEL': 'DEBUG',
        'ENVIRONMENT': 'test'
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


# ===================== CLEANUP HOOKS =====================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup is handled by temp_dir fixture


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to tests that take > 5 seconds
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add unit marker to unit tests
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)


# ===================== ASYNC TEST HELPERS =====================

@pytest.fixture
def anyio_backend():
    """Use asyncio backend for async tests."""
    return "asyncio"


# ===================== DATABASE FIXTURES =====================

@pytest.fixture
async def clean_database(temp_database):
    """Provide a clean database for each test."""
    # Setup
    conn = sqlite3.connect(temp_database)
    yield conn

    # Teardown
    conn.close()


# ===================== WEBSOCKET FIXTURES =====================

@pytest.fixture
async def websocket_client():
    """WebSocket test client for real-time communication testing."""
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        yield client


# ===================== MONITORING FIXTURES =====================

@pytest.fixture
def telemetry_collector():
    """Collect telemetry data during tests."""
    telemetry_data = []

    def collect(event_type: str, data: Dict[str, Any]):
        telemetry_data.append({
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': data
        })

    yield collect, telemetry_data