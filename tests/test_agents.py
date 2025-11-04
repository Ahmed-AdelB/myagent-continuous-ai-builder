"""
Comprehensive test suite for AI agents
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

# Import agents
from core.agents.base_agent import PersistentAgent
from core.agents.coder_agent import CoderAgent
from core.agents.tester_agent import TesterAgent
from core.agents.debugger_agent import DebuggerAgent
from core.agents.architect_agent import ArchitectAgent
from core.agents.analyzer_agent import AnalyzerAgent
from core.agents.ui_refiner_agent import UIRefinerAgent


class TestPersistentAgent:
    """Test base agent functionality"""

    @pytest.fixture
    def agent(self):
        return PersistentAgent(
            name="test_agent",
            role="Test Agent",
            capabilities=["test", "mock"]
        )

    def test_agent_initialization(self, agent):
        assert agent.name == "test_agent"
        assert agent.role == "Test Agent"
        assert "test" in agent.capabilities
        assert agent.status == "idle"

    def test_agent_state_management(self, agent):
        # Test saving state
        agent.save_state()
        state_file = agent.checkpoint_dir / f"{agent.name}_state.json"
        assert state_file.exists()

        # Test loading state
        agent.status = "working"
        agent.current_task = "test_task"
        agent.save_state()

        new_agent = PersistentAgent(
            name="test_agent",
            role="Test Agent",
            capabilities=["test"]
        )
        new_agent.load_state()
        assert new_agent.status == "working"
        assert new_agent.current_task == "test_task"

    @pytest.mark.asyncio
    async def test_checkpoint_recovery(self, agent):
        # Create checkpoint
        agent.status = "working"
        agent.current_task = "important_task"
        checkpoint_id = agent.create_checkpoint()

        # Simulate failure
        agent.status = "error"
        agent.current_task = None

        # Recover from checkpoint
        agent.recover_from_checkpoint(checkpoint_id)
        assert agent.status == "working"
        assert agent.current_task == "important_task"

    def test_capability_check(self, agent):
        assert agent.has_capability("test")
        assert not agent.has_capability("nonexistent")


class TestCoderAgent:
    """Test Coder Agent functionality"""

    @pytest.fixture
    def coder(self):
        return CoderAgent()

    def test_coder_initialization(self, coder):
        assert coder.name == "coder_agent"
        assert "code_generation" in coder.capabilities
        assert "refactoring" in coder.capabilities

    @pytest.mark.asyncio
    async def test_generate_code(self, coder):
        task = {
            "type": "generate_function",
            "description": "Create a function to calculate fibonacci",
            "language": "python"
        }

        with patch.object(coder, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
            result = await coder.execute(task)

            assert result["status"] == "success"
            assert "fibonacci" in result["code"]
            assert result["language"] == "python"

    @pytest.mark.asyncio
    async def test_refactor_code(self, coder):
        task = {
            "type": "refactor",
            "code": "def add(a,b): return a+b",
            "improvements": ["add type hints", "add docstring"]
        }

        with patch.object(coder, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = """
def add(a: int, b: int) -> int:
    '''Add two integers and return the sum.'''
    return a + b
"""
            result = await coder.execute(task)

            assert result["status"] == "success"
            assert "int" in result["refactored_code"]
            assert "'''" in result["refactored_code"]

    @pytest.mark.asyncio
    async def test_error_handling(self, coder):
        task = {
            "type": "invalid_task"
        }

        result = await coder.execute(task)
        assert result["status"] == "error"
        assert "error" in result


class TestTesterAgent:
    """Test Tester Agent functionality"""

    @pytest.fixture
    def tester(self):
        return TesterAgent()

    def test_tester_initialization(self, tester):
        assert tester.name == "tester_agent"
        assert "test_generation" in tester.capabilities
        assert "coverage_analysis" in tester.capabilities

    @pytest.mark.asyncio
    async def test_generate_unit_tests(self, tester):
        task = {
            "type": "generate_tests",
            "code": """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""",
            "framework": "pytest"
        }

        with patch.object(tester, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = """
import pytest

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0
"""
            result = await tester.execute(task)

            assert result["status"] == "success"
            assert "test_add" in result["tests"]
            assert "test_multiply" in result["tests"]
            assert result["framework"] == "pytest"

    @pytest.mark.asyncio
    async def test_coverage_analysis(self, tester):
        task = {
            "type": "analyze_coverage",
            "project_path": "/fake/path",
            "target_coverage": 80
        }

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "Total coverage: 85%"
            mock_run.return_value.returncode = 0

            result = await tester.execute(task)

            assert result["status"] == "success"
            assert result["coverage"] == 85
            assert result["meets_target"] is True


class TestDebuggerAgent:
    """Test Debugger Agent functionality"""

    @pytest.fixture
    def debugger(self):
        return DebuggerAgent()

    def test_debugger_initialization(self, debugger):
        assert debugger.name == "debugger_agent"
        assert "error_analysis" in debugger.capabilities
        assert "fix_suggestion" in debugger.capabilities

    @pytest.mark.asyncio
    async def test_analyze_error(self, debugger):
        task = {
            "type": "analyze_error",
            "error_message": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
            "stack_trace": "File 'test.py', line 5, in add\n    return a + b",
            "code_context": "def add(a, b):\n    return a + b"
        }

        with patch.object(debugger, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = json.dumps({
                "error_type": "TypeError",
                "cause": "Attempting to add incompatible types",
                "suggestion": "Ensure both operands are of the same type or convert them",
                "fix": "def add(a, b):\n    return int(a) + int(b)"
            })

            result = await debugger.execute(task)

            assert result["status"] == "success"
            assert result["analysis"]["error_type"] == "TypeError"
            assert "suggestion" in result["analysis"]
            assert "fix" in result["analysis"]

    @pytest.mark.asyncio
    async def test_suggest_fix(self, debugger):
        task = {
            "type": "suggest_fix",
            "error": {
                "type": "IndexError",
                "message": "list index out of range",
                "line": 10
            },
            "code": "items = [1, 2, 3]\nprint(items[5])"
        }

        result = await debugger.execute(task)

        assert result["status"] == "success"
        assert "suggestion" in result
        assert result["confidence"] > 0


class TestArchitectAgent:
    """Test Architect Agent functionality"""

    @pytest.fixture
    def architect(self):
        return ArchitectAgent()

    def test_architect_initialization(self, architect):
        assert architect.name == "architect_agent"
        assert "design_system" in architect.capabilities
        assert "review_architecture" in architect.capabilities

    @pytest.mark.asyncio
    async def test_design_system(self, architect):
        task = {
            "type": "design_system",
            "requirements": [
                "REST API with authentication",
                "PostgreSQL database",
                "Redis caching"
            ],
            "constraints": ["Must handle 1000 req/s", "99.9% uptime"]
        }

        result = await architect.execute(task)

        assert result["status"] == "success"
        assert "architecture" in result
        assert "components" in result["architecture"]
        assert "design_patterns" in result["architecture"]

    @pytest.mark.asyncio
    async def test_review_architecture(self, architect):
        task = {
            "type": "review_architecture",
            "current_architecture": {
                "frontend": "React",
                "backend": "Django",
                "database": "SQLite"
            },
            "issues": ["Slow queries", "No caching"]
        }

        result = await architect.execute(task)

        assert result["status"] == "success"
        assert "recommendations" in result
        assert "improvements" in result
        assert result["scalability_score"] >= 0


class TestAnalyzerAgent:
    """Test Analyzer Agent functionality"""

    @pytest.fixture
    def analyzer(self):
        return AnalyzerAgent()

    def test_analyzer_initialization(self, analyzer):
        assert analyzer.name == "analyzer_agent"
        assert "monitor_metrics" in analyzer.capabilities
        assert "detect_anomalies" in analyzer.capabilities

    @pytest.mark.asyncio
    async def test_analyze_metrics(self, analyzer):
        task = {
            "type": "analyze_metrics",
            "metrics": {
                "cpu_usage": [45, 50, 48, 52, 95, 47, 49],
                "memory_usage": [60, 62, 61, 63, 64, 62, 61],
                "response_time": [100, 105, 102, 500, 103, 101, 104]
            }
        }

        result = await analyzer.execute(task)

        assert result["status"] == "success"
        assert "anomalies" in result
        assert len(result["anomalies"]) > 0
        assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_predict_trends(self, analyzer):
        task = {
            "type": "predict_trends",
            "historical_data": {
                "test_coverage": [70, 72, 75, 77, 79, 81],
                "bug_count": [10, 9, 8, 7, 6, 5]
            },
            "forecast_periods": 3
        }

        result = await analyzer.execute(task)

        assert result["status"] == "success"
        assert "predictions" in result
        assert len(result["predictions"]["test_coverage"]) == 3
        assert result["predictions"]["test_coverage"][0] > 81


class TestUIRefinerAgent:
    """Test UI Refiner Agent functionality"""

    @pytest.fixture
    def ui_refiner(self):
        return UIRefinerAgent()

    def test_ui_refiner_initialization(self, ui_refiner):
        assert ui_refiner.name == "ui_refiner_agent"
        assert "analyze_ui" in ui_refiner.capabilities
        assert "optimize_accessibility" in ui_refiner.capabilities

    @pytest.mark.asyncio
    async def test_analyze_ui(self, ui_refiner):
        task = {
            "type": "analyze_ui",
            "html": "<div><button>Click</button><input type='text'></div>",
            "css": "button { color: #ccc; font-size: 10px; }"
        }

        result = await ui_refiner.execute(task)

        assert result["status"] == "success"
        assert "issues" in result
        assert "improvements" in result
        assert result["accessibility_score"] >= 0

    @pytest.mark.asyncio
    async def test_improve_accessibility(self, ui_refiner):
        task = {
            "type": "improve_accessibility",
            "components": [
                {"type": "button", "text": "Submit", "aria_label": None},
                {"type": "image", "src": "logo.png", "alt": None}
            ]
        }

        result = await ui_refiner.execute(task)

        assert result["status"] == "success"
        assert "improvements" in result
        assert len(result["improvements"]) > 0
        assert result["wcag_compliance"] is not None


class TestAgentIntegration:
    """Test agent integration and coordination"""

    @pytest.mark.asyncio
    async def test_agent_collaboration(self):
        """Test multiple agents working together"""
        coder = CoderAgent()
        tester = TesterAgent()
        debugger = DebuggerAgent()

        # Coder generates code
        code_task = {
            "type": "generate_function",
            "description": "Create a divide function",
            "language": "python"
        }

        with patch.object(coder, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = """
def divide(a, b):
    return a / b
"""
            code_result = await coder.execute(code_task)

        # Tester generates tests
        test_task = {
            "type": "generate_tests",
            "code": code_result["code"],
            "framework": "pytest"
        }

        with patch.object(tester, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = """
def test_divide():
    assert divide(10, 2) == 5
    assert divide(10, 0) == None  # This will fail
"""
            test_result = await tester.execute(test_task)

        # Debugger analyzes the error
        debug_task = {
            "type": "analyze_error",
            "error_message": "ZeroDivisionError: division by zero",
            "code_context": code_result["code"]
        }

        with patch.object(debugger, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = json.dumps({
                "error_type": "ZeroDivisionError",
                "cause": "Division by zero not handled",
                "fix": "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b"
            })
            debug_result = await debugger.execute(debug_task)

        assert code_result["status"] == "success"
        assert test_result["status"] == "success"
        assert debug_result["status"] == "success"
        assert "fix" in debug_result["analysis"]


# Test fixtures
@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator"""
    orchestrator = Mock()
    orchestrator.project_name = "test_project"
    orchestrator.send_message = AsyncMock()
    orchestrator.get_agent = Mock()
    return orchestrator


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core.agents", "--cov-report=term-missing"])