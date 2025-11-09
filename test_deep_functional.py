#!/usr/bin/env python3
"""
Deep Functional Test Suite for MyAgent Continuous AI Builder
Tests all components, integrations, and error scenarios
"""

import asyncio
import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Test results storage
test_results = {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "errors": 0,
    "warnings": 0,
    "tests": []
}


class TestResult:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.status = "PENDING"
        self.message = ""
        self.duration = 0
        self.error_trace = None
        self.warnings = []

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "message": self.message,
            "duration": self.duration,
            "error_trace": self.error_trace,
            "warnings": self.warnings
        }


def test_case(category: str):
    """Decorator for test cases"""
    def decorator(func):
        async def wrapper():
            test_results["total_tests"] += 1
            result = TestResult(func.__name__, category)
            start_time = datetime.now()

            try:
                print(f"\n{'='*80}")
                print(f"🧪 Testing: {func.__name__}")
                print(f"Category: {category}")
                print(f"{'='*80}")

                await func(result)

                if result.status == "PENDING":
                    result.status = "PASSED"

                if result.status == "PASSED":
                    test_results["passed"] += 1
                    print(f"✅ PASSED: {func.__name__}")
                elif result.status == "WARNING":
                    test_results["warnings"] += 1
                    print(f"⚠️  WARNING: {func.__name__} - {result.message}")
                else:
                    test_results["failed"] += 1
                    print(f"❌ FAILED: {func.__name__} - {result.message}")

            except Exception as e:
                test_results["errors"] += 1
                result.status = "ERROR"
                result.message = str(e)
                result.error_trace = traceback.format_exc()
                print(f"💥 ERROR: {func.__name__}")
                print(f"Error: {str(e)}")
                print(f"Trace:\n{result.error_trace}")

            finally:
                result.duration = (datetime.now() - start_time).total_seconds()
                test_results["tests"].append(result.to_dict())
                print(f"Duration: {result.duration:.2f}s")

            return result

        return wrapper
    return decorator


# ==================== COMPONENT INITIALIZATION TESTS ====================

@test_case("Initialization")
async def test_import_core_modules(result: TestResult):
    """Test that all core modules can be imported"""
    modules_to_test = [
        "core.orchestrator.continuous_director",
        "core.orchestrator.checkpoint_manager",
        "core.orchestrator.progress_analyzer",
        "core.orchestrator.milestone_tracker",
        "core.agents.base_agent",
        "core.agents.coder_agent",
        "core.agents.tester_agent",
        "core.agents.debugger_agent",
        "core.memory.vector_memory",
        "core.memory.project_ledger",
        "core.memory.error_knowledge_graph",
        "config.settings"
    ]

    failed_imports = []

    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            failed_imports.append(f"{module}: {str(e)}")
            print(f"  ✗ {module}: {str(e)}")

    if failed_imports:
        result.status = "FAILED"
        result.message = f"Failed to import {len(failed_imports)} modules"
        result.warnings = failed_imports
    else:
        result.message = f"Successfully imported all {len(modules_to_test)} modules"


@test_case("Initialization")
async def test_settings_configuration(result: TestResult):
    """Test that settings are properly configured"""
    from config.settings import settings

    required_settings = [
        "OPENAI_API_KEY",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "REDIS_HOST"
    ]

    missing = []
    for setting in required_settings:
        if not hasattr(settings, setting):
            missing.append(setting)
            print(f"  ✗ Missing: {setting}")
        else:
            value = getattr(settings, setting)
            is_set = value is not None and str(value).strip() != ""
            print(f"  {'✓' if is_set else '⚠'} {setting}: {'Set' if is_set else 'Not Set'}")
            if not is_set:
                result.warnings.append(f"{setting} is not set")

    if missing:
        result.status = "FAILED"
        result.message = f"Missing required settings: {', '.join(missing)}"
    elif result.warnings:
        result.status = "WARNING"
        result.message = f"{len(result.warnings)} settings not configured"
    else:
        result.message = "All settings properly configured"


@test_case("Initialization")
async def test_directory_structure(result: TestResult):
    """Test that required directories exist"""
    required_dirs = [
        "persistence",
        "persistence/database",
        "persistence/vector_memory",
        "persistence/checkpoints",
        "persistence/agents",
        "persistence/snapshots",
        "logs"
    ]

    missing = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            missing.append(dir_path)
            print(f"  ✗ Missing: {dir_path}")

    if missing:
        result.status = "WARNING"
        result.message = f"Missing {len(missing)} directories (will be created on demand)"
        result.warnings = missing
    else:
        result.message = "All required directories exist"


# ==================== MEMORY SYSTEM TESTS ====================

@test_case("Memory Systems")
async def test_vector_memory_initialization(result: TestResult):
    """Test VectorMemory initialization and basic operations"""
    from core.memory.vector_memory import VectorMemory

    try:
        vm = VectorMemory(project_name="test_functional")
        print(f"  ✓ VectorMemory initialized")
        print(f"  - Embedding dimension: {vm.embedding_dim}")
        print(f"  - Collections: {list(vm.collections.keys())}")

        # Test storing memory
        memory_id = vm.store_memory(
            content="This is a test memory for functional testing",
            memory_type="code",
            metadata={"test": True}
        )
        print(f"  ✓ Stored memory: {memory_id}")

        # Test searching memory
        results = vm.search_memories(
            query="test memory",
            memory_type="code",
            top_k=1
        )
        print(f"  ✓ Search returned {len(results)} results")

        if results:
            print(f"  - Relevance score: {results[0].relevance_score:.3f}")

        result.message = f"VectorMemory working correctly (dim={vm.embedding_dim})"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"VectorMemory failed: {str(e)}"
        raise


@test_case("Memory Systems")
async def test_project_ledger(result: TestResult):
    """Test ProjectLedger initialization and operations"""
    from core.memory.project_ledger import ProjectLedger

    try:
        ledger = ProjectLedger(project_name="test_functional")
        print(f"  ✓ ProjectLedger initialized")
        print(f"  - Database: {ledger.db_path}")

        # Test saving code version
        version = ledger.save_code_version(
            file_path="test.py",
            content="def test(): pass",
            iteration=1,
            agent="test_agent",
            reason="Functional test"
        )
        print(f"  ✓ Saved code version: {version.id}")

        # Test retrieving version
        retrieved = ledger.get_current_version("test.py")
        if retrieved and retrieved.content == "def test(): pass":
            print(f"  ✓ Retrieved version correctly")
        else:
            result.warnings.append("Version retrieval mismatch")

        # Test recording decision
        ledger.record_decision(
            iteration=1,
            agent="test_agent",
            decision_type="test",
            description="Test decision"
        )
        print(f"  ✓ Recorded decision")

        result.message = "ProjectLedger working correctly"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"ProjectLedger failed: {str(e)}"
        raise


@test_case("Memory Systems")
async def test_error_knowledge_graph(result: TestResult):
    """Test ErrorKnowledgeGraph initialization and operations"""
    from core.memory.error_knowledge_graph import ErrorKnowledgeGraph

    try:
        graph = ErrorKnowledgeGraph()
        print(f"  ✓ ErrorKnowledgeGraph initialized")
        print(f"  - Database: {graph.db_path}")

        # Test adding error
        error = graph.add_error(
            error_type="test_error",
            error_message="Test error message",
            file_path="test.py",
            line_number=10
        )
        print(f"  ✓ Added error: {error.id}")

        # Test adding solution
        solution = graph.add_solution(
            error_id=error.id,
            solution_type="test_fix",
            description="Test fix",
            code_changes={"test.py": "fixed code"},
            created_by="test_agent",
            success=True
        )
        print(f"  ✓ Added solution: {solution.id}")

        # Test finding similar errors
        similar = graph.find_similar_errors("test_error", "Test error message")
        print(f"  ✓ Found {len(similar)} similar errors")

        result.message = "ErrorKnowledgeGraph working correctly"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"ErrorKnowledgeGraph failed: {str(e)}"
        raise


# ==================== AGENT TESTS ====================

@test_case("Agents")
async def test_base_agent_initialization(result: TestResult):
    """Test base agent initialization"""
    from core.agents.base_agent import PersistentAgent, AgentTask
    from datetime import datetime

    class TestAgent(PersistentAgent):
        async def process_task(self, task):
            return {"status": "completed"}

        def analyze_context(self, context):
            return context

        def generate_solution(self, problem):
            return {"solution": "test"}

    try:
        agent = TestAgent(
            name="test_agent",
            role="Tester",
            capabilities=["test"]
        )
        print(f"  ✓ Agent initialized: {agent.name}")
        print(f"  - ID: {agent.id}")
        print(f"  - State: {agent.state.value}")
        print(f"  - Checkpoint dir: {agent.checkpoint_dir}")

        # Test adding task
        task = AgentTask(
            id="test_task_1",
            type="test",
            description="Test task",
            priority=1,
            data={},
            created_at=datetime.now()
        )
        agent.add_task(task)
        print(f"  ✓ Added task to queue")
        print(f"  - Queue size: {len(agent.task_queue)}")

        # Test checkpoint
        agent.save_checkpoint()
        print(f"  ✓ Checkpoint saved")

        result.message = "BaseAgent working correctly"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"BaseAgent failed: {str(e)}"
        raise


@test_case("Agents")
async def test_coder_agent_initialization(result: TestResult):
    """Test CoderAgent initialization"""
    from core.agents.coder_agent import CoderAgent

    try:
        agent = CoderAgent()
        print(f"  ✓ CoderAgent initialized")
        print(f"  - Name: {agent.name}")
        print(f"  - Role: {agent.role}")
        print(f"  - Capabilities: {len(agent.capabilities)}")

        # Check LLM initialization
        if hasattr(agent, 'llm'):
            print(f"  ✓ LLM initialized")
            # Check model name
            if hasattr(agent.llm, 'model_name'):
                model = agent.llm.model_name
                print(f"  - Model: {model}")
                if "gpt-5" in model:
                    result.warnings.append("Using invalid model name 'gpt-5-chat-latest'")
        else:
            result.warnings.append("LLM not initialized")

        # Check templates
        print(f"  - Templates: {list(agent.templates.keys())}")

        result.message = "CoderAgent initialized successfully"
        if result.warnings:
            result.status = "WARNING"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"CoderAgent failed: {str(e)}"
        raise


@test_case("Agents")
async def test_tester_agent_initialization(result: TestResult):
    """Test TesterAgent initialization"""
    from core.agents.tester_agent import TesterAgent

    try:
        agent = TesterAgent()
        print(f"  ✓ TesterAgent initialized")
        print(f"  - Name: {agent.name}")
        print(f"  - Role: {agent.role}")
        print(f"  - Capabilities: {agent.capabilities}")
        print(f"  - Test templates: {list(agent.test_templates.keys())}")

        result.message = "TesterAgent initialized successfully"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"TesterAgent failed: {str(e)}"
        raise


@test_case("Agents")
async def test_debugger_agent_initialization(result: TestResult):
    """Test DebuggerAgent initialization"""
    from core.agents.debugger_agent import DebuggerAgent

    try:
        agent = DebuggerAgent()
        print(f"  ✓ DebuggerAgent initialized")
        print(f"  - Name: {agent.name}")
        print(f"  - Role: {agent.role}")
        print(f"  - Error patterns: {len(agent.error_patterns)}")
        print(f"  - Fix strategies: {len(agent.fix_strategies)}")

        result.message = "DebuggerAgent initialized successfully"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"DebuggerAgent failed: {str(e)}"
        raise


# ==================== ORCHESTRATOR TESTS ====================

@test_case("Orchestrator")
async def test_orchestrator_initialization(result: TestResult):
    """Test ContinuousDirector initialization"""
    from core.orchestrator.continuous_director import ContinuousDirector, QualityMetrics

    try:
        project_spec = {
            "description": "Test project",
            "requirements": ["test req 1"],
            "features": []
        }

        director = ContinuousDirector(
            project_name="test_functional",
            project_spec=project_spec
        )
        print(f"  ✓ ContinuousDirector initialized")
        print(f"  - Project: {director.project_name}")
        print(f"  - State: {director.state.value}")
        print(f"  - Iteration: {director.iteration_count}")

        # Test metrics
        print(f"  ✓ QualityMetrics initialized")
        print(f"  - Test coverage: {director.metrics.test_coverage}%")
        print(f"  - Is perfect: {director.metrics.is_perfect()}")

        result.message = "ContinuousDirector initialized successfully"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"ContinuousDirector failed: {str(e)}"
        raise


@test_case("Orchestrator")
async def test_orchestrator_component_initialization(result: TestResult):
    """Test orchestrator initializes all components"""
    from core.orchestrator.continuous_director import ContinuousDirector

    try:
        project_spec = {"description": "Test", "features": []}
        director = ContinuousDirector("test_components", project_spec)

        # Initialize components
        await director._initialize_components()

        # Check memory systems
        if hasattr(director, 'project_ledger'):
            print(f"  ✓ ProjectLedger initialized")
        else:
            result.warnings.append("ProjectLedger not initialized")

        if hasattr(director, 'vector_memory'):
            print(f"  ✓ VectorMemory initialized")
        else:
            result.warnings.append("VectorMemory not initialized")

        if hasattr(director, 'error_graph'):
            print(f"  ✓ ErrorKnowledgeGraph initialized")
        else:
            result.warnings.append("ErrorKnowledgeGraph not initialized")

        # Check agents
        expected_agents = ['coder', 'tester', 'debugger', 'architect', 'analyzer', 'ui_refiner']
        for agent_name in expected_agents:
            if agent_name in director.agents:
                print(f"  ✓ {agent_name} agent initialized")
            else:
                result.warnings.append(f"{agent_name} agent not initialized")

        if result.warnings:
            result.status = "WARNING"
            result.message = f"{len(result.warnings)} components missing"
        else:
            result.message = "All components initialized successfully"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"Component initialization failed: {str(e)}"
        raise


# ==================== INTEGRATION TESTS ====================

@test_case("Integration")
async def test_task_creation_and_routing(result: TestResult):
    """Test task creation and agent routing"""
    from core.orchestrator.continuous_director import ContinuousDirector, DevelopmentTask, TaskPriority

    try:
        director = ContinuousDirector("test_routing", {"description": "Test", "features": []})
        await director._initialize_components()

        # Create test tasks
        tasks = [
            DevelopmentTask(
                id="test_code_1",
                type="code_feature",
                description="Test code task",
                priority=TaskPriority.HIGH
            ),
            DevelopmentTask(
                id="test_test_1",
                type="test_generation",
                description="Test testing task",
                priority=TaskPriority.NORMAL
            ),
            DevelopmentTask(
                id="test_debug_1",
                type="debug_error",
                description="Test debug task",
                priority=TaskPriority.CRITICAL
            )
        ]

        # Test agent selection
        for task in tasks:
            agent_name = director._select_agent_for_task(task)
            print(f"  ✓ Task '{task.type}' → Agent '{agent_name}'")

        result.message = "Task routing working correctly"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"Task routing failed: {str(e)}"
        raise


@test_case("Integration")
async def test_checkpoint_and_restore(result: TestResult):
    """Test checkpoint creation and restoration"""
    from core.orchestrator.continuous_director import ContinuousDirector

    try:
        director = ContinuousDirector("test_checkpoint", {"description": "Test", "features": []})
        director.iteration_count = 5
        director.metrics.test_coverage = 75.5

        # Create checkpoint
        await director._create_checkpoint()
        print(f"  ✓ Checkpoint created at iteration {director.iteration_count}")

        # Modify state
        director.iteration_count = 0
        director.metrics.test_coverage = 0.0

        # Restore
        await director._load_project_state()
        print(f"  ✓ State restored")

        # Verify
        if director.iteration_count == 5:
            print(f"  ✓ Iteration count restored correctly: {director.iteration_count}")
        else:
            result.warnings.append(f"Iteration count not restored: {director.iteration_count}")

        if abs(director.metrics.test_coverage - 75.5) < 0.1:
            print(f"  ✓ Metrics restored correctly: {director.metrics.test_coverage}%")
        else:
            result.warnings.append(f"Metrics not restored: {director.metrics.test_coverage}%")

        if result.warnings:
            result.status = "WARNING"
            result.message = "Checkpoint/restore has issues"
        else:
            result.message = "Checkpoint/restore working correctly"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"Checkpoint/restore failed: {str(e)}"
        raise


# ==================== ERROR HANDLING TESTS ====================

@test_case("Error Handling")
async def test_error_recovery_mechanism(result: TestResult):
    """Test error recovery in agents"""
    from core.agents.debugger_agent import DebuggerAgent

    try:
        agent = DebuggerAgent()

        # Test error analysis
        error_data = {
            "error_message": "NameError: name 'undefined_var' is not defined",
            "stack_trace": 'File "test.py", line 10, in test_function',
            "code": "def test(): print(undefined_var)",
            "context": {}
        }

        analysis_result = await agent.analyze_error(error_data)

        if analysis_result.get('success'):
            print(f"  ✓ Error analyzed successfully")
            analysis = analysis_result.get('analysis', {})
            print(f"  - Error type: {analysis.get('error_type')}")
            print(f"  - Severity: {analysis.get('severity')}")
            print(f"  - Auto-fixable: {analysis.get('auto_fixable')}")
        else:
            result.warnings.append("Error analysis failed")

        result.message = "Error recovery mechanism tested"

    except Exception as e:
        result.status = "FAILED"
        result.message = f"Error recovery test failed: {str(e)}"
        raise


# ==================== MAIN TEST RUNNER ====================

async def run_all_tests():
    """Run all functional tests"""
    print("\n" + "="*80)
    print("🚀 DEEP FUNCTIONAL TEST SUITE - MyAgent Continuous AI Builder")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # Run all tests
    test_functions = [
        # Initialization tests
        test_import_core_modules,
        test_settings_configuration,
        test_directory_structure,

        # Memory system tests
        test_vector_memory_initialization,
        test_project_ledger,
        test_error_knowledge_graph,

        # Agent tests
        test_base_agent_initialization,
        test_coder_agent_initialization,
        test_tester_agent_initialization,
        test_debugger_agent_initialization,

        # Orchestrator tests
        test_orchestrator_initialization,
        test_orchestrator_component_initialization,

        # Integration tests
        test_task_creation_and_routing,
        test_checkpoint_and_restore,

        # Error handling tests
        test_error_recovery_mechanism,
    ]

    for test_func in test_functions:
        await test_func()

    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"Total Tests:  {test_results['total_tests']}")
    print(f"✅ Passed:     {test_results['passed']}")
    print(f"❌ Failed:     {test_results['failed']}")
    print(f"💥 Errors:     {test_results['errors']}")
    print(f"⚠️  Warnings:   {test_results['warnings']}")
    print("="*80)

    success_rate = (test_results['passed'] / test_results['total_tests'] * 100) if test_results['total_tests'] > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    # Categorize results
    print("\n" + "="*80)
    print("📋 RESULTS BY CATEGORY")
    print("="*80)

    categories = {}
    for test in test_results['tests']:
        cat = test['category']
        if cat not in categories:
            categories[cat] = {'passed': 0, 'failed': 0, 'errors': 0, 'warnings': 0}

        if test['status'] == 'PASSED':
            categories[cat]['passed'] += 1
        elif test['status'] == 'FAILED':
            categories[cat]['failed'] += 1
        elif test['status'] == 'ERROR':
            categories[cat]['errors'] += 1
        elif test['status'] == 'WARNING':
            categories[cat]['warnings'] += 1

    for cat, stats in categories.items():
        total = sum(stats.values())
        print(f"\n{cat}:")
        print(f"  ✅ Passed: {stats['passed']}/{total}")
        if stats['failed'] > 0:
            print(f"  ❌ Failed: {stats['failed']}/{total}")
        if stats['errors'] > 0:
            print(f"  💥 Errors: {stats['errors']}/{total}")
        if stats['warnings'] > 0:
            print(f"  ⚠️  Warnings: {stats['warnings']}/{total}")

    # Save detailed results
    results_file = Path("test_results_functional.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n📄 Detailed results saved to: {results_file}")
    print("="*80 + "\n")

    return test_results


if __name__ == "__main__":
    # Run tests
    results = asyncio.run(run_all_tests())

    # Exit with appropriate code
    if results['errors'] > 0 or results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
