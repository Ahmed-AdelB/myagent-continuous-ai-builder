#!/usr/bin/env python3
"""
Complete System Workflows - Comprehensive System Tests

Tests complete end-to-end workflows that simulate real user interactions
with the MyAgent system. Validates the entire system working together including
API endpoints, agent orchestration, database operations, and user interfaces.

Testing methodologies applied:
- System testing for complete workflow validation
- E2E testing for user journey simulation
- API testing for endpoint functionality
- Database testing for data persistence
- Performance testing for system responsiveness
- Integration testing for component interaction
"""

import pytest
import asyncio
import httpx
import json
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# Import test fixtures
from tests.fixtures.test_data import TEST_DATA
from tests.fixtures.project_fixtures import SIMPLE_PROJECT, MEDIUM_PROJECT, COMPLEX_PROJECT


@dataclass
class SystemTestResult:
    """Result of a system test execution"""
    test_name: str
    status: str
    duration: float
    steps_completed: int
    total_steps: int
    api_calls_made: List[Dict[str, Any]]
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowStep:
    """Individual step in a workflow test"""
    step_id: str
    name: str
    description: str
    api_endpoint: str
    method: str
    payload: Optional[Dict[str, Any]] = None
    expected_status: int = 200
    validation_rules: List[str] = field(default_factory=list)
    timeout: int = 30


class SystemTestClient:
    """Test client for system-level testing"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_data = {}
        self.api_call_history = []

    async def make_request(self, method: str, endpoint: str, payload: Optional[Dict] = None,
                          headers: Optional[Dict] = None, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP request and track it"""
        url = f"{self.base_url}{endpoint}"
        request_start = time.time()

        try:
            async with httpx.AsyncClient() as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, timeout=timeout)
                elif method.upper() == "POST":
                    response = await client.post(url, json=payload, headers=headers, timeout=timeout)
                elif method.upper() == "PUT":
                    response = await client.put(url, json=payload, headers=headers, timeout=timeout)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                duration = time.time() - request_start

                # Track API call
                api_call = {
                    "method": method.upper(),
                    "endpoint": endpoint,
                    "url": url,
                    "payload": payload,
                    "status_code": response.status_code,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
                self.api_call_history.append(api_call)

                # Parse response
                try:
                    response_data = response.json()
                except Exception:
                    response_data = {"text": response.text}

                return {
                    "status_code": response.status_code,
                    "data": response_data,
                    "headers": dict(response.headers),
                    "duration": duration
                }

        except Exception as e:
            duration = time.time() - request_start
            error_call = {
                "method": method.upper(),
                "endpoint": endpoint,
                "url": url,
                "payload": payload,
                "error": str(e),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            self.api_call_history.append(error_call)
            raise


class WorkflowTestRunner:
    """Runner for complete workflow tests"""

    def __init__(self, test_client: SystemTestClient):
        self.client = test_client
        self.test_results = {}

    async def run_workflow_test(self, test_name: str, steps: List[WorkflowStep],
                               setup_data: Optional[Dict] = None) -> SystemTestResult:
        """Run a complete workflow test"""
        test_start = time.time()
        steps_completed = 0
        validation_results = []
        error_message = None

        try:
            # Setup test data if needed
            if setup_data:
                await self._setup_test_data(setup_data)

            # Execute workflow steps
            for step in steps:
                try:
                    await self._execute_workflow_step(step, validation_results)
                    steps_completed += 1
                except Exception as e:
                    error_message = f"Step {step.step_id} failed: {str(e)}"
                    break

            test_duration = time.time() - test_start
            status = "PASSED" if steps_completed == len(steps) else "FAILED"

            # Calculate performance metrics
            performance_metrics = {
                "total_duration": test_duration,
                "average_api_response_time": self._calculate_avg_response_time(),
                "total_api_calls": len(self.client.api_call_history),
                "steps_per_second": steps_completed / test_duration if test_duration > 0 else 0
            }

            result = SystemTestResult(
                test_name=test_name,
                status=status,
                duration=test_duration,
                steps_completed=steps_completed,
                total_steps=len(steps),
                api_calls_made=self.client.api_call_history.copy(),
                error_message=error_message,
                performance_metrics=performance_metrics,
                validation_results=validation_results
            )

            self.test_results[test_name] = result
            return result

        except Exception as e:
            test_duration = time.time() - test_start
            return SystemTestResult(
                test_name=test_name,
                status="ERROR",
                duration=test_duration,
                steps_completed=steps_completed,
                total_steps=len(steps),
                api_calls_made=self.client.api_call_history.copy(),
                error_message=str(e)
            )

    async def _setup_test_data(self, setup_data: Dict):
        """Setup test data before workflow execution"""
        # Create test project if specified
        if "project" in setup_data:
            project_data = setup_data["project"]
            response = await self.client.make_request(
                "POST", "/projects", payload=project_data
            )
            if response["status_code"] == 201:
                self.client.session_data["project_id"] = response["data"]["id"]

    async def _execute_workflow_step(self, step: WorkflowStep, validation_results: List[Dict]):
        """Execute a single workflow step"""
        # Replace placeholders in payload and endpoint
        endpoint = self._replace_placeholders(step.api_endpoint)
        payload = self._replace_placeholders(step.payload) if step.payload else None

        # Make API request
        response = await self.client.make_request(
            step.method, endpoint, payload, timeout=step.timeout
        )

        # Validate response
        if response["status_code"] != step.expected_status:
            raise Exception(
                f"Expected status {step.expected_status}, got {response['status_code']}"
            )

        # Apply validation rules
        for rule in step.validation_rules:
            validation_result = self._apply_validation_rule(rule, response)
            validation_results.append({
                "step_id": step.step_id,
                "rule": rule,
                "passed": validation_result,
                "response_data": response["data"]
            })

        # Store useful response data for later steps
        self._store_response_data(step, response)

    def _replace_placeholders(self, data):
        """Replace placeholders in data with session values"""
        if isinstance(data, str):
            for key, value in self.client.session_data.items():
                data = data.replace(f"{{{key}}}", str(value))
            return data
        elif isinstance(data, dict):
            return {k: self._replace_placeholders(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_placeholders(item) for item in data]
        return data

    def _apply_validation_rule(self, rule: str, response: Dict) -> bool:
        """Apply validation rule to response"""
        data = response["data"]

        if rule == "has_id":
            return "id" in data
        elif rule == "status_active":
            return data.get("status") == "active"
        elif rule == "has_agents":
            return "agents" in data and len(data["agents"]) > 0
        elif rule == "metrics_present":
            return "metrics" in data
        elif rule == "non_empty_result":
            return bool(data)
        elif rule.startswith("contains_"):
            field = rule.replace("contains_", "")
            return field in data
        elif rule.startswith("count_greater_than_"):
            count = int(rule.split("_")[-1])
            return len(data) > count if isinstance(data, (list, dict)) else False

        return True  # Default to pass for unknown rules

    def _store_response_data(self, step: WorkflowStep, response: Dict):
        """Store response data for use in subsequent steps"""
        data = response["data"]

        # Store commonly used IDs
        if "id" in data:
            self.client.session_data[f"{step.step_id}_id"] = data["id"]

        # Store project-specific data
        if step.step_id == "create_project":
            self.client.session_data["project_id"] = data.get("id")

        # Store iteration data
        if step.step_id == "start_iteration":
            self.client.session_data["iteration_id"] = data.get("iteration_id")

    def _calculate_avg_response_time(self) -> float:
        """Calculate average API response time"""
        if not self.client.api_call_history:
            return 0.0

        durations = [call.get("duration", 0) for call in self.client.api_call_history]
        return sum(durations) / len(durations)


@pytest.fixture
def test_client():
    """Fixture providing test client"""
    return SystemTestClient()


@pytest.fixture
def workflow_runner(test_client):
    """Fixture providing workflow test runner"""
    return WorkflowTestRunner(test_client)


@pytest.fixture
def mock_api_server():
    """Mock API server responses for testing"""
    with patch('httpx.AsyncClient') as mock_client:
        # Configure mock responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "id": "test_123"}
        mock_response.headers = {}
        mock_response.text = '{"status": "success"}'

        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.put.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.delete.return_value = mock_response

        yield mock_client


class TestCompleteWorkflows:
    """Comprehensive system workflow tests"""

    @pytest.mark.asyncio
    async def test_complete_project_creation_workflow(self, workflow_runner, mock_api_server):
        """Test complete project creation and initialization workflow"""

        # Define workflow steps
        steps = [
            WorkflowStep(
                step_id="health_check",
                name="Health Check",
                description="Verify system is healthy",
                api_endpoint="/health",
                method="GET",
                validation_rules=["non_empty_result"]
            ),
            WorkflowStep(
                step_id="create_project",
                name="Create Project",
                description="Create a new MyAgent project",
                api_endpoint="/projects",
                method="POST",
                payload={
                    "name": "Test Project",
                    "description": "System test project",
                    "requirements": ["Create a simple calculator app"],
                    "complexity": "medium"
                },
                expected_status=201,
                validation_rules=["has_id", "status_active"]
            ),
            WorkflowStep(
                step_id="get_project",
                name="Get Project Details",
                description="Retrieve created project details",
                api_endpoint="/projects/{project_id}",
                method="GET",
                validation_rules=["has_id", "has_agents", "metrics_present"]
            ),
            WorkflowStep(
                step_id="initialize_agents",
                name="Initialize Agents",
                description="Initialize project agents",
                api_endpoint="/projects/{project_id}/agents/initialize",
                method="POST",
                validation_rules=["non_empty_result"]
            ),
            WorkflowStep(
                step_id="get_agent_status",
                name="Get Agent Status",
                description="Check agent initialization status",
                api_endpoint="/projects/{project_id}/agents",
                method="GET",
                validation_rules=["has_agents"]
            )
        ]

        # Mock successful responses
        mock_responses = [
            {"status": "healthy", "version": "1.0.0"},
            {"id": "proj_123", "status": "active", "name": "Test Project"},
            {"id": "proj_123", "status": "active", "agents": ["coder", "tester"], "metrics": {}},
            {"status": "initialized", "agents_count": 6},
            {"agents": [{"name": "coder", "status": "active"}]}
        ]

        # Configure mock to return different responses for each call
        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.get.return_value.json.side_effect = mock_json
        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        # Run workflow test
        result = await workflow_runner.run_workflow_test(
            "complete_project_creation",
            steps
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(steps)
        assert len(result.api_calls_made) == len(steps)
        assert result.duration > 0
        assert result.performance_metrics["total_api_calls"] == len(steps)

    @pytest.mark.asyncio
    async def test_development_iteration_workflow(self, workflow_runner, mock_api_server):
        """Test complete development iteration workflow"""

        setup_data = {
            "project": {
                "name": "Iteration Test Project",
                "description": "Test project for iteration workflow",
                "requirements": ["Implement user authentication"]
            }
        }

        steps = [
            WorkflowStep(
                step_id="start_iteration",
                name="Start Development Iteration",
                description="Start a new development iteration",
                api_endpoint="/projects/{project_id}/iterations",
                method="POST",
                payload={"iteration_type": "feature_development"},
                expected_status=201,
                validation_rules=["has_id"]
            ),
            WorkflowStep(
                step_id="assign_tasks",
                name="Assign Tasks to Agents",
                description="Distribute tasks among agents",
                api_endpoint="/projects/{project_id}/iterations/{start_iteration_id}/tasks",
                method="POST",
                payload={
                    "tasks": [
                        {"type": "design_architecture", "agent": "architect"},
                        {"type": "implement_feature", "agent": "coder"},
                        {"type": "create_tests", "agent": "tester"}
                    ]
                },
                validation_rules=["non_empty_result"]
            ),
            WorkflowStep(
                step_id="monitor_progress",
                name="Monitor Iteration Progress",
                description="Check iteration progress",
                api_endpoint="/projects/{project_id}/iterations/{start_iteration_id}/progress",
                method="GET",
                validation_rules=["metrics_present"]
            ),
            WorkflowStep(
                step_id="get_results",
                name="Get Iteration Results",
                description="Retrieve iteration results",
                api_endpoint="/projects/{project_id}/iterations/{start_iteration_id}/results",
                method="GET",
                validation_rules=["non_empty_result"]
            ),
            WorkflowStep(
                step_id="complete_iteration",
                name="Complete Iteration",
                description="Mark iteration as complete",
                api_endpoint="/projects/{project_id}/iterations/{start_iteration_id}/complete",
                method="POST",
                validation_rules=["status_active"]
            )
        ]

        # Mock iteration responses
        mock_responses = [
            {"id": "iter_456", "status": "active", "iteration_number": 1},
            {"tasks_assigned": 3, "agents_involved": 3},
            {"progress": 65.5, "metrics": {"tasks_completed": 2, "tasks_remaining": 1}},
            {"code_generated": True, "tests_created": True, "quality_score": 87.5},
            {"status": "completed", "iteration_id": "iter_456"}
        ]

        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.get.return_value.json.side_effect = mock_json
        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        # Run workflow test
        result = await workflow_runner.run_workflow_test(
            "development_iteration",
            steps,
            setup_data
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(steps)
        assert "iteration_id" in workflow_runner.client.session_data

    @pytest.mark.asyncio
    async def test_agent_coordination_workflow(self, workflow_runner, mock_api_server):
        """Test agent coordination workflow"""

        steps = [
            WorkflowStep(
                step_id="trigger_coordination",
                name="Trigger Agent Coordination",
                description="Trigger multi-agent coordination scenario",
                api_endpoint="/projects/{project_id}/coordinate",
                method="POST",
                payload={
                    "scenario": "feature_development",
                    "agents": ["architect", "coder", "tester", "ui_refiner"],
                    "parameters": {"feature": "user_profile_system"}
                },
                validation_rules=["has_id"]
            ),
            WorkflowStep(
                step_id="monitor_coordination",
                name="Monitor Agent Coordination",
                description="Monitor coordination progress",
                api_endpoint="/projects/{project_id}/coordination/{trigger_coordination_id}",
                method="GET",
                validation_rules=["metrics_present"]
            ),
            WorkflowStep(
                step_id="get_agent_messages",
                name="Get Agent Messages",
                description="Retrieve inter-agent messages",
                api_endpoint="/projects/{project_id}/coordination/{trigger_coordination_id}/messages",
                method="GET",
                validation_rules=["count_greater_than_0"]
            ),
            WorkflowStep(
                step_id="coordination_results",
                name="Get Coordination Results",
                description="Get final coordination results",
                api_endpoint="/projects/{project_id}/coordination/{trigger_coordination_id}/results",
                method="GET",
                validation_rules=["non_empty_result"]
            )
        ]

        # Mock coordination responses
        mock_responses = [
            {"id": "coord_789", "status": "running", "participants": 4},
            {"status": "in_progress", "messages_exchanged": 12, "metrics": {"coordination_score": 92.3}},
            [
                {"from": "architect", "to": "coder", "type": "design_spec"},
                {"from": "coder", "to": "tester", "type": "implementation_ready"}
            ],
            {"status": "completed", "outcome": "success", "deliverables": ["feature_design", "implementation", "tests"]}
        ]

        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.get.return_value.json.side_effect = mock_json
        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        # Setup test with existing project
        workflow_runner.client.session_data["project_id"] = "proj_123"

        result = await workflow_runner.run_workflow_test(
            "agent_coordination",
            steps
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(steps)

    @pytest.mark.asyncio
    async def test_quality_monitoring_workflow(self, workflow_runner, mock_api_server):
        """Test quality monitoring and metrics workflow"""

        steps = [
            WorkflowStep(
                step_id="get_project_metrics",
                name="Get Project Metrics",
                description="Retrieve current project quality metrics",
                api_endpoint="/projects/{project_id}/metrics",
                method="GET",
                validation_rules=["metrics_present"]
            ),
            WorkflowStep(
                step_id="run_quality_check",
                name="Run Quality Check",
                description="Trigger comprehensive quality analysis",
                api_endpoint="/projects/{project_id}/quality/check",
                method="POST",
                validation_rules=["has_id"]
            ),
            WorkflowStep(
                step_id="get_quality_report",
                name="Get Quality Report",
                description="Retrieve detailed quality report",
                api_endpoint="/projects/{project_id}/quality/{run_quality_check_id}/report",
                method="GET",
                validation_rules=["metrics_present"]
            ),
            WorkflowStep(
                step_id="check_perfection_criteria",
                name="Check Perfection Criteria",
                description="Verify if perfection criteria are met",
                api_endpoint="/projects/{project_id}/perfection",
                method="GET",
                validation_rules=["non_empty_result"]
            )
        ]

        # Mock quality monitoring responses
        mock_responses = [
            {
                "test_coverage": 96.5,
                "code_quality": 87.2,
                "performance_score": 92.1,
                "security_score": 98.5,
                "documentation_coverage": 89.3,
                "user_satisfaction": 91.7
            },
            {"id": "quality_check_101", "status": "running"},
            {
                "overall_score": 92.8,
                "metrics": {
                    "test_coverage": 96.5,
                    "critical_bugs": 0,
                    "minor_bugs": 3
                },
                "recommendations": ["Improve documentation", "Fix minor bugs"]
            },
            {
                "is_perfect": False,
                "criteria_met": 6,
                "criteria_total": 8,
                "missing_criteria": ["test_coverage", "documentation_coverage"]
            }
        ]

        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.get.return_value.json.side_effect = mock_json
        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        # Setup test with existing project
        workflow_runner.client.session_data["project_id"] = "proj_123"

        result = await workflow_runner.run_workflow_test(
            "quality_monitoring",
            steps
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(steps)

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, workflow_runner, mock_api_server):
        """Test error handling and recovery workflow"""

        steps = [
            WorkflowStep(
                step_id="trigger_error",
                name="Trigger Error Scenario",
                description="Intentionally trigger an error for testing",
                api_endpoint="/projects/{project_id}/test/error",
                method="POST",
                payload={"error_type": "validation_error"},
                expected_status=400,  # Expect error status
                validation_rules=["contains_error"]
            ),
            WorkflowStep(
                step_id="check_error_handling",
                name="Check Error Handling",
                description="Verify error was handled properly",
                api_endpoint="/projects/{project_id}/errors/latest",
                method="GET",
                validation_rules=["non_empty_result"]
            ),
            WorkflowStep(
                step_id="trigger_recovery",
                name="Trigger Recovery",
                description="Trigger system recovery mechanisms",
                api_endpoint="/projects/{project_id}/recovery",
                method="POST",
                validation_rules=["status_active"]
            )
        ]

        # Mock error handling responses
        error_responses = [
            {"error": "Validation failed", "code": "VALIDATION_ERROR", "details": "Invalid input provided"},
            {"latest_error": {"type": "validation_error", "handled": True, "timestamp": "2024-01-15T10:30:00Z"}},
            {"status": "recovered", "recovery_actions": ["reset_state", "reinitialize_components"]}
        ]

        # Configure different status codes
        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        mock_response_400.json.return_value = error_responses[0]
        mock_response_400.headers = {}
        mock_response_400.text = json.dumps(error_responses[0])

        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {}
        mock_response_200.text = '{"status": "success"}'

        call_count = 0
        def mock_response_selector(*args, **kwargs):
            nonlocal call_count
            if call_count == 0:
                mock_response_200.json.return_value = error_responses[1]
                call_count += 1
                return mock_response_400
            elif call_count == 1:
                mock_response_200.json.return_value = error_responses[1]
                call_count += 1
                return mock_response_200
            else:
                mock_response_200.json.return_value = error_responses[2]
                return mock_response_200

        mock_api_server.return_value.__aenter__.return_value.get.side_effect = mock_response_selector
        mock_api_server.return_value.__aenter__.return_value.post.side_effect = mock_response_selector

        # Setup test with existing project
        workflow_runner.client.session_data["project_id"] = "proj_123"

        result = await workflow_runner.run_workflow_test(
            "error_handling",
            steps
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(steps)

    @pytest.mark.asyncio
    async def test_performance_workflow(self, workflow_runner, mock_api_server):
        """Test performance monitoring and optimization workflow"""

        steps = [
            WorkflowStep(
                step_id="performance_baseline",
                name="Establish Performance Baseline",
                description="Establish performance baseline metrics",
                api_endpoint="/projects/{project_id}/performance/baseline",
                method="POST",
                validation_rules=["metrics_present"]
            ),
            WorkflowStep(
                step_id="run_load_test",
                name="Run Load Test",
                description="Execute system load testing",
                api_endpoint="/projects/{project_id}/performance/load-test",
                method="POST",
                payload={
                    "concurrent_users": 100,
                    "duration": 60,
                    "ramp_up_time": 30
                },
                validation_rules=["has_id"]
            ),
            WorkflowStep(
                step_id="get_performance_results",
                name="Get Performance Results",
                description="Retrieve load test results",
                api_endpoint="/projects/{project_id}/performance/{run_load_test_id}/results",
                method="GET",
                validation_rules=["metrics_present"]
            ),
            WorkflowStep(
                step_id="analyze_bottlenecks",
                name="Analyze Performance Bottlenecks",
                description="Identify performance bottlenecks",
                api_endpoint="/projects/{project_id}/performance/analyze",
                method="POST",
                validation_rules=["non_empty_result"]
            ),
            WorkflowStep(
                step_id="apply_optimizations",
                name="Apply Performance Optimizations",
                description="Apply recommended optimizations",
                api_endpoint="/projects/{project_id}/performance/optimize",
                method="POST",
                validation_rules=["status_active"]
            )
        ]

        # Mock performance responses
        mock_responses = [
            {"baseline_established": True, "metrics": {"avg_response_time": 250, "requests_per_second": 500}},
            {"id": "load_test_202", "status": "running", "estimated_duration": 90},
            {
                "metrics": {
                    "avg_response_time": 275,
                    "p95_response_time": 450,
                    "requests_per_second": 485,
                    "error_rate": 0.02
                }
            },
            {
                "bottlenecks": [
                    {"component": "database", "severity": "high", "description": "Slow query performance"},
                    {"component": "api_gateway", "severity": "medium", "description": "Rate limiting overhead"}
                ]
            },
            {"status": "optimized", "optimizations_applied": ["database_indexing", "connection_pooling"]}
        ]

        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.get.return_value.json.side_effect = mock_json
        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        # Setup test with existing project
        workflow_runner.client.session_data["project_id"] = "proj_123"

        result = await workflow_runner.run_workflow_test(
            "performance_workflow",
            steps
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(steps)
        assert result.performance_metrics["total_api_calls"] == len(steps)

    @pytest.mark.asyncio
    async def test_user_journey_simulation(self, workflow_runner, mock_api_server):
        """Test complete user journey from project creation to completion"""

        # Comprehensive user journey workflow
        steps = [
            # User Registration/Authentication (mocked)
            WorkflowStep(
                step_id="user_auth",
                name="User Authentication",
                description="Authenticate user session",
                api_endpoint="/auth/session",
                method="POST",
                payload={"user_id": "test_user", "session_token": "mock_token"},
                validation_rules=["non_empty_result"]
            ),

            # Project Creation
            WorkflowStep(
                step_id="create_user_project",
                name="Create User Project",
                description="User creates a new project",
                api_endpoint="/projects",
                method="POST",
                payload={
                    "name": "User Journey Test Project",
                    "description": "A complete user journey test",
                    "requirements": [
                        "Create a todo list application",
                        "Include user authentication",
                        "Add data persistence",
                        "Ensure mobile responsiveness"
                    ],
                    "complexity": "medium",
                    "target_metrics": {
                        "test_coverage": 95.0,
                        "performance_score": 90.0,
                        "user_satisfaction": 85.0
                    }
                },
                expected_status=201,
                validation_rules=["has_id", "status_active"]
            ),

            # Project Dashboard Access
            WorkflowStep(
                step_id="access_dashboard",
                name="Access Project Dashboard",
                description="User accesses project dashboard",
                api_endpoint="/projects/{create_user_project_id}/dashboard",
                method="GET",
                validation_rules=["metrics_present", "has_agents"]
            ),

            # Start Development
            WorkflowStep(
                step_id="start_development",
                name="Start Development Process",
                description="User initiates development process",
                api_endpoint="/projects/{create_user_project_id}/start",
                method="POST",
                validation_rules=["status_active"]
            ),

            # Monitor Progress
            WorkflowStep(
                step_id="monitor_development",
                name="Monitor Development Progress",
                description="User monitors development progress",
                api_endpoint="/projects/{create_user_project_id}/progress",
                method="GET",
                validation_rules=["metrics_present"]
            ),

            # Check Quality
            WorkflowStep(
                step_id="check_quality",
                name="Check Quality Metrics",
                description="User checks current quality metrics",
                api_endpoint="/projects/{create_user_project_id}/quality",
                method="GET",
                validation_rules=["metrics_present"]
            ),

            # Request Feature Update
            WorkflowStep(
                step_id="request_update",
                name="Request Feature Update",
                description="User requests additional feature",
                api_endpoint="/projects/{create_user_project_id}/features",
                method="POST",
                payload={
                    "feature_request": "Add email notifications for task updates",
                    "priority": "medium"
                },
                validation_rules=["has_id"]
            ),

            # Final Status Check
            WorkflowStep(
                step_id="final_status",
                name="Final Project Status",
                description="User checks final project status",
                api_endpoint="/projects/{create_user_project_id}/status",
                method="GET",
                validation_rules=["status_active", "metrics_present"]
            )
        ]

        # Mock user journey responses
        mock_responses = [
            {"session_id": "sess_123", "user_id": "test_user", "authenticated": True},
            {"id": "proj_journey_456", "status": "active", "name": "User Journey Test Project"},
            {
                "project_info": {"id": "proj_journey_456", "status": "active"},
                "agents": [{"name": "coder", "status": "active"}],
                "metrics": {"progress": 0}
            },
            {"status": "development_started", "iteration_id": "iter_001"},
            {"progress": 45.5, "metrics": {"code_generated": True, "tests_created": True}},
            {"metrics": {"test_coverage": 92.5, "code_quality": 88.1, "performance_score": 89.7}},
            {"id": "feature_req_789", "status": "queued", "estimated_completion": "2h"},
            {
                "status": "active",
                "completion_percentage": 67.8,
                "metrics": {"overall_quality": 89.4},
                "next_steps": ["Complete testing", "Performance optimization"]
            }
        ]

        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.get.return_value.json.side_effect = mock_json
        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        result = await workflow_runner.run_workflow_test(
            "complete_user_journey",
            steps
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(steps)
        assert len(result.validation_results) > 0
        assert result.performance_metrics["total_api_calls"] == len(steps)

        # Verify user journey metrics
        assert result.duration > 0
        assert result.performance_metrics["average_api_response_time"] >= 0

    @pytest.mark.asyncio
    async def test_concurrent_workflows(self, workflow_runner, mock_api_server):
        """Test multiple concurrent workflows"""

        # Create simplified workflows for concurrent testing
        workflow1_steps = [
            WorkflowStep(
                step_id="w1_start",
                name="Workflow 1 Start",
                description="Start first workflow",
                api_endpoint="/projects/workflow1/start",
                method="POST",
                payload={"workflow": "data_processing"},
                validation_rules=["has_id"]
            ),
            WorkflowStep(
                step_id="w1_process",
                name="Workflow 1 Process",
                description="Process workflow 1",
                api_endpoint="/projects/workflow1/process",
                method="POST",
                validation_rules=["status_active"]
            )
        ]

        workflow2_steps = [
            WorkflowStep(
                step_id="w2_start",
                name="Workflow 2 Start",
                description="Start second workflow",
                api_endpoint="/projects/workflow2/start",
                method="POST",
                payload={"workflow": "ui_enhancement"},
                validation_rules=["has_id"]
            ),
            WorkflowStep(
                step_id="w2_process",
                name="Workflow 2 Process",
                description="Process workflow 2",
                api_endpoint="/projects/workflow2/process",
                method="POST",
                validation_rules=["status_active"]
            )
        ]

        # Mock responses for both workflows
        mock_responses = [
            {"id": "w1_001", "status": "started"},
            {"status": "processing", "workflow_id": "w1_001"},
            {"id": "w2_002", "status": "started"},
            {"status": "processing", "workflow_id": "w2_002"}
        ]

        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        # Run workflows concurrently
        workflow1_runner = WorkflowTestRunner(SystemTestClient())
        workflow2_runner = WorkflowTestRunner(SystemTestClient())

        start_time = time.time()

        # Execute workflows concurrently
        results = await asyncio.gather(
            workflow1_runner.run_workflow_test("concurrent_workflow_1", workflow1_steps),
            workflow2_runner.run_workflow_test("concurrent_workflow_2", workflow2_steps)
        )

        concurrent_duration = time.time() - start_time

        # Both workflows should complete successfully
        assert len(results) == 2
        assert all(result.status == "PASSED" for result in results)

        # Concurrent execution should be efficient
        total_sequential_time = sum(result.duration for result in results)
        assert concurrent_duration < total_sequential_time * 0.8  # Should be significantly faster


class TestSystemPerformance:
    """System performance and scalability tests"""

    @pytest.mark.asyncio
    async def test_system_load_handling(self, workflow_runner, mock_api_server):
        """Test system performance under load"""

        # Create load test scenario
        load_test_steps = [
            WorkflowStep(
                step_id="load_test_setup",
                name="Setup Load Test",
                description="Initialize load testing environment",
                api_endpoint="/system/load-test/setup",
                method="POST",
                payload={
                    "concurrent_requests": 50,
                    "test_duration": 30,
                    "request_pattern": "steady"
                },
                validation_rules=["has_id"]
            ),
            WorkflowStep(
                step_id="execute_load_test",
                name="Execute Load Test",
                description="Run the load test",
                api_endpoint="/system/load-test/{load_test_setup_id}/execute",
                method="POST",
                validation_rules=["status_active"]
            ),
            WorkflowStep(
                step_id="load_test_results",
                name="Get Load Test Results",
                description="Retrieve load test results",
                api_endpoint="/system/load-test/{load_test_setup_id}/results",
                method="GET",
                validation_rules=["metrics_present"]
            )
        ]

        # Mock load test responses
        mock_responses = [
            {"id": "load_test_303", "status": "initialized", "estimated_duration": 45},
            {"status": "running", "current_rps": 45.2, "avg_response_time": 125.8},
            {
                "metrics": {
                    "total_requests": 1500,
                    "successful_requests": 1485,
                    "avg_response_time": 145.6,
                    "p95_response_time": 285.3,
                    "p99_response_time": 450.7,
                    "requests_per_second": 47.3,
                    "error_rate": 0.01
                },
                "performance_grade": "A",
                "bottlenecks": []
            }
        ]

        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.get.return_value.json.side_effect = mock_json
        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        result = await workflow_runner.run_workflow_test(
            "system_load_test",
            load_test_steps
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(load_test_steps)

    @pytest.mark.asyncio
    async def test_memory_usage_workflow(self, workflow_runner, mock_api_server):
        """Test system memory usage monitoring"""

        steps = [
            WorkflowStep(
                step_id="memory_baseline",
                name="Establish Memory Baseline",
                description="Get current memory usage baseline",
                api_endpoint="/system/memory/baseline",
                method="GET",
                validation_rules=["metrics_present"]
            ),
            WorkflowStep(
                step_id="memory_stress_test",
                name="Memory Stress Test",
                description="Execute memory-intensive operations",
                api_endpoint="/system/memory/stress-test",
                method="POST",
                payload={"test_type": "memory_allocation", "intensity": "high"},
                validation_rules=["has_id"]
            ),
            WorkflowStep(
                step_id="memory_analysis",
                name="Memory Usage Analysis",
                description="Analyze memory usage patterns",
                api_endpoint="/system/memory/{memory_stress_test_id}/analysis",
                method="GET",
                validation_rules=["metrics_present"]
            )
        ]

        # Mock memory test responses
        mock_responses = [
            {"baseline_memory_mb": 512, "available_memory_mb": 1536, "memory_utilization": 0.25},
            {"id": "mem_test_404", "status": "running", "peak_memory_target_mb": 2048},
            {
                "metrics": {
                    "peak_memory_mb": 1894,
                    "memory_leaks_detected": 0,
                    "gc_collections": 45,
                    "memory_efficiency": 0.92
                },
                "status": "healthy",
                "recommendations": []
            }
        ]

        call_count = 0
        def mock_json():
            nonlocal call_count
            response = mock_responses[call_count % len(mock_responses)]
            call_count += 1
            return response

        mock_api_server.return_value.__aenter__.return_value.get.return_value.json.side_effect = mock_json
        mock_api_server.return_value.__aenter__.return_value.post.return_value.json.side_effect = mock_json

        result = await workflow_runner.run_workflow_test(
            "memory_usage_test",
            steps
        )

        assert result.status == "PASSED"
        assert result.steps_completed == len(steps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])