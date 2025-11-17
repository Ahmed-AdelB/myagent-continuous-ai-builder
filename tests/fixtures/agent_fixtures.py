"""
Agent-specific test fixtures and mock data
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock


@dataclass
class AgentTestData:
    """Container for agent test data"""
    name: str
    role: str
    capabilities: List[str]
    mock_responses: Dict[str, str]
    expected_outputs: Dict[str, Any]


def create_mock_agent_response(agent_type: str, task_type: str) -> str:
    """Create mock LLM responses for different agent types and tasks"""
    responses = {
        'coder': {
            'generate_function': '''def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)''',
            'refactor_code': '''def fibonacci(n):
    """Calculate the nth Fibonacci number using memoization."""
    if n <= 1:
        return n

    memo = {}
    def fib(x):
        if x in memo:
            return memo[x]
        memo[x] = fib(x-1) + fib(x-2)
        return memo[x]

    return fib(n)''',
            'fix_bug': '''def safe_divide(a, b):
    """Safely divide two numbers."""
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b''',
            'generate_class': '''class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result'''
        },
        'tester': {
            'generate_unit_tests': '''import pytest
from calculator import Calculator

class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5
        assert calc.add(-1, 1) == 0

    def test_multiply(self):
        calc = Calculator()
        assert calc.multiply(2, 3) == 6
        assert calc.multiply(-2, 3) == -6

    def test_history(self):
        calc = Calculator()
        calc.add(1, 2)
        calc.multiply(3, 4)
        assert len(calc.history) == 2''',
            'generate_integration_tests': '''import pytest
import asyncio
from api.main import app
from fastapi.testclient import TestClient

@pytest.mark.integration
class TestProjectAPI:
    def setup_method(self):
        self.client = TestClient(app)

    def test_create_project(self):
        project_data = {
            "name": "Test Project",
            "description": "Test description",
            "requirements": ["req1", "req2"]
        }
        response = self.client.post("/api/projects", json=project_data)
        assert response.status_code == 201
        assert response.json()["name"] == "Test Project"

    def test_get_project(self):
        # Create project first
        project_data = {"name": "Test", "description": "Test"}
        create_response = self.client.post("/api/projects", json=project_data)
        project_id = create_response.json()["id"]

        # Get project
        response = self.client.get(f"/api/projects/{project_id}")
        assert response.status_code == 200''',
            'generate_performance_tests': '''import pytest
import time
from memory_profiler import profile

@pytest.mark.performance
class TestPerformance:
    def test_fibonacci_performance(self):
        start_time = time.time()
        result = fibonacci(30)
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Execution too slow: {execution_time}s"
        assert result == 832040

    @profile
    def test_memory_usage(self):
        large_list = [i for i in range(100000)]
        processed = [x * 2 for x in large_list]
        return len(processed)'''
        },
        'debugger': {
            'analyze_error': '''Error Analysis Report:

Error Type: ZeroDivisionError
Location: line 15 in calculate_average()
Root Cause: Division by zero when list is empty

Suggested Fix:
1. Add validation for empty list
2. Return None or raise appropriate exception

def calculate_average(numbers):
    if not numbers:
        return None  # or raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)''',
            'fix_performance_issue': '''Performance Issue Analysis:

Problem: O(n²) complexity in nested loop
Location: data_processor.py, lines 25-30

Optimization:
1. Replace nested loop with dictionary lookup
2. Pre-compute values where possible

Before:
for item in items:
    for category in categories:
        if item.category == category.name:
            process(item, category)

After:
category_map = {cat.name: cat for cat in categories}
for item in items:
    if item.category in category_map:
        process(item, category_map[item.category])''',
            'security_vulnerability': '''Security Vulnerability Report:

Type: SQL Injection
Severity: HIGH
Location: user_service.py, line 45

Vulnerable Code:
query = f"SELECT * FROM users WHERE id = {user_id}"

Fix:
Use parameterized queries:
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))'''
        },
        'architect': {
            'design_review': '''Architecture Review Report:

Strengths:
- Clear separation of concerns
- Modular design with well-defined interfaces
- Proper use of dependency injection

Recommendations:
1. Implement Observer pattern for event handling
2. Add caching layer for frequently accessed data
3. Consider using Factory pattern for object creation
4. Implement circuit breaker for external API calls

Design Pattern Suggestions:
- Strategy pattern for different algorithm implementations
- Command pattern for undo/redo functionality
- Repository pattern for data access layer''',
            'scalability_analysis': '''Scalability Analysis:

Current Bottlenecks:
1. Database queries not optimized
2. No caching mechanism
3. Synchronous processing of large datasets

Scaling Solutions:
1. Implement horizontal scaling with load balancers
2. Add Redis cache for session management
3. Use async processing for I/O operations
4. Database sharding for large datasets
5. CDN for static assets

Performance Targets:
- Handle 10K concurrent users
- Response time < 200ms for 95% of requests
- 99.9% uptime SLA''',
            'technology_recommendations': '''Technology Stack Recommendations:

Frontend:
- React 18 with TypeScript for type safety
- Redux Toolkit for state management
- Material-UI for consistent design

Backend:
- FastAPI for high-performance APIs
- PostgreSQL for relational data
- Redis for caching and sessions
- Celery for background tasks

Infrastructure:
- Docker for containerization
- Kubernetes for orchestration
- Prometheus for monitoring
- ELK stack for logging'''
        },
        'analyzer': {
            'performance_metrics': '''Performance Metrics Report:

Current Metrics:
- CPU Usage: 65% average
- Memory Usage: 2.3GB (78% of available)
- Response Time: 245ms average
- Throughput: 1,200 requests/minute
- Error Rate: 0.3%

Trend Analysis:
- Response time increased 15% over last week
- Memory usage stable
- CPU spikes during peak hours

Recommendations:
1. Optimize database queries (3 slow queries identified)
2. Implement caching for frequently accessed data
3. Scale horizontally during peak hours
4. Monitor memory leaks in background processes''',
            'code_quality_analysis': '''Code Quality Analysis:

Metrics:
- Test Coverage: 87%
- Cyclomatic Complexity: 8.5 average
- Code Duplication: 12%
- Technical Debt: 2.3 hours

Issues Found:
1. 15 functions exceed complexity threshold (>10)
2. 8 files have duplicate code blocks
3. 23 missing docstrings
4. 5 unused imports

Quality Score: B+ (85/100)

Priority Fixes:
1. Refactor complex functions in data_processor.py
2. Extract common logic to utility functions
3. Add missing documentation
4. Remove unused code''',
            'security_analysis': '''Security Analysis Report:

Vulnerabilities Found:
- 0 Critical
- 2 High
- 5 Medium
- 12 Low

High Priority Issues:
1. Unvalidated user input in file upload (HIGH)
2. Weak password policy enforcement (HIGH)

Medium Priority Issues:
- Missing HTTPS enforcement
- Inadequate session timeout
- Insufficient logging for security events
- Missing rate limiting
- Weak CORS configuration

Recommendations:
1. Implement input validation middleware
2. Enforce strong password requirements
3. Add comprehensive security headers
4. Implement rate limiting
5. Regular security audits'''
        },
        'ui_refiner': {
            'accessibility_improvements': '''Accessibility Improvements:

WCAG Compliance Issues:
1. Missing alt text for 8 images
2. Low color contrast in 3 components
3. No keyboard navigation for dropdown menus
4. Missing ARIA labels for form controls

Improvements:
1. Add descriptive alt text to all images
2. Increase contrast ratios to meet AA standards
3. Implement proper focus management
4. Add ARIA labels and landmarks

<button aria-label="Close dialog" onClick={closeModal}>
  <span aria-hidden="true">&times;</span>
</button>

<img src="chart.png" alt="Sales performance chart showing 25% increase over Q3" />''',
            'user_experience_optimization': '''UX Optimization Report:

Current Issues:
1. Long loading times on dashboard
2. Confusing navigation structure
3. No loading indicators for async operations
4. Inconsistent button styles

Improvements:
1. Add skeleton loading screens
2. Implement breadcrumb navigation
3. Show progress indicators for uploads
4. Create design system for consistency

User Feedback:
- 78% find navigation confusing
- 65% want faster page loads
- 82% appreciate real-time updates
- 71% need better error messages

Priority Changes:
1. Implement loading states
2. Redesign main navigation
3. Add helpful error messages
4. Create user onboarding flow''',
            'responsive_design': '''Responsive Design Improvements:

Breakpoints Analysis:
- Mobile (320-768px): 45% of users
- Tablet (768-1024px): 22% of users
- Desktop (1024px+): 33% of users

Issues Found:
1. Tables not scrollable on mobile
2. Modal dialogs too large for small screens
3. Touch targets smaller than 44px
4. Text not readable on mobile

Solutions:
1. Implement horizontal scrolling for tables
2. Create mobile-friendly modal layouts
3. Increase button and link sizes
4. Optimize font sizes and line height

CSS Updates:
@media (max-width: 768px) {
  .table-container {
    overflow-x: auto;
  }

  .button {
    min-height: 44px;
    padding: 12px 16px;
  }
}'''
        }
    }

    return responses.get(agent_type, {}).get(task_type, "Mock response")


# Agent test data configurations
CODER_TEST_DATA = AgentTestData(
    name="coder_agent",
    role="Software Engineer",
    capabilities=["code_generation", "refactoring", "documentation", "bug_fixing"],
    mock_responses={
        "generate_function": create_mock_agent_response("coder", "generate_function"),
        "refactor_code": create_mock_agent_response("coder", "refactor_code"),
        "fix_bug": create_mock_agent_response("coder", "fix_bug")
    },
    expected_outputs={
        "function_signature": "def fibonacci(n):",
        "return_statement": "return",
        "docstring": '"""Calculate the nth Fibonacci number."""'
    }
)

TESTER_TEST_DATA = AgentTestData(
    name="tester_agent",
    role="QA Engineer",
    capabilities=["test_generation", "test_execution", "coverage_analysis"],
    mock_responses={
        "unit_tests": create_mock_agent_response("tester", "generate_unit_tests"),
        "integration_tests": create_mock_agent_response("tester", "generate_integration_tests"),
        "performance_tests": create_mock_agent_response("tester", "generate_performance_tests")
    },
    expected_outputs={
        "test_class": "class Test",
        "test_method": "def test_",
        "assertion": "assert"
    }
)

DEBUGGER_TEST_DATA = AgentTestData(
    name="debugger_agent",
    role="Debug Specialist",
    capabilities=["error_analysis", "performance_debugging", "security_analysis"],
    mock_responses={
        "error_analysis": create_mock_agent_response("debugger", "analyze_error"),
        "performance_fix": create_mock_agent_response("debugger", "fix_performance_issue"),
        "security_fix": create_mock_agent_response("debugger", "security_vulnerability")
    },
    expected_outputs={
        "error_type": "Error Type:",
        "root_cause": "Root Cause:",
        "suggested_fix": "Suggested Fix:"
    }
)

ARCHITECT_TEST_DATA = AgentTestData(
    name="architect_agent",
    role="System Architect",
    capabilities=["design_review", "scalability_analysis", "technology_selection"],
    mock_responses={
        "design_review": create_mock_agent_response("architect", "design_review"),
        "scalability": create_mock_agent_response("architect", "scalability_analysis"),
        "tech_stack": create_mock_agent_response("architect", "technology_recommendations")
    },
    expected_outputs={
        "recommendations": "Recommendations:",
        "design_patterns": "Strategy pattern",
        "scalability": "Scaling Solutions:"
    }
)

ANALYZER_TEST_DATA = AgentTestData(
    name="analyzer_agent",
    role="Performance Analyst",
    capabilities=["performance_monitoring", "code_quality_analysis", "security_scanning"],
    mock_responses={
        "performance": create_mock_agent_response("analyzer", "performance_metrics"),
        "code_quality": create_mock_agent_response("analyzer", "code_quality_analysis"),
        "security": create_mock_agent_response("analyzer", "security_analysis")
    },
    expected_outputs={
        "metrics": "Current Metrics:",
        "quality_score": "Quality Score:",
        "vulnerabilities": "Vulnerabilities Found:"
    }
)

UI_REFINER_TEST_DATA = AgentTestData(
    name="ui_refiner_agent",
    role="UX Designer",
    capabilities=["accessibility_improvements", "user_experience_optimization", "responsive_design"],
    mock_responses={
        "accessibility": create_mock_agent_response("ui_refiner", "accessibility_improvements"),
        "user_experience": create_mock_agent_response("ui_refiner", "user_experience_optimization"),
        "responsive": create_mock_agent_response("ui_refiner", "responsive_design")
    },
    expected_outputs={
        "wcag_compliance": "WCAG Compliance",
        "user_feedback": "User Feedback:",
        "improvements": "Improvements:"
    }
)

# Collection of all agent test data
ALL_AGENT_TEST_DATA = {
    'coder': CODER_TEST_DATA,
    'tester': TESTER_TEST_DATA,
    'debugger': DEBUGGER_TEST_DATA,
    'architect': ARCHITECT_TEST_DATA,
    'analyzer': ANALYZER_TEST_DATA,
    'ui_refiner': UI_REFINER_TEST_DATA
}