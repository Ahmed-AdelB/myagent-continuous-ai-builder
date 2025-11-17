"""
Project-specific test fixtures and configurations
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class ProjectTestData:
    """Container for project test data"""
    project_config: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    expected_milestones: List[Dict[str, Any]]
    code_samples: Dict[str, str]
    test_scenarios: List[Dict[str, Any]]


def create_test_project_config(
    name: str = "Test Project",
    complexity: str = "medium"
) -> Dict[str, Any]:
    """Create test project configurations for different complexities"""

    base_config = {
        "name": name,
        "description": f"A {complexity} complexity test project",
        "created_at": datetime.now().isoformat(),
        "max_iterations": 100
    }

    complexity_configs = {
        "simple": {
            "requirements": [
                "Create a simple calculator function",
                "Add basic error handling",
                "Include unit tests"
            ],
            "target_metrics": {
                "test_coverage": 90.0,
                "code_quality": 80.0,
                "performance_score": 85.0,
                "documentation_coverage": 85.0,
                "security_score": 90.0,
                "user_satisfaction": 85.0,
                "critical_bugs": 0,
                "minor_bugs": 2
            },
            "estimated_hours": 5
        },
        "medium": {
            "requirements": [
                "Create a web application with user authentication",
                "Implement data persistence with database",
                "Add REST API endpoints",
                "Include comprehensive test suite",
                "Implement basic security measures"
            ],
            "target_metrics": {
                "test_coverage": 95.0,
                "code_quality": 85.0,
                "performance_score": 90.0,
                "documentation_coverage": 90.0,
                "security_score": 95.0,
                "user_satisfaction": 90.0,
                "critical_bugs": 0,
                "minor_bugs": 5
            },
            "estimated_hours": 40
        },
        "complex": {
            "requirements": [
                "Build a microservices architecture",
                "Implement event-driven communication",
                "Add monitoring and observability",
                "Create CI/CD pipeline",
                "Implement advanced security features",
                "Add real-time data processing",
                "Include load balancing and scaling"
            ],
            "target_metrics": {
                "test_coverage": 98.0,
                "code_quality": 90.0,
                "performance_score": 95.0,
                "documentation_coverage": 95.0,
                "security_score": 98.0,
                "user_satisfaction": 95.0,
                "critical_bugs": 0,
                "minor_bugs": 3
            },
            "estimated_hours": 200
        }
    }

    return {**base_config, **complexity_configs.get(complexity, complexity_configs["medium"])}


def create_test_task_data(task_type: str = "generate_function") -> Dict[str, Any]:
    """Create test task configurations"""

    task_templates = {
        "generate_function": {
            "type": "generate_function",
            "description": "Create a function to calculate fibonacci numbers",
            "priority": 5,
            "agent": "coder",
            "data": {
                "language": "python",
                "function_name": "fibonacci",
                "parameters": ["n"],
                "return_type": "int",
                "requirements": [
                    "Handle negative numbers appropriately",
                    "Include proper docstring",
                    "Add type hints"
                ]
            },
            "acceptance_criteria": [
                "Function returns correct fibonacci numbers",
                "Handles edge cases (0, 1, negative)",
                "Includes comprehensive docstring",
                "Uses type hints"
            ]
        },
        "create_api_endpoint": {
            "type": "create_api_endpoint",
            "description": "Create a REST API endpoint for user management",
            "priority": 7,
            "agent": "coder",
            "data": {
                "endpoint": "/api/users",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "authentication": True,
                "validation": True,
                "database_model": "User"
            },
            "acceptance_criteria": [
                "Endpoint handles all CRUD operations",
                "Proper authentication and authorization",
                "Input validation and error handling",
                "API documentation"
            ]
        },
        "write_tests": {
            "type": "generate_tests",
            "description": "Create comprehensive test suite for calculator module",
            "priority": 6,
            "agent": "tester",
            "data": {
                "test_types": ["unit", "integration", "edge_cases"],
                "coverage_target": 95,
                "frameworks": ["pytest"],
                "modules_to_test": ["calculator.py"]
            },
            "acceptance_criteria": [
                "Achieves target coverage",
                "Tests all edge cases",
                "Includes performance tests",
                "Follows testing best practices"
            ]
        },
        "fix_bug": {
            "type": "fix_bug",
            "description": "Fix division by zero error in calculator",
            "priority": 9,
            "agent": "debugger",
            "data": {
                "error_type": "ZeroDivisionError",
                "file": "calculator.py",
                "line": 25,
                "stack_trace": "Traceback...",
                "reproduction_steps": [
                    "Call divide(10, 0)",
                    "Error occurs"
                ]
            },
            "acceptance_criteria": [
                "Error is properly handled",
                "Appropriate exception is raised or default value returned",
                "Tests added to prevent regression",
                "Documentation updated"
            ]
        },
        "optimize_performance": {
            "type": "optimize_performance",
            "description": "Improve database query performance",
            "priority": 6,
            "agent": "analyzer",
            "data": {
                "performance_issue": "Slow database queries",
                "current_response_time": "2.5s",
                "target_response_time": "0.5s",
                "affected_endpoints": ["/api/users", "/api/projects"]
            },
            "acceptance_criteria": [
                "Response time meets target",
                "Database queries optimized",
                "Caching implemented where appropriate",
                "Performance monitoring added"
            ]
        },
        "improve_ui": {
            "type": "improve_accessibility",
            "description": "Improve UI accessibility and user experience",
            "priority": 5,
            "agent": "ui_refiner",
            "data": {
                "accessibility_issues": [
                    "Missing alt text",
                    "Low color contrast",
                    "No keyboard navigation"
                ],
                "target_components": ["LoginForm", "Dashboard", "UserProfile"]
            },
            "acceptance_criteria": [
                "WCAG 2.1 AA compliance",
                "Improved color contrast ratios",
                "Full keyboard navigation",
                "Screen reader compatibility"
            ]
        }
    }

    return task_templates.get(task_type, task_templates["generate_function"])


def create_test_code_snippets() -> Dict[str, str]:
    """Create various code snippets for testing"""

    return {
        "simple_function": '''
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b
''',
        "buggy_function": '''
def divide_numbers(a, b):
    """Divide two numbers."""
    return a / b  # Bug: no zero division check
''',
        "complex_class": '''
class DataProcessor:
    """Process data with various operations."""

    def __init__(self):
        self.data = []
        self.processed = False

    def load_data(self, source):
        """Load data from source."""
        if not source:
            raise ValueError("Source cannot be empty")
        # Load data logic here
        self.data = source

    def process_data(self):
        """Process loaded data."""
        if not self.data:
            return []

        processed_data = []
        for item in self.data:
            # Complex processing logic
            processed_item = self._transform_item(item)
            if self._validate_item(processed_item):
                processed_data.append(processed_item)

        self.processed = True
        return processed_data

    def _transform_item(self, item):
        """Transform a single item."""
        # Transformation logic
        return item.upper() if isinstance(item, str) else str(item)

    def _validate_item(self, item):
        """Validate a transformed item."""
        return len(item) > 0
''',
        "api_endpoint": '''
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from .models import User, UserCreate
from .database import get_db

router = APIRouter()

@router.get("/users", response_model=List[User])
async def get_users(db: Session = Depends(get_db)):
    """Get all users."""
    users = db.query(User).all()
    return users

@router.post("/users", response_model=User)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user."""
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get a specific user."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
''',
        "test_suite": '''
import pytest
from unittest.mock import Mock, patch
from calculator import Calculator

class TestCalculator:
    """Test suite for Calculator class."""

    def setup_method(self):
        """Setup for each test."""
        self.calculator = Calculator()

    def test_add_positive_numbers(self):
        """Test adding positive numbers."""
        result = self.calculator.add(2, 3)
        assert result == 5

    def test_add_negative_numbers(self):
        """Test adding negative numbers."""
        result = self.calculator.add(-2, -3)
        assert result == -5

    def test_divide_by_zero_raises_error(self):
        """Test that division by zero raises appropriate error."""
        with pytest.raises(ValueError, match="Division by zero"):
            self.calculator.divide(10, 0)

    @pytest.mark.parametrize("a,b,expected", [
        (10, 2, 5),
        (15, 3, 5),
        (-10, 2, -5),
        (0, 5, 0)
    ])
    def test_divide_parametrized(self, a, b, expected):
        """Test division with multiple parameter sets."""
        result = self.calculator.divide(a, b)
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_calculation(self):
        """Test asynchronous calculation."""
        result = await self.calculator.async_add(2, 3)
        assert result == 5
''',
        "frontend_component": '''
import React, { useState, useEffect } from 'react';
import { Button, TextField, Alert } from '@mui/material';

interface User {
  id: number;
  name: string;
  email: string;
}

const UserForm: React.FC = () => {
  const [user, setUser] = useState<User>({ id: 0, name: '', email: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(user)
      });

      if (!response.ok) {
        throw new Error('Failed to create user');
      }

      // Success handling
      setUser({ id: 0, name: '', email: '' });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <TextField
        label="Name"
        value={user.name}
        onChange={(e) => setUser({ ...user, name: e.target.value })}
        required
        fullWidth
        margin="normal"
      />
      <TextField
        label="Email"
        type="email"
        value={user.email}
        onChange={(e) => setUser({ ...user, email: e.target.value })}
        required
        fullWidth
        margin="normal"
      />
      {error && <Alert severity="error">{error}</Alert>}
      <Button
        type="submit"
        variant="contained"
        disabled={loading}
        fullWidth
      >
        {loading ? 'Creating...' : 'Create User'}
      </Button>
    </form>
  );
};

export default UserForm;
'''
    }


# Pre-defined project configurations
SIMPLE_PROJECT = ProjectTestData(
    project_config=create_test_project_config("Simple Calculator", "simple"),
    tasks=[
        create_test_task_data("generate_function"),
        create_test_task_data("write_tests"),
        create_test_task_data("fix_bug")
    ],
    expected_milestones=[
        {
            "name": "Function Implementation",
            "description": "Basic calculator function created",
            "completion_criteria": ["Function implemented", "Basic tests passing"],
            "estimated_completion": datetime.now() + timedelta(hours=2)
        },
        {
            "name": "Testing Complete",
            "description": "Comprehensive test suite implemented",
            "completion_criteria": ["95% test coverage", "All edge cases tested"],
            "estimated_completion": datetime.now() + timedelta(hours=4)
        }
    ],
    code_samples=create_test_code_snippets(),
    test_scenarios=[
        {
            "name": "Basic Functionality",
            "steps": ["Create function", "Test basic cases", "Handle errors"],
            "expected_outcome": "Function works correctly"
        }
    ]
)

MEDIUM_PROJECT = ProjectTestData(
    project_config=create_test_project_config("Web Application", "medium"),
    tasks=[
        create_test_task_data("create_api_endpoint"),
        create_test_task_data("write_tests"),
        create_test_task_data("optimize_performance"),
        create_test_task_data("improve_ui")
    ],
    expected_milestones=[
        {
            "name": "API Development",
            "description": "REST API endpoints implemented",
            "completion_criteria": ["All CRUD operations", "Authentication", "Validation"],
            "estimated_completion": datetime.now() + timedelta(hours=15)
        },
        {
            "name": "UI Implementation",
            "description": "User interface completed",
            "completion_criteria": ["Responsive design", "Accessibility", "User testing"],
            "estimated_completion": datetime.now() + timedelta(hours=25)
        },
        {
            "name": "Production Ready",
            "description": "Application ready for deployment",
            "completion_criteria": ["All tests passing", "Performance optimized", "Security reviewed"],
            "estimated_completion": datetime.now() + timedelta(hours=40)
        }
    ],
    code_samples=create_test_code_snippets(),
    test_scenarios=[
        {
            "name": "User Registration Flow",
            "steps": ["Submit form", "Validate input", "Create user", "Send confirmation"],
            "expected_outcome": "User successfully registered"
        },
        {
            "name": "API Integration",
            "steps": ["Call API", "Handle response", "Update UI", "Handle errors"],
            "expected_outcome": "Data displayed correctly"
        }
    ]
)

COMPLEX_PROJECT = ProjectTestData(
    project_config=create_test_project_config("Microservices Platform", "complex"),
    tasks=[
        create_test_task_data("create_api_endpoint"),
        create_test_task_data("optimize_performance"),
        create_test_task_data("write_tests"),
        create_test_task_data("improve_ui"),
        create_test_task_data("fix_bug")
    ],
    expected_milestones=[
        {
            "name": "Service Architecture",
            "description": "Microservices architecture established",
            "completion_criteria": ["Service discovery", "Load balancing", "Circuit breakers"],
            "estimated_completion": datetime.now() + timedelta(hours=50)
        },
        {
            "name": "Observability",
            "description": "Monitoring and logging implemented",
            "completion_criteria": ["Metrics collection", "Distributed tracing", "Alerting"],
            "estimated_completion": datetime.now() + timedelta(hours=80)
        },
        {
            "name": "Production Deployment",
            "description": "System deployed to production",
            "completion_criteria": ["CI/CD pipeline", "Auto-scaling", "Disaster recovery"],
            "estimated_completion": datetime.now() + timedelta(hours=200)
        }
    ],
    code_samples=create_test_code_snippets(),
    test_scenarios=[
        {
            "name": "Service Communication",
            "steps": ["Service A calls Service B", "Handle network failures", "Circuit breaker activation"],
            "expected_outcome": "Graceful failure handling"
        },
        {
            "name": "Load Testing",
            "steps": ["Generate load", "Monitor performance", "Auto-scaling triggered"],
            "expected_outcome": "System scales automatically"
        }
    ]
)

# Collection of all project test data
ALL_PROJECT_TEST_DATA = {
    'simple': SIMPLE_PROJECT,
    'medium': MEDIUM_PROJECT,
    'complex': COMPLEX_PROJECT
}