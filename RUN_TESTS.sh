#!/bin/bash
# Run all tests for MyAgent

set -e

echo "🧪 MyAgent Test Suite"
echo "===================="
echo ""

# Activate venv
source venv/bin/activate

# Run tests with coverage
echo "Running integration tests..."
pytest tests/test_integration.py -v --asyncio-mode=auto

echo ""
echo "Running agent tests..."
pytest tests/test_agents.py -v

echo ""
echo "Running all tests with coverage..."
pytest tests/ --cov=core --cov=api --cov=config --cov-report=term --cov-report=html

echo ""
echo "✅ All tests complete!"
echo "Coverage report: htmlcov/index.html"
