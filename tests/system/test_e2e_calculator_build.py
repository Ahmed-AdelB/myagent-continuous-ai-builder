"""
End-to-End Test: Build a Calculator Application from Scratch
"""

import asyncio
import pytest
from core.orchestrator.continuous_director import ContinuousDirector, ProjectState

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

@pytest.mark.e2e
async def test_build_calculator_from_scratch():
    """
    Test the system's ability to build a simple command-line calculator
    from a project specification to completion.
    """
    # GEMINI-EDIT - 2025-11-18 - Created E2E test for building a calculator.
    
    # 1. Define the project specification for the calculator
    calculator_project_spec = {
        "description": "A simple command-line calculator that can perform addition, subtraction, multiplication, and division.",
        "requirements": [
            "Create a main application file `calculator/main.py`.",
            "Implement a function for each operation: add, subtract, multiply, divide.",
            "Implement a command-line interface to take user input (e.g., '5 + 3').",
            "Handle basic errors like division by zero and invalid input.",
        ],
        "initial_files": {
            "calculator/__init__.py": "",
        }
    }

    # 2. Initialize the ContinuousDirector
    director = ContinuousDirector(
        project_name="E2E_Test_Calculator",
        project_spec=calculator_project_spec
    )

    # 3. Run the director's main loop
    # We'll run this with a timeout to prevent the test from running forever
    # in case of a deadlock or an infinite loop in the director.
    try:
        await asyncio.wait_for(director.start(), timeout=600) # 10-minute timeout
    except asyncio.TimeoutError:
        pytest.fail("The E2E test timed out after 10 minutes.")

    # 4. Assert that the project has reached the 'COMPLETED' state
    # This is the ultimate success condition for the E2E test.
    # It means the director believes it has met all quality metrics.
    assert director.state == ProjectState.COMPLETED, f"Director finished in state {director.state}, not COMPLETED."

    # 5. (Optional) Add more specific assertions here
    # For example, check if the final files exist, or if test coverage is high.
    # These would require the director to expose more state about the final project.
