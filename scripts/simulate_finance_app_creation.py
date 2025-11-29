import asyncio
import os
import sys
import uuid
import json
import logging
import threading
import http.server
import socketserver
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("LiveSimulation")

# Mock environment
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"

# Mock dependencies BEFORE imports
sys.modules["aiosqlite"] = MagicMock()
sys.modules["aiosqlite"].connect.return_value.__aenter__.return_value = AsyncMock()

# Mock openai to avoid version mismatch issues in langchain_openai
mock_openai = MagicMock()
mock_openai.__spec__ = MagicMock() # Required for importlib
mock_openai.DefaultHttpxClient = object
sys.modules["openai"] = mock_openai

# Import core components
from core.orchestrator.continuous_director import ContinuousDirector, ProjectState

# Global state for JSON logs
EVENTS_FILE = "scripts/simulation_events.json"
simulation_state = {
    "status": "Initializing",
    "metrics": {"coverage": 0, "security": 0, "quality": 0, "iteration": 0},
    "logs": []
}

def write_event(agent, message):
    """Write event to JSON file and print to console"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Console output
    print(f"[{timestamp}] {agent}: {message}")
    
    # JSON output
    entry = {"timestamp": timestamp, "agent": agent, "message": message}
    simulation_state["logs"].append(entry)
    
    with open(EVENTS_FILE, "w") as f:
        json.dump(simulation_state, f, indent=2)

def update_metrics(coverage, security, quality, iteration):
    """Update metrics in JSON file"""
    simulation_state["metrics"] = {
        "coverage": coverage,
        "security": security,
        "quality": quality,
        "iteration": iteration
    }
    with open(EVENTS_FILE, "w") as f:
        json.dump(simulation_state, f, indent=2)

def update_status(status):
    """Update system status"""
    simulation_state["status"] = status
    with open(EVENTS_FILE, "w") as f:
        json.dump(simulation_state, f, indent=2)

def start_server():
    """Start HTTP server to serve the HTML viewer"""
    PORT = 8080
    DIRECTORY = "scripts"
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)
            
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n🌍 Simulation Viewer running at: http://localhost:{PORT}/simulation_viewer.html\n")
        httpd.serve_forever()

async def run_simulation():
    # Initialize events file
    with open(EVENTS_FILE, "w") as f:
        json.dump(simulation_state, f, indent=2)

    print("🚀 Starting Live Simulation...")
    
    # Start HTTP server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Give user time to open browser
    print("Waiting 5 seconds for server startup...")
    await asyncio.sleep(2)

    # 1. Setup
    write_event("System", "Initializing System (Headless Mode)...")
    update_status("Initializing")
    
    # Mock DB Manager
    mock_db = MagicMock()
    mock_db.create_project = AsyncMock(return_value="proj-123")
    mock_db.get_project = AsyncMock(return_value={
        "id": "proj-123", 
        "name": "Finance Planner", 
        "state": "planning"
    })
    
    # Mock LLM to return "Finance App" specific content
    mock_llm = MagicMock()
    mock_llm.apredict = AsyncMock(return_value="""
    Plan for Finance Planner:
    1. Create Expense model
    2. Create Budget controller
    3. Implement Dashboard UI
    """)
    
    # Patch dependencies
    with patch("config.database.DatabaseManager", return_value=mock_db), \
         patch("config.database.init_database", new_callable=AsyncMock), \
         patch("langchain_openai.ChatOpenAI", return_value=mock_llm):

        # 2. User Request
        write_event("User", "I want to build a Finance Planning App that tracks expenses and income.")
        
        project_id = str(uuid.uuid4())
        director = ContinuousDirector(
            project_name="Finance Planner",
            project_spec={
                "initial_requirements": ["Track expenses", "Track income", "Dashboard"],
                "target_metrics": {"test_coverage": 90.0}
            }
        )
        
        # 3. Start Orchestrator
        write_event("Orchestrator", "Starting AI Agents...")
        update_status("Starting Agents")
        
        await director._initialize_components()
        await director._initialize_agents()
        
        write_event("Orchestrator", "Agents Assigned: Architect, Coder, Tester, Debugger")
        update_status("Running")

        # 4. Simulate Planning Phase
        write_event("Architect", "Analyzing requirements for 'Finance Planner'...")
        await asyncio.sleep(2)
        write_event("Architect", "Drafting database schema: Users, Expenses, Income, Budgets.")
        await asyncio.sleep(1.5)
        write_event("Architect", "Defining API endpoints: /expenses, /income, /reports.")
        await asyncio.sleep(1.5)
        write_event("Architect", "Plan approved. Moving to Execution.")
        update_metrics(0, 0, 0, 1)

        # 5. Simulate Execution Phase
        write_event("Coder", "Generating `models/expense.py`...")
        await asyncio.sleep(2)
        write_event("Coder", "Implementing POST /expenses endpoint...")
        await asyncio.sleep(2)
        write_event("Coder", "Generating `frontend/src/Dashboard.tsx`...")
        await asyncio.sleep(2)
        update_metrics(0, 50, 60, 1)

        # 6. Simulate Verification Phase
        write_event("Tester", "Running unit tests...")
        await asyncio.sleep(2)
        write_event("Tester", "✓ test_create_expense PASSED")
        write_event("Tester", "✓ test_calculate_budget PASSED")
        write_event("Tester", "✗ test_invalid_input FAILED")
        update_metrics(65, 80, 70, 1)
        
        write_event("Debugger", "Detected failure in `test_invalid_input`.")
        await asyncio.sleep(1.5)
        write_event("Debugger", "Fixing validation logic in `api/routes/expenses.py`...")
        await asyncio.sleep(2)
        
        write_event("Tester", "Re-running tests...")
        await asyncio.sleep(1.5)
        write_event("Tester", "✓ test_invalid_input PASSED")
        update_metrics(98, 100, 95, 2)

        # 7. Completion
        write_event("System", "Project 'Finance Planner' built successfully!")
        update_status("Completed")
        
        # Keep server running for a bit
        print("\nSimulation complete. Keep script running to view results in browser.")
        print("Press Ctrl+C to exit.")
        await asyncio.sleep(300) # Keep alive for 5 mins
        
if __name__ == "__main__":
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        print("\nExiting...")
