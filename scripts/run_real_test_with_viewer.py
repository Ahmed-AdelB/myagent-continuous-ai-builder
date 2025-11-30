import asyncio
import os
import sys
import json
import time
import re
import threading
import http.server
import socketserver
import subprocess
from datetime import datetime

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

def parse_log_line(line):
    """Parse a log line from pytest output and extract agent/message"""
    line = line.strip()
    if not line:
        return None, None

    # Match standard log format: date | level | module | func | line - message
    # Example: 2025-11-30 03:29:21 | INFO | core.orchestrator.continuous_director: ...
    
    # Regex for loguru format (approximate)
    log_pattern = r".*\|\s+(\w+)\s+\|\s+([\w\.]+):.*-\s+(.*)"
    match = re.search(log_pattern, line)
    
    if match:
        level, module, message = match.groups()
        
        agent = "System"
        if "orchestrator" in module:
            agent = "Orchestrator"
        elif "coder" in module or "coder_agent" in message:
            agent = "Coder"
        elif "tester" in module or "tester_agent" in message:
            agent = "Tester"
        elif "architect" in module or "architect_agent" in message:
            agent = "Architect"
        elif "debugger" in module or "debugger_agent" in message:
            agent = "Debugger"
        
        return agent, message
    
    # Pytest output
    if "PASSED" in line:
        return "Tester", f"✅ {line}"
    if "FAILED" in line:
        return "Tester", f"❌ {line}"
    if "collected" in line:
        return "System", line
        
    return None, None

import webbrowser

# ... (imports)

def run_real_test():
    # Initialize events file
    with open(EVENTS_FILE, "w") as f:
        json.dump(simulation_state, f, indent=2)

    print("🚀 Starting Real System Test...")
    update_status("Running Tests")
    
    # Start HTTP server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Open browser automatically
    url = "http://localhost:8080/simulation_viewer.html"
    print(f"Opening browser at: {url}")
    webbrowser.open(url)
    
    # Give user time to open browser
    print("Waiting 3 seconds for server startup...")
    time.sleep(3)

    # Run pytest command
    cmd = [
        "python3", "-m", "pytest", 
        "tests/system/test_agent_logic_simulation.py", 
        "-v", 
        "--log-cli-level=INFO" # Enable live logging
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output
    for line in process.stdout:
        agent, message = parse_log_line(line)
        if agent and message:
            write_event(agent, message)
        else:
            # Print raw line to console but not to JSON (unless it looks important)
            print(line.strip())
            
    process.wait()
    
    if process.returncode == 0:
        write_event("System", "All System Tests Passed! 🚀")
        update_status("Completed (Success)")
    else:
        write_event("System", "Tests Failed. Check logs.")
        update_status("Completed (Failed)")

    print("\nTest execution complete. Keep script running to view results in browser.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    run_real_test()
