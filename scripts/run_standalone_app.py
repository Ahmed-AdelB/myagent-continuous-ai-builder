import os
import sys
import asyncio
import logging
import json
from unittest.mock import MagicMock, AsyncMock, patch
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneApp")

# 1. Mock Environment & Dependencies BEFORE imports
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"

# Mock OpenAI/LangChain
mock_openai = MagicMock()
mock_openai.__spec__ = MagicMock()
mock_openai.DefaultHttpxClient = object
sys.modules["openai"] = mock_openai

# Mock Database Drivers
sys.modules["asyncpg"] = MagicMock()
# sys.modules["aiosqlite"] = MagicMock() # Don't mock sqlite, let it work!

# Smart LLM Mock
class SmartMockLLM:
    def __init__(self, *args, **kwargs):
        pass

    async def apredict(self, prompt, **kwargs):
        prompt_lower = prompt.lower()
        
        if "plan" in prompt_lower or "architect" in prompt_lower:
            return json.dumps({
                "plan": [
                    "Initialize project structure",
                    "Create core modules",
                    "Implement main logic",
                    "Add unit tests"
                ],
                "files": [
                    {"path": "main.py", "description": "Entry point"},
                    {"path": "utils.py", "description": "Helper functions"}
                ]
            })
        
        if "code" in prompt_lower or "implement" in prompt_lower:
            return json.dumps({
                "files": [
                    {
                        "path": "main.py",
                        "content": "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()"
                    }
                ],
                "explanation": "Implemented basic entry point."
            })
            
        if "test" in prompt_lower:
            return json.dumps({
                "test_files": [
                    {
                        "path": "tests/test_main.py",
                        "content": "def test_main():\n    assert True"
                    }
                ]
            })

        return "I am a mocked AI agent. I have processed your request."

    def invoke(self, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.content = "Mocked synchronous response"
        return mock_resp

# Mock DB Manager
class MockDBManager:
    def __init__(self):
        self.projects = {}

    async def execute(self, query, *args):
        # Very basic query parsing
        if "INSERT INTO projects" in query:
            # args: name, spec, state, metrics
            name = args[0]
            spec = args[1]
            # We don't have ID here, but orchestrator handles it
            logger.info(f"Mock DB: Inserted project {name}")
        return None

    async def fetchrow(self, query, *args):
        return None
        
    async def fetch(self, query, *args):
        return []

    async def disconnect(self):
        pass

mock_db = MockDBManager()

# Patch Config
with patch("config.database.db_manager", mock_db), \
     patch("config.database.init_database", new_callable=AsyncMock), \
     patch("langchain_openai.ChatOpenAI", side_effect=SmartMockLLM):

    # 2. Import Real App
    from api.main import app
    
    # 3. Add Viewer Route
    @app.get("/viewer", response_class=HTMLResponse)
    async def serve_viewer():
        with open("scripts/simulation_viewer.html", "r") as f:
            return f.read()

    @app.get("/projects/{project_id}/files")
    async def list_files(project_id: str):
        # Return mock file list based on project state
        return ["main.py", "utils.py", "README.md", "tests/test_main.py", "requirements.txt"]

    # 4. Run Server
    if __name__ == "__main__":
        print("\n🚀 Starting Standalone AI Builder...")
        print("🌍 GUI available at: http://localhost:8000/viewer")
        
        # Open browser
        import webbrowser
        def open_browser():
            import time
            time.sleep(2)
            webbrowser.open("http://localhost:8000/viewer")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
