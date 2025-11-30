import os
import sys
import asyncio
import logging
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
sys.modules["aiosqlite"] = MagicMock()

# Mock DB Manager
mock_db = MagicMock()
mock_db.execute = AsyncMock()
mock_db.fetchval = AsyncMock()
mock_db.fetchrow = AsyncMock()
mock_db.disconnect = AsyncMock()

# Patch Config
with patch("config.database.db_manager", mock_db), \
     patch("config.database.init_database", new_callable=AsyncMock), \
     patch("langchain_openai.ChatOpenAI") as mock_llm_class:

    # Configure Mock LLM
    mock_llm_instance = MagicMock()
    mock_llm_instance.apredict = AsyncMock(return_value="Mocked LLM Response")
    mock_llm_class.return_value = mock_llm_instance

    # 2. Import Real App
    # We need to import api.main inside the patch context so it uses the mocks
    from api.main import app
    
    # 3. Add Viewer Route
    @app.get("/viewer", response_class=HTMLResponse)
    async def serve_viewer():
        with open("scripts/simulation_viewer.html", "r") as f:
            return f.read()

    # 4. Run Server
    if __name__ == "__main__":
        print("\n🚀 Starting Standalone AI Builder...")
        print("🌍 GUI available at: http://localhost:8000/viewer")
        print("📡 API available at: http://localhost:8000/docs\n")
        
        # Open browser
        import webbrowser
        def open_browser():
            import time
            time.sleep(2)
            webbrowser.open("http://localhost:8000/viewer")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
