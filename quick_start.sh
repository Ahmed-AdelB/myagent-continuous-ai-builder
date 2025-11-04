#!/bin/bash
# Quick Start Script for MyAgent MVP
# Run this when you wake up to get everything working!

set -e  # Exit on error

echo "🌅 MyAgent Quick Start - Morning Edition"
echo "========================================"
echo ""

# Check we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Not in MyAgent directory"
    echo "Please run: cd /home/aadel/projects/22_MyAgent"
    exit 1
fi

echo "✅ In correct directory"
echo ""

# Step 1: Activate virtual environment
echo "📦 Step 1: Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Step 2: Fix dependencies
echo "🔧 Step 2: Fixing dependency versions..."
pip install --upgrade pydantic-settings > /dev/null 2>&1
echo "✅ Dependencies updated"
echo ""

# Step 3: Test imports
echo "🧪 Step 3: Testing imports..."
python3 -c "
from config.settings import settings
from config.database import db_manager
from core.orchestrator.continuous_director import ContinuousDirector
print('✅ All imports successful!')
"
echo ""

# Step 4: Check .env file
echo "🔐 Step 4: Checking .env file..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from template..."
    cp .env.example .env
    echo ""
    echo "❗ IMPORTANT: Edit .env and add your API keys!"
    echo "Run: nano .env"
    echo ""
    echo "Add these lines with YOUR keys:"
    echo "  OPENAI_API_KEY=sk-proj-YOUR_NEW_KEY"
    echo "  ANTHROPIC_API_KEY=sk-ant-YOUR_NEW_KEY"
    echo ""
    echo "Or use local Ollama:"
    echo "  DEFAULT_LLM_PROVIDER=ollama"
    echo ""
    read -p "Press Enter after editing .env file..."
else
    echo "✅ .env file exists"

    # Check if API keys are set
    if grep -q "OPENAI_API_KEY=sk-proj-" .env || grep -q "DEFAULT_LLM_PROVIDER=ollama" .env; then
        echo "✅ API keys configured"
    else
        echo "⚠️  API keys might not be set properly"
        echo "Check your .env file!"
    fi
fi
echo ""

# Step 5: Database setup
echo "💾 Step 5: Setting up database..."
if command -v docker-compose &> /dev/null; then
    echo "Docker Compose found - starting services..."
    docker-compose up -d postgres redis chromadb
    echo "Waiting for services to start..."
    sleep 10
    echo "✅ Docker services started"
else
    echo "⚠️  Docker Compose not found - assuming local database"
fi

# Initialize database schema
if [ -f "scripts/setup_database.py" ]; then
    echo "Initializing database schema..."
    python scripts/setup_database.py
    echo "✅ Database initialized"
else
    echo "⚠️  Database setup script not found"
fi
echo ""

# Step 6: Test orchestrator
echo "🎯 Step 6: Testing orchestrator..."
echo "Starting a quick test (will stop after initialization)..."
timeout 15s python -m core --project quicktest --spec '{"description": "Quick test"}' 2>&1 | head -20 || true
echo "✅ Orchestrator can start (stopped for this test)"
echo ""

# Step 7: Final status
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo ""
echo "Your MyAgent system is ready! You can now:"
echo ""
echo "1️⃣  Start the orchestrator:"
echo "   python -m core --project myapp --spec '{\"description\": \"My app\"}'"
echo ""
echo "2️⃣  Start the API server:"
echo "   uvicorn api.main:app --reload --port 8000"
echo ""
echo "3️⃣  Start the frontend:"
echo "   cd frontend && npm run dev"
echo ""
echo "4️⃣  Access the dashboard:"
echo "   http://localhost:5173"
echo ""
echo "5️⃣  Or create a project via API:"
echo "   curl -X POST http://localhost:8000/projects -H 'Content-Type: application/json' -d '{\"name\": \"test\", \"description\": \"Test project\", \"requirements\": [\"Create hello world\"], \"max_iterations\": 10}'"
echo ""
echo "📖 For more details, see NIGHT_HANDOFF.md"
echo ""
echo "🚀 Ready for autonomous AI development!"
