#!/bin/bash
echo "🚀 Starting MyAgent Development Environment..."

# Activate Python environment
source venv/bin/activate

# Start Redis (if not running)
if ! pgrep redis-server > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Start PostgreSQL (if not running)
if ! pgrep postgres > /dev/null; then
    echo "Starting PostgreSQL..."
    brew services start postgresql@14 2>/dev/null || sudo service postgresql start 2>/dev/null || echo "Please start PostgreSQL manually"
fi

# Start API server
echo "Starting API server..."
uvicorn api.main:app --reload --port 8000 &
API_PID=$!

# Start frontend
echo "Starting frontend..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo "✅ MyAgent is now running!"
echo "🌐 API: http://localhost:8000"
echo "🖥️  Frontend: http://localhost:3000"
echo "📚 Docs: http://localhost:8000/docs"

# Wait for user input to stop
read -p "Press Enter to stop all services..."

# Kill background processes
kill $API_PID $FRONTEND_PID 2>/dev/null
echo "🛑 Services stopped"
