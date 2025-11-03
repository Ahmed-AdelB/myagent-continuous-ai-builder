#!/bin/bash
echo "🚀 Starting MyAgent Development Environment..."

# Activate Python environment
echo "Activating Python environment..."
source venv/bin/activate

# Check if PostgreSQL is running
if ! pgrep -x "postgres" > /dev/null; then
    echo "⚠️  PostgreSQL not running. Please start PostgreSQL:"
    echo "   macOS: brew services start postgresql"
    echo "   Ubuntu: sudo service postgresql start"
    echo "   Docker: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:14"
fi

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "⚠️  Redis not running. Please start Redis:"
    echo "   macOS: brew services start redis"
    echo "   Ubuntu: sudo service redis-server start"
    echo "   Docker: docker run -d -p 6379:6379 redis:7-alpine"
fi

# Start API server
echo "🌐 Starting FastAPI server..."
uvicorn api.main:app --reload --port 8000 &
API_PID=$!

# Wait a moment for API to start
sleep 2

# Start frontend (if exists)
if [ -d "frontend" ]; then
    echo "🎨 Starting React frontend..."
    cd frontend && npm run dev &
    FRONTEND_PID=$!
    cd ..
else
    echo "⚠️  Frontend directory not found"
fi

echo ""
echo "✅ MyAgent Development Environment Started!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 API Server: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🖥️  Frontend Dashboard: http://localhost:3000"
echo "🔍 Agent Status: http://localhost:8000/api/agents/status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Quick Commands:"
echo "   Test API: curl http://localhost:8000/health"
echo "   View logs: tail -f logs/myagent.log"
echo "   Run tests: pytest tests/ -v"
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping MyAgent services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
        echo "   ✅ API server stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "   ✅ Frontend stopped"
    fi
    echo "🎉 MyAgent development environment stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Wait for user input or services to exit
wait