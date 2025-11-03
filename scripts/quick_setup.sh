#!/bin/bash

# 🚀 MyAgent Continuous AI Builder - Quick Setup Script
# This script sets up the entire system on a new instance

set -e

echo "
╔══════════════════════════════════════════════════════════════╗
║     MyAgent Continuous AI Builder - Quick Setup Script      ║
║                                                              ║
║  This system will NEVER STOP until perfection is achieved   ║
╚══════════════════════════════════════════════════════════════╝
"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check operating system
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# Step 1: Check Python version
echo ""
echo "Step 1: Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.11 or 3.12"
    exit 1
fi

# Step 2: Check Node.js
echo ""
echo "Step 2: Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "Node.js $NODE_VERSION found"
else
    print_error "Node.js not found. Please install Node.js 18.x or 20.x"
    exit 1
fi

# Step 3: Install PostgreSQL if needed
echo ""
echo "Step 3: Setting up PostgreSQL..."
if command -v psql &> /dev/null; then
    print_status "PostgreSQL already installed"
else
    print_warning "PostgreSQL not found. Installing..."
    if [[ "$OS" == "macos" ]]; then
        brew install postgresql@15
        brew services start postgresql@15
    else
        sudo apt update
        sudo apt install -y postgresql-15
        sudo systemctl start postgresql
    fi
fi

# Create database
echo "Creating database..."
if [[ "$OS" == "macos" ]]; then
    createdb myagent_db 2>/dev/null || print_warning "Database already exists"
else
    sudo -u postgres createdb myagent_db 2>/dev/null || print_warning "Database already exists"
fi
print_status "Database ready"

# Step 4: Install Redis if needed
echo ""
echo "Step 4: Setting up Redis..."
if command -v redis-cli &> /dev/null; then
    print_status "Redis already installed"
else
    print_warning "Redis not found. Installing..."
    if [[ "$OS" == "macos" ]]; then
        brew install redis
        brew services start redis
    else
        sudo apt install -y redis-server
        sudo systemctl start redis-server
    fi
fi

# Test Redis connection
redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status "Redis is running"
else
    print_error "Redis is not running. Please start Redis manually"
fi

# Step 5: Python virtual environment
echo ""
echo "Step 5: Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
print_status "Python dependencies installed"

# Step 6: Frontend dependencies
echo ""
echo "Step 6: Setting up Frontend..."
cd frontend
npm install > /dev/null 2>&1
print_status "Frontend dependencies installed"
cd ..

# Step 7: Initialize database tables
echo ""
echo "Step 7: Initializing database..."
python3 << 'EOF'
import asyncpg
import asyncio

async def init_db():
    try:
        conn = await asyncpg.connect('postgresql://localhost:5432/myagent_db')

        # Create tables
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id UUID PRIMARY KEY,
                name VARCHAR(255),
                description TEXT,
                requirements JSONB,
                target_metrics JSONB,
                max_iterations INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                id UUID PRIMARY KEY,
                project_id UUID REFERENCES projects(id),
                name VARCHAR(100),
                role VARCHAR(100),
                status VARCHAR(50),
                metrics JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS iterations (
                id UUID PRIMARY KEY,
                project_id UUID REFERENCES projects(id),
                iteration_number INTEGER,
                metrics JSONB,
                tasks_completed JSONB,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        ''')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id UUID PRIMARY KEY,
                project_id UUID,
                error_type VARCHAR(100),
                error_message TEXT,
                stack_trace TEXT,
                solution TEXT,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        ''')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS milestones (
                id UUID PRIMARY KEY,
                project_id UUID,
                name VARCHAR(255),
                description TEXT,
                completed BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        ''')

        await conn.close()
        print("✅ Database tables created")

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

asyncio.run(init_db())
EOF

# Step 8: Create .env file if it doesn't exist
echo ""
echo "Step 8: Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# API Keys (REPLACE WITH YOUR ACTUAL KEYS)
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE

# Database
POSTGRES_URL=postgresql://localhost:5432/myagent_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=

# Redis
REDIS_URL=redis://localhost:6379

# Development Settings
DEV_MODE=true
MAX_ITERATIONS=1000
CHECKPOINT_INTERVAL=10

# Agent Configuration
AGENT_MODEL=gpt-4-turbo-preview
AGENT_TEMPERATURE=0.7
AGENT_MAX_TOKENS=4000

# Monitoring
QUALITY_CHECK_INTERVAL=300
EMERGENCY_DEBUG_THRESHOLD=1
EOF
    print_warning ".env file created - PLEASE ADD YOUR API KEYS"
else
    print_status ".env file already exists"
fi

# Step 9: Create necessary directories
echo ""
echo "Step 9: Creating directories..."
mkdir -p persistence/{database,vector_memory,checkpoints,agents,knowledge_graph}
mkdir -p logs/agents
mkdir -p scripts
print_status "Directories created"

# Step 10: System test
echo ""
echo "Step 10: Running system test..."
python3 << 'EOF'
try:
    from core.orchestrator.continuous_director import ContinuousDirector, QualityMetrics
    from core.memory.vector_memory import VectorMemory
    from core.memory.project_ledger import ProjectLedger
    from core.memory.error_knowledge_graph import ErrorKnowledgeGraph

    # Test orchestrator
    o = ContinuousDirector('test', {})
    print("✅ Orchestrator: OK")

    # Test memory systems
    vm = VectorMemory('test')
    print("✅ VectorMemory: OK")

    pl = ProjectLedger('test')
    print("✅ ProjectLedger: OK")

    eg = ErrorKnowledgeGraph()
    print("✅ ErrorKnowledgeGraph: OK")

    # Test metrics
    m = QualityMetrics()
    m.to_dict()
    print("✅ QualityMetrics: OK")

    print("\n🎉 All systems operational!")

except Exception as e:
    print(f"❌ System test failed: {e}")
    print("Please check the error and run setup again")
EOF

# Create start script
echo ""
echo "Creating start script..."
cat > start_system.sh << 'EOF'
#!/bin/bash

echo "Starting MyAgent Continuous AI Builder..."

# Terminal 1: API
echo "Starting API server..."
source venv/bin/activate
uvicorn api.main:app --reload --port 8000 &
API_PID=$!

# Terminal 2: Frontend
echo "Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  MyAgent System Started Successfully!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  API:       http://localhost:8000"
echo "  Frontend:  http://localhost:5173"
echo "  Docs:      http://localhost:8000/docs"
echo ""
echo "  API PID:      $API_PID"
echo "  Frontend PID: $FRONTEND_PID"
echo ""
echo "  To stop: kill $API_PID $FRONTEND_PID"
echo ""
echo "  The system will now run CONTINUOUSLY until it"
echo "  achieves PERFECTION. It will NEVER STOP working."
echo "═══════════════════════════════════════════════════════"

# Keep script running
wait
EOF

chmod +x start_system.sh
print_status "Start script created: ./start_system.sh"

# Final summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE! 🎉                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. Add your API keys to .env file"
echo "2. Run: ./start_system.sh"
echo "3. Create a project at http://localhost:5173"
echo ""
echo "The system will then run CONTINUOUSLY until achieving:"
echo "  • 95% test coverage"
echo "  • 0 critical bugs"
echo "  • 90% performance score"
echo "  • 90% documentation coverage"
echo "  • 95% security score"
echo ""
echo "It will NEVER STOP until PERFECTION is achieved."
echo ""
print_warning "Remember: This system runs forever. Use with caution."