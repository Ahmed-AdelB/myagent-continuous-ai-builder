# 🚀 MyAgent Continuous AI Builder - Complete Migration & Setup Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites & Requirements](#prerequisites--requirements)
3. [Installation Guide](#installation-guide)
4. [Configuration](#configuration)
5. [Architecture Deep Dive](#architecture-deep-dive)
6. [API Documentation](#api-documentation)
7. [Agent Specifications](#agent-specifications)
8. [Continuous Development Workflow](#continuous-development-workflow)
9. [Troubleshooting](#troubleshooting)
10. [Development Guidelines](#development-guidelines)

---

## 🎯 System Overview

MyAgent is an autonomous AI development system that continuously builds and improves applications until they achieve perfection. Unlike traditional generators, this system **never stops working** until all quality metrics are met.

### Core Philosophy
- **Continuous Operation**: Runs 24/7 until perfection is achieved
- **Self-Healing**: Automatically detects and fixes issues
- **Learning System**: Learns from every error and improves
- **Multi-Agent Architecture**: 6 specialized AI agents working in harmony

### Success Criteria (Perfection Metrics)
- ✅ Test Coverage ≥ 95%
- ✅ Critical Bugs = 0
- ✅ Performance Score ≥ 90%
- ✅ Documentation Coverage ≥ 90%
- ✅ Security Score ≥ 95%
- ✅ Code Quality Score ≥ 85%
- ✅ User Satisfaction ≥ 90%

---

## 📦 Prerequisites & Requirements

### System Requirements
```bash
# Operating System
- Ubuntu 20.04+ / macOS 12+ / Windows 11 with WSL2
- RAM: Minimum 8GB (16GB recommended)
- Storage: 20GB free space
- CPU: 4+ cores recommended

# Software Dependencies
- Python 3.11 or 3.12
- Node.js 18.x or 20.x
- PostgreSQL 15+
- Redis 7+
- Git 2.x
```

### Required API Keys
```bash
# Create .env file with these keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
POSTGRES_URL=postgresql://localhost:5432/myagent_db
REDIS_URL=redis://localhost:6379
```

---

## 🔧 Installation Guide

### Step 1: Clone Repository
```bash
git clone https://github.com/Ahmed-AdelB/myagent-continuous-ai-builder.git
cd myagent-continuous-ai-builder
```

### Step 2: Install PostgreSQL
```bash
# macOS
brew install postgresql@15
brew services start postgresql@15
createdb myagent_db

# Ubuntu
sudo apt update
sudo apt install postgresql-15
sudo systemctl start postgresql
sudo -u postgres createdb myagent_db

# Windows (WSL2)
sudo apt install postgresql
sudo service postgresql start
sudo -u postgres createdb myagent_db
```

### Step 3: Install Redis
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/WSL2
sudo apt install redis-server
sudo systemctl start redis-server
```

### Step 4: Python Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for agents
pip install langchain-openai langchain-core sentence-transformers chromadb
pip install pytest pytest-cov pytest-asyncio coverage
```

### Step 5: Frontend Setup
```bash
cd frontend
npm install
npm run build
cd ..
```

### Step 6: Database Initialization
```bash
# Run database setup script
python3 scripts/initialize_database.py

# Or manually via Python
python3 << EOF
import asyncpg
import asyncio

async def create_tables():
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

    await conn.close()
    print("✅ Database tables created successfully")

asyncio.run(create_tables())
EOF
```

---

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# API Keys
OPENAI_API_KEY=sk-proj-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Database
POSTGRES_URL=postgresql://localhost:5432/myagent_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword

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
QUALITY_CHECK_INTERVAL=300  # 5 minutes
EMERGENCY_DEBUG_THRESHOLD=1  # Number of critical bugs to trigger emergency mode
```

### API Configuration (api/config.py)
```python
DATABASE_CONFIG = {
    "min_connections": 5,
    "max_connections": 20,
    "timeout": 10
}

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000"
]

WEBSOCKET_CONFIG = {
    "heartbeat_interval": 30,
    "max_connections_per_project": 10
}
```

---

## 🏗️ Architecture Deep Dive

### System Components

#### 1. Continuous Director (Orchestrator)
**Location**: `core/orchestrator/continuous_director.py`

**Responsibilities**:
- Manages the entire development lifecycle
- Coordinates all agents
- Monitors quality metrics
- Triggers self-healing operations

**Key Methods**:
```python
async def start()           # Starts continuous development
async def continuous_quality_monitor()  # Monitors metrics every 5 minutes
async def emergency_debug_mode()        # Activated when critical bugs detected
async def trigger_test_intensification()  # Boosts test coverage
```

#### 2. Memory Systems

**Project Ledger** (`core/memory/project_ledger.py`)
- Complete version history
- Change tracking
- Rollback support

**Vector Memory** (`core/memory/vector_memory.py`)
- Semantic search
- Context retrieval
- Pattern recognition

**Error Knowledge Graph** (`core/memory/error_knowledge_graph.py`)
- Error pattern learning
- Solution mapping
- Prevention strategies

#### 3. AI Agents

Each agent inherits from `PersistentAgent` base class and has specialized capabilities:

```python
agents = {
    "coder": CoderAgent(),       # Code generation
    "tester": TesterAgent(),      # Test creation & execution
    "debugger": DebuggerAgent(),  # Bug fixing
    "architect": ArchitectAgent(), # System design
    "analyzer": AnalyzerAgent(),   # Performance & quality
    "ui_refiner": UIRefinerAgent() # UX improvements
}
```

#### 4. Quality Monitoring

**Continuous Quality Monitor**:
- Runs every 5 minutes
- Checks all metrics against thresholds
- Triggers appropriate optimizations
- Maintains system health

**Self-Healing Triggers**:
```python
if metrics["test_coverage"] < 95:
    await trigger_test_intensification()

if metrics["bug_count_critical"] > 0:
    await emergency_debug_mode()

if metrics["performance_score"] < 90:
    await trigger_performance_optimization()
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Project Management

**Create Project**
```http
POST /projects
Content-Type: application/json

{
  "name": "Project Name",
  "description": "Project description",
  "requirements": ["feature1", "feature2"],
  "target_metrics": {
    "test_coverage": 95,
    "performance": 90
  },
  "max_iterations": 1000
}

Response: 200 OK
{
  "id": "uuid",
  "name": "Project Name",
  "state": "initializing",
  "iteration": 0,
  "metrics": {},
  "estimated_completion": null
}
```

**Get Project Status**
```http
GET /projects/{project_id}

Response: 200 OK
{
  "id": "uuid",
  "state": "developing",
  "iteration": 42,
  "metrics": {
    "test_coverage": 87.5,
    "bug_count_critical": 0,
    "performance_score": 92.3
  }
}
```

**List All Projects**
```http
GET /projects

Response: 200 OK
[
  {
    "id": "uuid",
    "name": "Project 1",
    "state": "developing"
  }
]
```

#### 2. Control Operations

**Pause Project**
```http
POST /projects/{project_id}/pause
```

**Resume Project**
```http
POST /projects/{project_id}/resume
```

**Delete Project**
```http
DELETE /projects/{project_id}
```

#### 3. WebSocket Connections

**Project-Specific WebSocket**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/{project_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time updates
};
```

**General WebSocket**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
// Connects to first available project or waits
```

---

## 🤖 Agent Specifications

### 1. Coder Agent
**File**: `core/agents/coder_agent.py`

**Capabilities**:
- Generate code from requirements
- Implement features
- Refactor existing code
- Add documentation

**Key Tasks**:
```python
{
  "type": "implement_feature",
  "description": "Add user authentication",
  "priority": 8,
  "data": {
    "requirements": ["JWT tokens", "OAuth2"],
    "target_files": ["auth.py", "middleware.py"]
  }
}
```

### 2. Tester Agent
**File**: `core/agents/tester_agent.py`

**Capabilities**:
- Generate unit tests
- Create integration tests
- Measure coverage
- Run test suites

**Coverage Target**: 95% minimum

### 3. Debugger Agent
**File**: `core/agents/debugger_agent.py`

**Capabilities**:
- Analyze error logs
- Fix bugs
- Prevent error recurrence
- Emergency debug mode

**Emergency Mode**: Activated when critical bugs > 0

### 4. Architect Agent
**File**: `core/agents/architect_agent.py`

**Capabilities**:
- Review system design
- Suggest improvements
- Ensure scalability
- Maintain patterns

### 5. Analyzer Agent
**File**: `core/agents/analyzer_agent.py`

**Capabilities**:
- Performance profiling
- Security audits
- Code quality analysis
- Metric tracking

### 6. UI Refiner Agent
**File**: `core/agents/ui_refiner_agent.py`

**Capabilities**:
- Improve UX
- Enhance accessibility
- Optimize frontend performance
- User feedback integration

---

## 🔄 Continuous Development Workflow

### The Eternal Loop
```python
while not metrics.is_perfect():
    # 1. Planning Phase
    analyze_current_state()
    generate_tasks()
    prioritize_tasks()

    # 2. Implementation Phase
    distribute_tasks_to_agents()
    execute_parallel_development()

    # 3. Testing Phase
    run_tests()
    measure_coverage()

    # 4. Optimization Phase
    optimize_performance()
    fix_bugs()
    improve_documentation()

    # 5. Learning Phase
    analyze_errors()
    update_knowledge_graph()
    adapt_strategies()

    # 6. Checkpoint
    save_progress()
    update_metrics()

    # Check every 5 minutes
    if needs_optimization():
        trigger_self_healing()
```

### Task Priority System
1. **Priority 10**: Emergency (critical bugs)
2. **Priority 9**: Security issues
3. **Priority 8**: Test coverage
4. **Priority 7**: Performance
5. **Priority 6**: Documentation
6. **Priority 5**: Features
7. **Priority 4**: Refactoring
8. **Priority 3**: UI/UX
9. **Priority 2**: Minor improvements
10. **Priority 1**: Nice-to-have

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. VectorMemory Initialization Error
```bash
Error: VectorMemory.__init__() missing 1 required positional argument: 'project_name'
```
**Solution**: Fixed in latest version - VectorMemory now receives project_name

#### 2. Database Connection Failed
```bash
Error: could not connect to server: Connection refused
```
**Solution**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Start if not running
sudo systemctl start postgresql

# Verify database exists
psql -U postgres -l | grep myagent_db
```

#### 3. Redis Connection Issues
```bash
Error: Redis connection refused
```
**Solution**:
```bash
# Check Redis status
redis-cli ping

# Start Redis
brew services start redis  # macOS
sudo systemctl start redis-server  # Ubuntu
```

#### 4. Agent Initialization Failures
```bash
Error: Agent initialization failed
```
**Solution**:
- Check OpenAI API key is valid
- Ensure all dependencies installed
- Verify agent parameters match orchestrator

#### 5. Low Test Coverage
```bash
Warning: Test coverage 82% < 95% required
```
**Solution**: System will automatically trigger test intensification

#### 6. WebSocket Connection Failed
```bash
Error: WebSocket connection failed
```
**Solution**:
- Ensure API is running on port 8000
- Check CORS settings
- Verify project_id exists

---

## 👨‍💻 Development Guidelines

### For New Engineers

#### 1. Understanding the Codebase
```bash
# Key directories
core/
├── orchestrator/     # Main control system
├── agents/          # AI agents
├── memory/          # Persistence systems
└── learning/        # ML components

api/                 # FastAPI backend
frontend/           # React frontend
tests/             # Test suites
scripts/           # Utility scripts
```

#### 2. Making Changes
```bash
# Always work in a branch
git checkout -b feature/your-feature

# Run tests before committing
pytest tests/ -v --cov=core

# Ensure coverage ≥ 95%
coverage report

# Commit with descriptive messages
git commit -m "feat: Add new capability to Coder Agent"
```

#### 3. Adding New Agents
```python
# Inherit from PersistentAgent
class NewAgent(PersistentAgent):
    def __init__(self, orchestrator=None):
        super().__init__(
            name="new_agent",
            role="Description",
            capabilities=["capability1", "capability2"],
            orchestrator=orchestrator
        )

    async def initialize(self):
        # Setup code
        pass

    async def process_task(self, task: AgentTask):
        # Task processing logic
        pass
```

#### 4. Testing Standards
- Every feature needs tests
- Coverage must be ≥ 95%
- Use pytest fixtures
- Mock external API calls
- Test both success and failure cases

### For AI Agents Continuing Work

#### Understanding Context
```python
# Access project history
project_ledger = ProjectLedger(project_name)
history = project_ledger.get_iteration_history()

# Access error patterns
error_graph = ErrorKnowledgeGraph()
patterns = error_graph.get_error_patterns()

# Access vector memory
vector_memory = VectorMemory(project_name)
context = vector_memory.get_context_window(current_task)
```

#### Task Execution
```python
# Receive task from orchestrator
task = await receive_task()

# Check task type and priority
if task.priority >= 9:
    # High priority - execute immediately
    result = await execute_urgent_task(task)
else:
    # Normal priority - plan execution
    plan = await create_execution_plan(task)
    result = await execute_plan(plan)

# Report results
await report_to_orchestrator(result)
```

#### Learning from Errors
```python
# When error occurs
error_graph.add_error(
    error_pattern=error_signature,
    context=task_context,
    attempted_solution=solution
)

# Learn from success
if solution_worked:
    error_graph.add_solution(
        error_pattern=error_signature,
        solution=solution,
        success_rate=1.0
    )
```

---

## 🚀 Starting the System

### Quick Start
```bash
# Terminal 1: Start API
source venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start Frontend
cd frontend
npm run dev

# Terminal 3: Monitor logs
tail -f logs/orchestrator.log

# Terminal 4: Create a project
python3 test_api.py
```

### Production Start
```bash
# Use Docker Compose
docker-compose up -d

# Or use systemd services
sudo systemctl start myagent-api
sudo systemctl start myagent-frontend
sudo systemctl start myagent-orchestrator
```

### Monitoring
```bash
# Check system status
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/projects/{id}/metrics

# Watch real-time logs
journalctl -u myagent-orchestrator -f
```

---

## 📊 Success Verification

### System Health Check
```python
# Run health check script
python3 scripts/health_check.py

# Expected output:
✅ Database: Connected
✅ Redis: Connected
✅ API: Running (port 8000)
✅ Frontend: Running (port 5173)
✅ Agents: All initialized
✅ Quality Monitor: Active
✅ Self-Healing: Enabled
```

### Metrics Dashboard
Access: http://localhost:5173/dashboard

Shows:
- Current iteration
- All quality metrics
- Agent status
- Task queue
- Error patterns
- Learning progress

---

## 📝 Important Notes

### For Migration
1. **Database**: Export/import PostgreSQL data
2. **Memory**: Copy `persistence/` directory
3. **Environment**: Set all required API keys
4. **Dependencies**: Use exact versions from requirements.txt

### Critical Files
```bash
# Must be preserved during migration
persistence/
├── database/          # SQLite ledgers
├── vector_memory/     # ChromaDB embeddings
├── checkpoints/       # System checkpoints
└── knowledge_graph/   # Error patterns

.env                   # API keys and config
```

### Performance Tips
- Allocate sufficient RAM for vector operations
- Use SSD for database storage
- Enable Redis persistence
- Monitor CPU usage during peak operations
- Scale horizontally by running agents on separate machines

---

## 🆘 Support & Contact

### GitHub Repository
https://github.com/Ahmed-AdelB/myagent-continuous-ai-builder

### Issue Reporting
Create issues with:
- Error messages
- System logs
- Steps to reproduce
- Environment details

### Community
- Discussions: GitHub Discussions
- Updates: Watch repository for releases
- Contributing: See CONTRIBUTING.md

---

## 📚 Additional Resources

### Documentation
- [API Reference](./docs/api_reference.md)
- [Agent Development Guide](./docs/agent_guide.md)
- [Memory Systems](./docs/memory_systems.md)
- [Learning Engine](./docs/learning_engine.md)

### Scripts
- `scripts/initialize_database.py` - Database setup
- `scripts/health_check.py` - System health verification
- `scripts/export_memory.py` - Export memory for migration
- `scripts/import_memory.py` - Import memory after migration

---

**Last Updated**: November 2024
**Version**: 1.0.0 (GPT-5 Enhanced)
**Status**: Production Ready

---

*This system will continue developing until perfection is achieved. It never gives up.*