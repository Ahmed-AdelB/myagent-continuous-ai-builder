# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Continuous AI App Builder (MyAgent)** - A never-stopping AI development platform that employs persistent multi-agent architecture to iteratively build and improve applications until they achieve production-grade perfection across 8 quality metrics.

**Core Philosophy**: Not a 5-minute generator, but a tireless AI development team that works continuously across days, weeks, or months until the application meets enterprise-grade quality standards.

## Commands

### Development Setup
```bash
# Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..

# Environment configuration
cp .env.example .env
# Edit .env with required API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

# Initialize databases
python scripts/setup_database.py
```

### Running the System
```bash
# Main orchestrator (the continuous development loop)
python -m core.orchestrator.continuous_director

# API server
uvicorn api.main:app --reload --port 8000

# Frontend dashboard
cd frontend && npm run dev

# Access dashboard at: http://localhost:5173
```

### Docker Deployment
```bash
# Development environment
docker-compose up

# Production
docker-compose --profile prod up -d

# With monitoring stack
docker-compose --profile monitoring up -d
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=core --cov-report=term --cov-report=xml

# Run specific test file
pytest tests/test_continuous_system.py -v

# Async tests
pytest tests/ --asyncio-mode=auto
```

### Code Quality
```bash
# Format code
black core/ api/ tests/

# Lint
pylint core/ api/
flake8 core/ api/

# Type checking
mypy core/ api/

# Security audit
bandit -r core/
safety check
```

### Frontend
```bash
cd frontend

# Development server
npm run dev

# Production build
npm run build

# Lint
npm run lint
```

## Architecture

### System Structure

```
┌─────────────────────────────────┐
│  Frontend (React + WebSocket)   │  Real-time dashboard
├─────────────────────────────────┤
│  API Layer (FastAPI)            │  REST + WebSocket endpoints
├─────────────────────────────────┤
│  Orchestration (Director)       │  Continuous coordination loop
├─────────────────────────────────┤
│  Agent Layer (6 Specialists)    │  Coder, Tester, Debugger, etc.
├─────────────────────────────────┤
│  Memory Layer                   │  Ledger, Vector Memory, Error Graph
├─────────────────────────────────┤
│  Data Layer                     │  PostgreSQL, Redis, ChromaDB
└─────────────────────────────────┘
```

### Multi-Agent System

**6 Specialized Agents** working in harmony through the orchestrator:

1. **Coder Agent** (`core/agents/coder_agent.py`)
   - Code generation and refactoring
   - Feature implementation
   - Documentation writing

2. **Tester Agent** (`core/agents/tester_agent.py`)
   - Test creation (unit, integration)
   - Test execution
   - Coverage measurement

3. **Debugger Agent** (`core/agents/debugger_agent.py`)
   - Error analysis and resolution
   - Stack trace interpretation
   - Fix generation

4. **Architect Agent** (`core/agents/architect_agent.py`)
   - System design review
   - Architecture optimization
   - Pattern recommendations

5. **Analyzer Agent** (`core/agents/analyzer_agent.py`)
   - Performance monitoring
   - Metrics tracking
   - Trend analysis

6. **UI Refiner Agent** (`core/agents/ui_refiner_agent.py`)
   - UX improvements
   - Accessibility enhancements
   - UI optimization

**Communication**: All agents communicate through the orchestrator (no direct agent-to-agent communication) using a message-passing protocol.

### Memory Systems

**Project Ledger** (`core/memory/project_ledger.py`)
- Complete immutable version history (Event Sourcing pattern)
- All code changes, decisions, and iteration summaries
- SQLite-based for structured data

**Vector Memory** (`core/memory/vector_memory.py`)
- Semantic search using ChromaDB
- Code snippets, decisions, errors, and patterns
- Enables context-aware retrieval

**Error Knowledge Graph** (`core/memory/error_knowledge_graph.py`)
- Graph-based error-solution mapping (NetworkX)
- Learns from every error encountered
- Builds reusable solution patterns

### The Continuous Loop

The heart of the system at `core/orchestrator/continuous_director.py`:

```python
while not self.metrics.is_perfect():
    await self._execute_iteration()
    # Quality monitor runs in parallel (every 5 minutes)
    # Self-healing triggers automatically
```

**Perfection Criteria** (all must be met):
- Test Coverage ≥ 95%
- Critical Bugs = 0
- Minor Bugs ≤ 5
- Performance Score ≥ 90%
- Documentation Coverage ≥ 90%
- Code Quality Score ≥ 85%
- User Satisfaction ≥ 90%
- Security Score ≥ 95%

### Self-Healing Mechanisms

The system automatically triggers optimizations when metrics drop:
- Test coverage < 95% → Test intensification
- Critical bugs > 0 → Emergency debug mode (highest priority)
- Performance < 90% → Performance optimization
- Security < 95% → Security audit and fixes

## Key Implementation Patterns

### Agent Implementation Pattern

```python
class NewAgent(PersistentAgent):
    def __init__(self, orchestrator=None):
        super().__init__(
            name="agent_name",
            role="Agent Role",
            capabilities=["capability1", "capability2"],
            orchestrator=orchestrator
        )

    async def initialize(self):
        """Called once during orchestrator initialization"""
        await self._load_checkpoint()
        await self._setup_llm()

    async def process_task(self, task: AgentTask):
        """Main task processing method"""
        try:
            result = await self._execute_task(task)
            await self._save_to_memory(result)
            return result
        except Exception as e:
            await self._handle_error(e, task)

    async def cleanup(self):
        """Called on shutdown"""
        await self._save_checkpoint()
```

### Memory Access Pattern

```python
# Always use context manager for memory operations
async with self.orchestrator.vector_memory as memory:
    # Store new memory
    memory.store_memory(
        content="Solution details",
        memory_type="solutions",
        metadata={"iteration": self.iteration}
    )

    # Retrieve relevant memories
    relevant = memory.search_memories(
        query="error description",
        memory_type="errors",
        top_k=5
    )
```

### Error Handling & Learning Pattern

```python
try:
    result = await risky_operation()
except Exception as e:
    # Record in error graph
    self.orchestrator.error_graph.add_error(
        error_type=type(e).__name__,
        error_message=str(e),
        context=current_context,
        stack_trace=traceback.format_exc()
    )

    # Check if we've seen this before
    known_solution = self.orchestrator.error_graph.find_solution(e)
    if known_solution:
        result = await apply_solution(known_solution)
    else:
        # Try to fix and learn
        result = await attempt_fix(e)
        if result.success:
            self.orchestrator.error_graph.add_solution(
                error_pattern=e,
                solution=result.solution
            )
```

## Critical Files (DO NOT DELETE)

### Core System Files
```
core/orchestrator/continuous_director.py    # Main control loop
core/orchestrator/milestone_tracker.py      # Progress tracking
core/orchestrator/checkpoint_manager.py     # State snapshots
core/orchestrator/progress_analyzer.py      # Metric analysis
```

### Memory System Files
```
core/memory/project_ledger.py              # Version history
core/memory/vector_memory.py               # Semantic memory
core/memory/error_knowledge_graph.py       # Error learning
```

### Agent Files
```
core/agents/base_agent.py                  # Agent foundation
core/agents/coder_agent.py                 # Code generation
core/agents/tester_agent.py                # Test creation
core/agents/debugger_agent.py              # Bug fixing
core/agents/architect_agent.py             # System design
core/agents/analyzer_agent.py              # Performance monitoring
core/agents/ui_refiner_agent.py            # UX improvement
```

### Persistence Directories (MUST PRESERVE)
```
persistence/database/         # SQLite databases
persistence/vector_memory/    # ChromaDB data
persistence/checkpoints/      # System state snapshots
persistence/agents/           # Agent states
persistence/knowledge_graph/  # Error patterns
```

## Technology Stack

**Backend**: Python 3.11+, FastAPI, LangChain, OpenAI/Anthropic/Ollama APIs
**Frontend**: React 18, TypeScript, Vite, TailwindCSS, Chart.js
**Databases**: PostgreSQL 15, Redis 7, ChromaDB, MongoDB
**Vector Search**: FAISS, ChromaDB
**Task Queue**: Celery with Redis
**Testing**: pytest, pytest-asyncio, pytest-cov
**DevOps**: Docker, Docker Compose, GitHub Actions

## Debugging & Monitoring

### Check System Status
```bash
# Current metrics
curl http://localhost:8000/projects/{id}/metrics | jq

# Monitor iteration progress
watch -n 5 'curl -s http://localhost:8000/projects/{id} | jq .iteration'

# Agent status
curl http://localhost:8000/projects/{id}/agents | jq
```

### Log Locations
```bash
# Orchestrator logs
tail -f logs/orchestrator.log

# Agent-specific logs
tail -f logs/agents/coder_agent.log
tail -f logs/agents/debugger_agent.log

# API logs
tail -f logs/api.log
```

### Test Component Initialization
```bash
# Test orchestrator
python3 -c "from core.orchestrator.continuous_director import ContinuousDirector; o = ContinuousDirector('test', {}); print('✅ OK')"

# Test memory
python3 -c "from core.memory.vector_memory import VectorMemory; vm = VectorMemory('test'); print('✅ OK')"

# Test agents
python3 -c "from core.agents.coder_agent import CoderAgent; agent = CoderAgent(); print('✅ OK')"
```

## CI/CD Pipeline

**File**: `.github/workflows/continuous_quality.yml`

- Runs on push, PR, and **hourly schedule** (continuous quality enforcement)
- Matrix testing: Python 3.11/3.12, Node 18.x/20.x
- Services: PostgreSQL 15, Redis 7
- **Quality Gates**:
  - Test coverage ≥ 95% (enforced)
  - Critical bugs = 0 (enforced)
  - Performance tests (< 2s init time)
  - Security audit (Bandit + Safety)
  - Documentation coverage ≥ 90%
- Auto-deployment when all metrics are perfect on main branch

## Development Workflow

### Adding a New Feature

1. **Requirement Analysis** (Architect Agent)
   - Analyze requirements and create design document
   - Identify dependencies and complexity

2. **Implementation** (Coder Agent)
   - Generate code with inline documentation
   - Create initial structure

3. **Testing** (Tester Agent)
   - Generate unit and integration tests
   - Ensure edge cases are covered

4. **Optimization** (Analyzer Agent)
   - Performance profiling
   - Security audit
   - Code quality check

5. **Refinement** (UI Refiner Agent)
   - UI/UX improvements
   - Accessibility enhancements

### Handling Errors

When errors occur, the system:
1. Records error in Error Knowledge Graph
2. Searches for similar historical errors
3. Applies known solution if available
4. Otherwise, attempts fix and learns from result
5. Adds regression tests to prevent recurrence

## Important Notes

### Never-Stopping Philosophy
The system literally runs forever until perfection criteria are met. This is intentional design, not a bug.

### State Persistence
All state must persist across sessions. Nothing is lost. The system can be stopped and resumed at any checkpoint.

### Learning Accumulation
Knowledge accumulates across all projects. Patterns learned in one project can benefit future projects.

### Quality-Driven Development
Every action is measured against the 8 quality metrics. The system will not stop until all thresholds are met.

### Emergency Debug Mode
When critical bugs are detected (count > 0), the system enters emergency mode where all other tasks are suspended until critical bugs are resolved.

## Performance Expectations

**Initialization Times**:
- Orchestrator: < 2 seconds
- Agents: < 0.5 seconds each
- Memory load: < 1 second

**Operation Times**:
- Task execution: < 30 seconds average
- Iteration cycle: < 5 minutes
- Quality check: < 10 seconds

**Resource Usage**:
- RAM: < 4GB normal, < 8GB peak
- CPU: < 50% average, < 80% peak
- Disk I/O: < 100 MB/s

## Security Considerations

- Never commit API keys to repository (use .env)
- All generated code is scanned for vulnerabilities
- Security score must be ≥ 95%
- Regular key rotation recommended
- Database encryption at rest
- JWT with short expiration for API auth

## Emergency Procedures

### If System Stops Unexpectedly
```bash
# Check logs
tail -f logs/orchestrator.log

# Verify services
docker-compose ps

# Restore from checkpoint
python3 scripts/restore_checkpoint.py

# Resume
curl -X POST http://localhost:8000/projects/{id}/resume
```

### If Metrics Stuck Below Threshold
```bash
# Trigger manual optimization
curl -X POST http://localhost:8000/projects/{id}/optimize

# Force iteration
python3 scripts/force_iteration.py {project_id}
```

## References

- **Documentation**: `/docs/ARCHITECTURE.md`, `TECHNICAL_REFERENCE.md`, `README.md`
- **Inspiration**: Base44.com (all-in-one AI builder), Emergent.sh (multi-agent #1 on SWE-Bench)
- **API Docs**: Available at http://localhost:8000/docs when API is running

## GitHub Workflow & Auto-Sync 🐙

### Repository Information
- **GitHub Repository**: https://github.com/Ahmed-AdelB/myagent-continuous-ai-builder
- **Main Branch**: main 
- **Authentication**: GitHub CLI (gh) configured as Ahmed-AdelB

### Auto-Sync System
The project includes a complete auto-sync system that automatically commits and pushes changes to GitHub:

#### Manual Sync Commands
```bash
# Immediate sync to GitHub
./auto-sync.sh

# Quick sync alias (after running setup)
sync
```

#### Automatic Sync Features
1. **Post-Commit Hook**: Automatically pushes to GitHub after every `git commit`
2. **Manual Sync Script**: `./auto-sync.sh` for on-demand synchronization
3. **Shell Alias**: `sync` command for quick access
4. **Optional Cron Job**: Periodic sync every 30 minutes during work hours

#### Setup Scripts
```bash
# Initial auto-sync setup (already completed)
python setup_github_autosync.py

# Add sync command to shell
./setup-sync-command.sh

# Enable automatic periodic sync (optional)
./setup-auto-sync-cron.sh
```

### Git Workflow
```bash
# Standard git workflow with auto-push
git add .
git commit -m "Your commit message"
# Auto-push happens via post-commit hook

# Manual sync if needed
./auto-sync.sh
```

### Monitoring Sync Status
```bash
# Check git status
git status

# View recent commits
git log --oneline -5

# Check GitHub CLI authentication
gh auth status

# View cron jobs (if enabled)
crontab -l
```

### Sync Script Details
- **auto-sync.sh**: Main synchronization script
  - Detects changes with `git status --porcelain`
  - Commits with timestamp and Claude Code attribution
  - Pushes to `origin master:main`
  - Provides feedback on sync status

### Troubleshooting Sync Issues
```bash
# Re-authenticate GitHub CLI if needed
gh auth login

# Check remote repository connection
git remote -v

# Test manual push
git push origin master:main

# Recreate post-commit hook
chmod +x .git/hooks/post-commit

# View sync logs (if cron enabled)
grep "auto-sync" /var/log/cron
```

