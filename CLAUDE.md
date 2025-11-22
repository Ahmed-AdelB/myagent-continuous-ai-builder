# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Project Overview

The **Continuous AI App Builder** is a never-stopping AI development system that builds applications through persistent iteration until perfection is achieved. It employs a tri-agent architecture (Claude + Codex + Gemini) that continuously improves applications over extended periods.

**Critical**: This is NOT a quick generator - it's a tireless AI development team that never gives up until quality metrics are met.

## 🏗️ System Architecture

### Core Orchestration Flow

```
Continuous Director (core/orchestrator/continuous_director.py)
    ├── Manages infinite iteration loop until perfection
    ├── Coordinates 6 specialized agents
    └── Enforces quality gates (95% coverage, 0 critical bugs, etc.)

Tri-Agent SDLC (core/orchestrator/tri_agent_sdlc.py)
    ├── Claude Code (Sonnet 4.5): Requirements & Integration
    ├── Aider/Codex (GPT-5.1): Code Implementation
    └── Gemini (2.5/3.0 Pro): Code Review & Approval
    └── Requires 3/3 consensus for phase transitions
```

### Agent Architecture (core/agents/)

All agents inherit from `PersistentAgent` base class:

```python
base_agent.py              # Base class with checkpoint/recovery
    ├── async def initialize()      # Called before agent starts
    ├── async def process_task()    # Main task processing (NOT execute_task)
    └── async def _custom_init()    # Subclass initialization hook

Specialized Agents:
├── coder_agent.py         # Implements features (uses core/utils/filesystem.py)
├── tester_agent.py        # Generates/runs tests (uses LangChain + ChatOpenAI)
├── debugger_agent.py      # Analyzes failures
├── architect_agent.py     # Reviews design
├── analyzer_agent.py      # Monitors metrics
└── ui_refiner_agent.py    # Improves UX

CLI Agent Wrappers (core/agents/cli_agents/):
├── aider_codex_agent.py   # Wraps `aider` CLI for GPT-5.1
├── gemini_cli_agent.py    # Wraps `google-gemini` CLI
└── claude_code_agent.py   # Self-referential Claude agent
```

### Memory Systems (core/memory/)

```python
project_ledger.py          # Complete version history (SQLite)
    └── record_version()   # SYNCHRONOUS - do not await

error_knowledge_graph.py   # Learns from failures (NetworkX + SQLite)
    ├── add_error()        # Returns ErrorNode
    ├── add_solution()     # Links solution to error
    └── export_graph()     # Returns dict (not None)

vector_memory.py          # Semantic search (ChromaDB) + Cache Eviction
    ├── store_memory()    # NOT store_embedding()
    ├── evict_old_memories()        # Age-based eviction (90 days)
    ├── evict_by_collection_size()  # Size-based LRU eviction
    └── auto_evict_all_collections() # Automatic cleanup

cache_eviction.py         # Cache management (NEW)
    ├── LRU, LFU, TTL, HYBRID policies
    ├── Max 10k entries, 500 MB limit
    └── Prevents memory leaks in 24/7 operation

memory_orchestrator.py    # Coordinates all memory systems
```

### API Backend (api/)

FastAPI server on port 8000:

```python
main.py                   # Main FastAPI application
    ├── Endpoints use /projects/{id}/* pattern
    ├── WebSocket: /ws/{project_id}
    └── Health check: /health

✅ FIXED: Frontend now uses /projects/{id}/* paths via ProjectContext
```

### Frontend (frontend/)

React application on port 5173:

```javascript
App.jsx (canonical)       # Main app (App.tsx was deleted - conflict resolved)
components/
    ├── Dashboard.jsx     # Main dashboard
    ├── ProjectManager.jsx # Project CRUD
    ├── AgentMonitor.jsx  # Real-time agent status
    ├── MetricsView.jsx   # Metrics visualization
    ├── MetricsPanel.jsx  # Metrics display
    ├── ErrorAnalytics.jsx # Error analysis
    └── IterationHistory.jsx # Iteration timeline
```

## 🚀 Development Commands

### Environment Setup

```bash
# Activate virtual environment (ALWAYS do this first)
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Running the System

```bash
# 1. Start API server (port 8000)
uvicorn api.main:app --reload --port 8000

# 2. Start frontend (port 5173)
cd frontend && npm run dev

# 3. Access dashboard
# http://localhost:5173
```

### Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov --cov-report=html

# Run specific test directory
pytest tests/test_agents/ -v
pytest tests/test_memory/ -v
pytest tests/test_orchestrator/ -v
pytest tests/test_learning/ -v          # NEW: Pattern recognition, RL
pytest tests/test_persistence/ -v       # NEW: Project ledger, memory orchestrator
pytest tests/test_recovery/ -v          # NEW: Checkpoints, error recovery

# Run single test file
pytest tests/test_learning/test_pattern_recognition.py -v

# Run framework validation
python test_framework_validation.py

# View coverage report
open htmlcov/index.html
```

### Database Migrations

```bash
# Run migrations (Alembic is single source of truth)
alembic upgrade head

# Check current migration version
alembic current

# View migration history
alembic history

# Create new migration
alembic revision --autogenerate -m "Description"

# See alembic/README.md for complete migration guide
```

### Tri-Agent CLI Tools

**⚠️ IMPORTANT**: Use `codex` command (NOT `aider`)!

```bash
# Codex (GPT-5.1-Codex-Max) - Primary code implementation agent
codex exec --message "<task description>" <files>
codex exec --message "Implement feature X" src/component.py

# Gemini CLI - Security review and strategic analysis
google-gemini generate --model gemini-2.5-pro --prompt "<prompt>"

# Note: 'aider' was previously used but is now deprecated in favor of 'codex'
```

### Git Workflow

```bash
# Standard commit with tri-agent approval
git commit -m "feat: <description>

🤖 Tri-Agent Approval:
✅ Claude Code (Sonnet 4.5): APPROVE
✅ Codex (GPT-5.1): APPROVE
✅ Gemini (2.5 Pro): APPROVE

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## 🔑 Critical Implementation Rules

### Agent Development

1. **Method Naming**: Agents implement `process_task()` NOT `execute_task()`
2. **Initialization**: All agents MUST implement `async def initialize()`
3. **Callbacks**: Use `orchestrator.on_agent_task_complete(agent_id, task)`
4. **Inter-agent**: Use `orchestrator.route_message(message_dict)`

### Memory System Integration

```python
# CORRECT usage patterns:

# ErrorKnowledgeGraph - no constructor params
error_graph = ErrorKnowledgeGraph()  # NOT ErrorKnowledgeGraph(project_name)

# Add error then solution (2-step process)
error_node = error_graph.add_error(...)
error_graph.add_solution(error_id=error_node.id, ...)

# ProjectLedger - synchronous methods
ledger.record_decision(...)  # Do NOT await

# VectorMemory
vector_memory.store_memory(...)  # NOT store_embedding()
```

### Filesystem Operations

```python
# Use async utilities from core/utils/filesystem.py
from core.utils.filesystem import list_directory, read_file, write_file

# All are async
files = await list_directory(path, recursive=True)
content = await read_file(file_path)
await write_file(file_path, content)
```

### Security

✅ **FIXED**: `api/auth.py` eval() RCE vulnerability patched
- All eval() calls replaced with json.loads()
- Never use eval() - always use json.loads() for deserializing data

## 📋 Quality Gates (Never Compromise)

The system works until ALL these are met:

```python
QualityMetrics:
    test_coverage >= 95.0%
    bug_count_critical == 0
    bug_count_minor <= 5
    performance_score >= 90.0%
    documentation_coverage >= 90.0%
    code_quality_score >= 85.0%
    security_score >= 95.0%
```

## 🔄 Continuous Development Loop

```python
while not app.is_perfect():
    # Phase 1: Planning
    self.state = ProjectState.PLANNING
    tasks = await self._plan_iteration()

    # Phase 2: Development
    self.state = ProjectState.DEVELOPING
    await self._execute_development_tasks(tasks)

    # Phase 3: Testing
    self.state = ProjectState.TESTING
    test_results = await self._run_tests()

    # Phase 4: Debugging (if failures)
    if test_results.get("failures"):
        self.state = ProjectState.DEBUGGING
        await self._debug_and_fix(test_results["failures"])

    # Phase 5: Optimization
    self.state = ProjectState.OPTIMIZING
    await self._optimize_performance()

    # Phase 6: Validation
    self.state = ProjectState.VALIDATING
    await self._validate_quality()

    # Phase 7: Learning
    await self._learn_from_iteration()

    # Checkpoint every hour
    if datetime.now() - self.last_checkpoint > self.checkpoint_interval:
        await self._create_checkpoint()
```

## 🎯 Current Implementation Status

See `COMPREHENSIVE_PLAN.md` for detailed status.

**Progress**: 18/26 items complete (69%)

**Recently Completed**:
- ✅ P2 #13: API path mismatches fixed (frontend uses ProjectContext)
- ✅ P3 #17: eval() RCE vulnerability patched
- ✅ P3 #19: Test directories filled (1,377 lines of tests)
- ✅ P2 #15: Schema management unified (Alembic single source of truth)
- ✅ P2 #16: Cache eviction policies implemented (LRU/LFU/TTL/Hybrid)

**Current Work**:
- 🔄 P3 #18: Add transaction boundaries for database consistency
- 📋 P3 #20: Frontend component tests (Jest/RTL)
- 📋 P3 #22: Documentation coverage (docstrings)
- 📋 Phase 4: Comprehensive test suite validation

**Foundation Complete**:
- ✅ All 8 P1 critical fixes
- ✅ Tri-agent SDLC infrastructure
- ✅ All 6 agents initialized properly
- ✅ Memory systems with cache eviction
- ✅ Guardrail system for safe autonomous operation

## 📁 Persistence Locations

```
persistence/
├── database/
│   ├── *_ledger.db          # SQLite databases
│   └── error_knowledge.db   # Error learning
├── snapshots/
│   └── checkpoint_*.json    # Iteration checkpoints
├── storage/
│   └── vector_db/           # ChromaDB embeddings
└── milestones_*.json        # Milestone tracking
```

## 🐛 Common Issues

### Issue: Agent task execution fails
**Cause**: Called `execute_task()` instead of `process_task()`
**Fix**: Use `await agent.process_task(task)` (line 337 in continuous_director.py)

### Issue: Memory system integration errors
**Cause**: Method signature mismatches
**Fix**: Check COMPREHENSIVE_PLAN.md items #5, #11 for correct patterns

### Issue: Memory leaks during long-running operation
**Cause**: No cache eviction configured
**Fix**: VectorMemory now includes automatic eviction. Call `auto_evict_all_collections()` periodically

### Issue: Database schema drift
**Cause**: Schema defined in multiple places
**Fix**: ✅ FIXED - Alembic is now single source of truth. Use `alembic upgrade head` instead of raw SQL

## 🔧 Debugging

```bash
# Check orchestrator logs
tail -f logs/orchestrator.log

# Inspect checkpoints
ls -lt persistence/snapshots/

# Examine error knowledge
sqlite3 persistence/database/error_knowledge.db "SELECT * FROM errors;"

# View project history
sqlite3 persistence/database/*_ledger.db "SELECT * FROM versions ORDER BY timestamp DESC LIMIT 10;"
```

## 📚 Key Files to Understand

```python
# Orchestration core (the heart of the system)
core/orchestrator/continuous_director.py:174-295
    # Main loop that never stops
    # Integrates guardrails and task validation

# Tri-agent collaboration protocol
core/orchestrator/tri_agent_sdlc.py:68-150
    # SDLC phases with consensus voting

# Guardrails for safe autonomous operation
core/orchestrator/guardrails.py
    # Blocks dangerous operations (eval, rm -rf, force push, DROP DATABASE)
    # Risk assessment: SAFE → CRITICAL
    # Audit trail for all operations

# Task validation (prevents false completion claims)
core/orchestrator/task_validator.py
    # Validates tasks truly complete with proof
    # File existence, syntax validation, test execution

# Cache eviction (prevents memory leaks)
core/memory/cache_eviction.py
    # LRU, LFU, TTL, HYBRID policies
    # Max 10k entries, 500 MB limit

# Error learning system
core/memory/error_knowledge_graph.py:207-274
    # How the system learns from failures

# Agent base class
core/agents/base_agent.py:15-80
    # Foundation for all agents

# Database migrations (single source of truth)
alembic/versions/0001_init.py
    # Complete schema definition
    # See alembic/README.md for workflow
```

## 🚨 Remember

1. **Never skip tri-agent approval** for code changes
2. **Never use eval()** - always use json.loads()
3. **Never forget to await async methods** (except ProjectLedger)
4. **Always check COMPREHENSIVE_PLAN.md** for current priorities
5. **Persistence is key** - checkpoint often, recover gracefully
6. **Memory management** - Call `auto_evict_all_collections()` periodically to prevent leaks
7. **Database changes** - Use Alembic migrations, not raw SQL in code
8. **Guardrails active** - Dangerous operations blocked in autonomous mode

---

*Last Updated: 2025-11-19*
*Current Phase: Completing comprehensive plan (69% done)*
*Progress: 18/26 items complete*
*Next: Transaction boundaries + Frontend tests + Documentation*
