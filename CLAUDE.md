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

vector_memory.py          # Semantic search (ChromaDB)
    └── store_memory()    # NOT store_embedding()

memory_orchestrator.py    # Coordinates all memory systems
```

### API Backend (api/)

FastAPI server on port 8000:

```python
main.py                   # Main FastAPI application
    ├── Endpoints use /projects/{id}/* pattern (NOT /api/*)
    ├── WebSocket: /ws/{project_id}
    └── Health check: /health

Critical: Frontend expects /api/* but backend serves /projects/*
         This is item #13 in COMPREHENSIVE_PLAN.md (pending fix)
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

# Run framework validation
python test_framework_validation.py

# View coverage report
open htmlcov/index.html
```

### Tri-Agent CLI Tools

```bash
# Aider (Codex GPT-5.1) - Note: Use --reasoning-effort NOT --max-reasoning
aider --model o1-preview --reasoning-effort high --yes <files>

# Gemini CLI
google-gemini generate --model gemini-2.5-pro --prompt "<prompt>"
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

**CRITICAL**: `api/auth.py` has eval() calls on lines 170, 217, 235
- **Never use eval()** - it's a Remote Code Execution vulnerability
- **Always use json.loads()** for deserializing Redis data
- This is item #17 in COMPREHENSIVE_PLAN.md (highest priority)

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

**Progress**: 14/22 items complete (64%)

**Critical Pending**:
- P3 #17: Fix eval() security vulnerability (HIGHEST PRIORITY)
- P2 #13: Fix API path mismatches (/api/* vs /projects/*)
- P3 #19: Complete empty test directories

**Completed**:
- ✅ All 8 P1 critical fixes
- ✅ Tri-agent SDLC infrastructure
- ✅ All 6 agents initialized properly
- ✅ Memory systems integrated

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

### Issue: Frontend components not found
**Cause**: Components were missing
**Fix**: Now implemented - ProjectManager.jsx, AgentMonitor.jsx, MetricsView.jsx

### Issue: API calls fail from frontend
**Cause**: Path mismatch - frontend uses /api/*, backend serves /projects/*
**Fix**: Pending (item #13 in plan)

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

# Tri-agent collaboration protocol
core/orchestrator/tri_agent_sdlc.py:68-150
    # SDLC phases with consensus voting

# Error learning system
core/memory/error_knowledge_graph.py:207-274
    # How the system learns from failures

# Agent base class
core/agents/base_agent.py:15-80
    # Foundation for all agents
```

## 🚨 Remember

1. **Never skip tri-agent approval** for code changes
2. **Never use eval()** - always use json.loads()
3. **Never forget to await async methods** (except ProjectLedger)
4. **Always check COMPREHENSIVE_PLAN.md** for current priorities
5. **Persistence is key** - checkpoint often, recover gracefully

---

*Last Updated: 2025-11-19*
*Current Phase: Completing comprehensive plan (64% done)*
*Next Priority: Fix eval() security vulnerability (P3 #17)*
