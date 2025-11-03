# CLAUDE.md - AI Assistant Guide for 22_MyAgent

## 🎯 Project Overview

This is the **Continuous AI App Builder** - a never-stopping AI development system that builds applications through persistent iteration until perfection is achieved. Unlike traditional "quick generators," this system employs a multi-agent architecture that continuously improves the application over days, weeks, or months.

## ⚠️ Critical Understanding

**This is NOT a 5-minute app generator!** This is a tireless AI development team that:
- Never stops working until the app is perfect
- Learns from every error and builds solution patterns
- Remembers all context across sessions
- Continuously iterates and improves

## 🏗️ System Architecture

### Core Components

1. **Continuous Director** (`core/orchestrator/continuous_director.py`)
   - Master orchestrator that never stops
   - Manages iteration cycles and quality metrics
   - Coordinates all agents and tracks progress

2. **Persistent Memory Systems**
   - **Project Ledger** (`core/memory/project_ledger.py`): Complete version history
   - **Error Knowledge Graph** (`core/memory/error_knowledge_graph.py`): Learns from failures
   - **Vector Memory** (`core/memory/vector_memory.py`): Semantic understanding

3. **Multi-Agent Team** (`core/agents/`)
   - Coder Agent: Implements features
   - Tester Agent: Generates and runs tests
   - Debugger Agent: Analyzes failures
   - Architect Agent: Reviews system design
   - Analyzer Agent: Monitors metrics
   - UI Refiner Agent: Improves UX

4. **Learning Engine** (`core/learning/`)
   - Pattern recognition from successes
   - Error solution mapping
   - Adaptive behavior improvement

## 📋 Current Implementation Status

### ✅ Completed
- Basic folder structure
- Python virtual environment setup
- Continuous orchestrator system
- Persistent memory system (Project Ledger)
- Error Knowledge Graph
- Milestone Tracker
- Comprehensive documentation

### 🚧 Pending Tasks
1. Build the base persistent agent class
2. Create individual agents (Coder, Tester, Debugger)
3. Build the learning engine
4. Create FastAPI backend server
5. Set up React frontend with dashboard
6. Implement continuous development workflow
7. Create Docker configuration
8. Test the continuous development system

## 🛠️ Development Guidelines

### When Working on This Project

1. **Understand the Philosophy**: This system continuously works until perfection. Every component should support this goal.

2. **Maintain Persistence**: All state, decisions, and learning must persist across sessions.

3. **Quality Metrics Are Sacred**: The system works until these are achieved:
   - Test Coverage ≥ 95%
   - Critical Bugs = 0
   - Performance Score ≥ 90%
   - Documentation Coverage ≥ 90%
   - Security Score ≥ 95%

### Key Files to Understand

```python
# The main orchestration loop
core/orchestrator/continuous_director.py:146-157
# This is the heart of the continuous development philosophy

# Error learning system
core/memory/error_knowledge_graph.py:89-134
# Critical for the system to learn from failures

# Version control system
core/memory/project_ledger.py:161-229
# Maintains complete history of all changes
```

## 🚀 Running the System

### Prerequisites
```bash
# Python 3.11+, Node.js 18+, PostgreSQL 15+, Redis 7+
cd /Users/aadel/22_MyAgent
```

### Start Development
```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Start the orchestrator (this never stops!)
python -m core.orchestrator.continuous_director

# 3. Start API server
uvicorn api.main:app --reload --port 8000

# 4. Start frontend
cd frontend && npm run dev
```

## 💡 Implementation Tips

### For Agent Development
- All agents must inherit from `PersistentAgent` base class
- Implement checkpoint/recovery mechanisms
- Log all decisions to the Project Ledger
- Report errors to the Error Knowledge Graph

### For Memory Systems
- Use SQLite for structured data (Project Ledger)
- Use ChromaDB for vector embeddings
- Implement proper transaction handling
- Always save state before potentially failing operations

### For Learning Engine
- Build on the Error Knowledge Graph
- Implement pattern matching for similar errors
- Create solution templates that can be reused
- Track success rates of different approaches

## 🔄 Continuous Development Workflow

The system follows this infinite loop:
```python
while not app.is_perfect():
    plan_iteration()
    implement_features()
    run_tests()
    debug_failures()
    optimize_performance()
    validate_quality()
    learn_from_iteration()
    checkpoint_progress()
    integrate_feedback()
```

## 📊 Monitoring Progress

- Check `persistence/milestones_*.json` for milestone tracking
- Review `persistence/database/*_ledger.db` for complete history
- Monitor `logs/` directory for detailed execution logs
- Dashboard at `http://localhost:5173` (when frontend is running)

## 🐛 Common Issues and Solutions

### Issue: System stops unexpectedly
- Check checkpoint files in `persistence/checkpoints/`
- Resume from last checkpoint using `CheckpointManager.load_latest()`

### Issue: Memory grows too large
- Run `ProjectLedger.cleanup_old_versions()`
- Compress older iterations in the database

### Issue: Agents not coordinating
- Check Redis connection for message queue
- Verify all agents are registered with the orchestrator

## 🔑 Important Environment Variables

```bash
# Required API Keys (in .env file)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Database Configuration
POSTGRES_URL=postgresql://localhost/myagent_db
REDIS_URL=redis://localhost:6379

# Development Settings
DEV_MODE=true
MAX_ITERATIONS=1000
CHECKPOINT_INTERVAL=10
```

## 📝 Next Steps for Implementation

1. **Priority 1: Base Agent Class**
   - Create `core/agents/base_agent.py`
   - Implement checkpoint/recovery
   - Add communication with orchestrator

2. **Priority 2: Coder Agent**
   - Implement code generation capabilities
   - Integration with LangChain
   - Version control integration

3. **Priority 3: Testing Infrastructure**
   - Set up pytest framework
   - Create test generation templates
   - Implement coverage tracking

## 🤝 Contributing Guidelines

- Every feature must support continuous operation
- All state must be persistable
- Errors should be learning opportunities
- Code quality must improve with each iteration
- Document all architectural decisions

## 📚 References

- Inspired by Base44.com architecture
- Multi-agent concepts from Emergent.sh
- Uses `multi-agent-generator` Python package
- Continuous improvement philosophy from DevOps practices

## ⚡ Quick Commands Reference

```bash
# Check system status
python -m core.orchestrator.status

# View current iteration
python -m core.utils.show_iteration

# Analyze error patterns
python -m core.memory.analyze_errors

# Generate progress report
python -m core.reports.generate_progress
```

## 🎯 Remember

This system is designed to **never give up**. It will continue working, learning, and improving until the application meets all quality criteria. Every error is a learning opportunity, every iteration brings improvement, and persistence is the key to perfection.

---

*Last Updated: 2025-11-03*
*Project Started: 2025-11-03*
*Current Status: Core systems implemented, agents pending*