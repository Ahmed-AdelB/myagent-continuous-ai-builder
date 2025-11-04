# 🔬 Technical Reference Manual - MyAgent Continuous AI Builder

## For Engineers & AI Agents Continuing Development

---

## 🎯 Critical Understanding

**THIS SYSTEM NEVER STOPS.** It will continue iterating until all metrics achieve perfection. Your role is to maintain and enhance this continuous operation.

---

## 📊 Current System State

### Latest Fixes Applied (November 2024)
```python
# Critical fixes implemented:
1. VectorMemory.__init__(project_name) ✅
2. QualityMetrics.to_dict() ✅
3. milestone_tracker initialization ✅
4. progress_analyzer initialization ✅
5. WebSocket dual endpoints (/ws and /ws/{project_id}) ✅
6. Agent initialization with orchestrator ✅
7. Continuous quality monitor loop ✅
8. Self-healing triggers ✅
```

### Known Working Configuration
```python
# Verified working versions
Python: 3.11.x or 3.12.x
Node.js: 18.x or 20.x
PostgreSQL: 15.x
Redis: 7.x
LangChain: 1.0+ (langchain-openai, langchain-core)
FastAPI: Latest
React: 18.x
```

---

## 🔧 Core System Mechanics

### 1. The Continuous Loop
```python
# Location: core/orchestrator/continuous_director.py

async def start(self):
    """The eternal loop that never stops"""
    while not self.metrics.is_perfect():
        # This loop continues forever until perfection
        await self._execute_iteration()

        # Quality monitor runs in parallel
        # Checks every 5 minutes (300 seconds)
        # Triggers optimizations automatically
```

### 2. Perfection Criteria
```python
def is_perfect(self) -> bool:
    """System stops ONLY when this returns True"""
    return (
        self.test_coverage >= 95.0 and          # Must have 95% test coverage
        self.bug_count_critical == 0 and        # Zero critical bugs allowed
        self.bug_count_minor <= 5 and           # Max 5 minor bugs
        self.performance_score >= 90.0 and      # 90% performance minimum
        self.documentation_coverage >= 90.0 and # 90% docs minimum
        self.code_quality_score >= 85.0 and     # 85% quality minimum
        self.user_satisfaction >= 90.0 and      # 90% satisfaction
        self.security_score >= 95.0             # 95% security minimum
    )
```

### 3. Self-Healing Mechanisms
```python
# Automatic triggers when metrics drop:

async def continuous_quality_monitor(self):
    """Runs every 5 minutes checking all metrics"""

    if metrics["test_coverage"] < 95:
        await self.trigger_test_intensification()

    if metrics["bug_count_critical"] > 0:
        await self.emergency_debug_mode()  # HIGHEST PRIORITY

    if metrics["performance_score"] < 90:
        await self.trigger_performance_optimization()
```

---

## 🏗️ Component Architecture

### Memory Systems Hierarchy
```
VectorMemory (Semantic Search)
    ├── Code memories
    ├── Decision memories
    ├── Error memories
    ├── Context memories
    └── Pattern memories

ProjectLedger (Version Control)
    ├── Code versions
    ├── Change history
    ├── Iteration summaries
    └── Rollback points

ErrorKnowledgeGraph (Learning)
    ├── Error patterns
    ├── Solution mappings
    ├── Success strategies
    └── Prevention rules
```

### Agent Communication Protocol
```python
# Agents communicate through orchestrator
orchestrator.agents["coder"] -> task_queue -> orchestrator
orchestrator -> agents["tester"] -> results -> orchestrator

# Direct agent-to-agent is discouraged
# All communication flows through orchestrator for tracking
```

---

## 💻 Code Patterns & Standards

### 1. Agent Implementation Pattern
```python
class NewAgent(PersistentAgent):
    """Standard agent implementation pattern"""

    def __init__(self, orchestrator=None):
        super().__init__(
            name="agent_name",
            role="Agent Role",
            capabilities=["cap1", "cap2"],
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

### 2. Memory Access Pattern
```python
# Always use context manager for memory operations
async with self.orchestrator.vector_memory as memory:
    # Store new memory
    memory.store_memory(
        content="Solution to authentication bug",
        memory_type="solutions",
        metadata={"iteration": self.iteration}
    )

    # Retrieve relevant memories
    relevant = memory.search_memories(
        query="authentication error",
        memory_type="errors",
        top_k=5
    )
```

### 3. Error Handling Pattern
```python
# All errors must be learned from
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

---

## 🚨 Critical Files & Their Purposes

### Core Files (DO NOT DELETE)
```bash
core/orchestrator/continuous_director.py  # Main control loop
core/orchestrator/milestone_tracker.py    # Progress tracking
core/orchestrator/progress_analyzer.py    # Metric analysis

core/memory/project_ledger.py            # Version history
core/memory/vector_memory.py             # Semantic memory
core/memory/error_knowledge_graph.py     # Error learning

core/agents/base_agent.py                # Agent foundation
core/agents/coder_agent.py               # Code generation
core/agents/tester_agent.py              # Test creation
core/agents/debugger_agent.py            # Bug fixing
core/agents/architect_agent.py           # System design
core/agents/analyzer_agent.py            # Performance
core/agents/ui_refiner_agent.py          # UX improvement
```

### Configuration Files
```bash
.env                                      # API keys (CRITICAL)
requirements.txt                          # Python deps (EXACT VERSIONS)
package.json                             # Frontend deps
.github/workflows/continuous_quality.yml # CI/CD pipeline
```

### Persistence (MUST PRESERVE)
```bash
persistence/
├── database/         # SQLite databases
├── vector_memory/    # ChromaDB data
├── checkpoints/      # System state
├── agents/          # Agent states
└── knowledge_graph/ # Error patterns
```

---

## 🔍 Debugging & Monitoring

### Key Log Locations
```python
# Orchestrator logs
tail -f logs/orchestrator.log

# Agent-specific logs
tail -f logs/agents/coder_agent.log
tail -f logs/agents/debugger_agent.log

# API logs
journalctl -u myagent-api -f

# Error patterns
sqlite3 persistence/knowledge_graph/errors.db "SELECT * FROM errors ORDER BY timestamp DESC LIMIT 10;"
```

### Performance Monitoring
```python
# Check current metrics
curl http://localhost:8000/projects/{id}/metrics | jq

# Monitor iteration progress
watch -n 5 'curl -s http://localhost:8000/projects/{id} | jq .iteration'

# Check agent status
curl http://localhost:8000/projects/{id}/agents | jq
```

### Common Debug Commands
```python
# Test orchestrator initialization
python3 -c "
from core.orchestrator.continuous_director import ContinuousDirector
o = ContinuousDirector('test', {})
print('✅ Orchestrator OK')
"

# Test memory systems
python3 -c "
from core.memory.vector_memory import VectorMemory
vm = VectorMemory('test')
print('✅ VectorMemory OK')
"

# Test agent initialization
python3 -c "
from core.agents.coder_agent import CoderAgent
agent = CoderAgent()
print('✅ Agents OK')
"
```

---

## 🛠️ Task Types & Priorities

### Task Structure
```python
@dataclass
class DevelopmentTask:
    id: str                    # Unique identifier
    type: str                  # Task category
    description: str           # What to do
    priority: int             # 1-10 (10 highest)
    assigned_agent: str       # Target agent
    data: Dict               # Task-specific data
    dependencies: List[str]  # Other task IDs
    status: TaskStatus       # pending/active/completed/failed
```

### Task Types by Agent

#### Coder Agent Tasks
```python
"implement_feature"      # Add new functionality
"refactor_code"         # Improve code structure
"add_documentation"     # Write docs
"fix_syntax_error"      # Correct code errors
```

#### Tester Agent Tasks
```python
"generate_unit_tests"    # Create unit tests
"generate_integration_tests" # Integration tests
"measure_coverage"       # Calculate coverage
"test_performance"       # Performance tests
```

#### Debugger Agent Tasks
```python
"fix_bug"               # Fix identified bug
"analyze_error"         # Understand error
"emergency_debug"       # Critical bug fix (P10)
"prevent_regression"    # Add regression tests
```

---

## 📈 Optimization Strategies

### When Test Coverage < 95%
```python
# System automatically:
1. Identifies untested code paths
2. Generates targeted test cases
3. Focuses on critical paths first
4. Adds edge case tests
5. Implements property-based tests
```

### When Performance < 90%
```python
# System automatically:
1. Profiles code execution
2. Identifies bottlenecks
3. Optimizes database queries
4. Implements caching
5. Parallelizes operations
```

### When Critical Bugs > 0
```python
# EMERGENCY MODE ACTIVATED:
1. All other tasks suspended
2. Full system analysis
3. Root cause identification
4. Immediate fix deployment
5. Regression test addition
```

---

## 🔄 Workflow for New Features

### 1. Requirement Analysis
```python
architect_agent.analyze_requirements(feature_spec)
→ Creates design document
→ Identifies dependencies
→ Estimates complexity
```

### 2. Implementation
```python
coder_agent.implement_feature(design_doc)
→ Generates code
→ Adds inline documentation
→ Creates initial structure
```

### 3. Testing
```python
tester_agent.generate_tests(implementation)
→ Unit tests
→ Integration tests
→ Edge cases
```

### 4. Optimization
```python
analyzer_agent.optimize(implementation)
→ Performance profiling
→ Security audit
→ Code quality check
```

### 5. Refinement
```python
ui_refiner_agent.improve_ux(implementation)
→ UI/UX improvements
→ Accessibility
→ User feedback integration
```

---

## ⚡ Performance Benchmarks

### Expected Performance
```python
# Initialization times
Orchestrator init: < 2 seconds
Agent init: < 0.5 seconds each
Memory load: < 1 second

# Operation times
Task execution: < 30 seconds average
Iteration cycle: < 5 minutes
Quality check: < 10 seconds

# Resource usage
RAM: < 4GB normal, < 8GB peak
CPU: < 50% average, < 80% peak
Disk I/O: < 100 MB/s
```

### Scaling Considerations
```python
# Horizontal scaling
- Run agents on separate machines
- Use Redis for task queue
- Distribute memory across nodes

# Vertical scaling
- Increase RAM for larger projects
- Use GPU for ML operations
- SSD for database storage
```

---

## 🔐 Security Considerations

### API Key Management
```python
# Never commit keys to repository
# Use environment variables
# Rotate keys regularly
# Monitor usage

# Key rotation script
scripts/rotate_api_keys.py
```

### Database Security
```sql
-- Regular backups
pg_dump myagent_db > backup_$(date +%Y%m%d).sql

-- Access control
REVOKE ALL ON DATABASE myagent_db FROM PUBLIC;
GRANT CONNECT ON DATABASE myagent_db TO myagent_user;
```

### Code Security
```python
# All generated code is scanned for:
- SQL injection vulnerabilities
- XSS vulnerabilities
- Insecure dependencies
- Hardcoded secrets
- Security score must be ≥ 95%
```

---

## 📝 Commit Standards

### Commit Message Format
```bash
<type>(<scope>): <subject>

<body>

<footer>

# Types
feat: New feature
fix: Bug fix
perf: Performance improvement
test: Test addition
docs: Documentation
refactor: Code refactoring

# Example
feat(coder-agent): Add automatic import optimization

Implements smart import detection and optimization
to reduce unnecessary imports and improve performance.

Closes #123
```

### Pre-commit Checks
```bash
# Automated checks before commit
1. Test coverage ≥ 95%
2. No critical bugs
3. Linting passes
4. Type checking passes
5. Security scan passes
```

---

## 🚀 Advanced Features

### Memory Consolidation
```python
# Runs periodically to optimize memory
vector_memory.consolidate_memories(
    memory_type="code",
    similarity_threshold=0.9
)
```

### Knowledge Transfer
```python
# Export learning for new projects
error_graph.export_knowledge("project1_knowledge.pkl")
new_project.error_graph.import_knowledge("project1_knowledge.pkl")
```

### Checkpoint Recovery
```python
# System can resume from any checkpoint
orchestrator.restore_checkpoint(checkpoint_id)
# Continues from exact state
```

---

## 📞 Emergency Procedures

### If System Stops Unexpectedly
```bash
1. Check logs: tail -f logs/orchestrator.log
2. Verify services: systemctl status myagent-*
3. Check database: psql -c "SELECT * FROM projects;"
4. Restore checkpoint: python3 scripts/restore_checkpoint.py
5. Resume: curl -X POST http://localhost:8000/projects/{id}/resume
```

### If Metrics Stuck Below Threshold
```bash
1. Trigger manual optimization:
   curl -X POST http://localhost:8000/projects/{id}/optimize

2. Clear task queue:
   redis-cli FLUSHDB

3. Reset metrics:
   python3 scripts/reset_metrics.py {project_id}

4. Force iteration:
   python3 scripts/force_iteration.py {project_id}
```

---

## 🎓 Learning Resources

### For Human Engineers
- Study `core/orchestrator/continuous_director.py` - The heart of the system
- Understand `core/memory/error_knowledge_graph.py` - How it learns
- Review `tests/test_continuous_system.py` - Test patterns

### For AI Agents
- Access context: `vector_memory.get_context_window()`
- Learn patterns: `error_graph.get_successful_patterns()`
- Review history: `project_ledger.get_iteration_history()`

---

**Remember: This system represents a new paradigm in software development - autonomous, continuous, and self-improving. It will not stop until perfection is achieved.**

---

*Technical Reference v1.0.0*
*Last Updated: November 2024*
*Next Review: When metrics < 100%*