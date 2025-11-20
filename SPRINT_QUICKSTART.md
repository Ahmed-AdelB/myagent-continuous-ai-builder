# Sprint Quickstart Guide

## 🚀 Getting Started with Sprint 1

### Prerequisites
```bash
# Ensure you're in the project directory
cd /home/aadel/projects/22_MyAgent

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Sprint 1 Kickoff Checklist

#### Week 1: Foundation (Days 1-7)

**Day 1-2: RAG Architecture Specification (#18)**
```bash
# Create documentation directory
mkdir -p docs/architecture

# Start working on RAG spec
gh issue view 18
# Assign to yourself
gh issue edit 18 --add-assignee @me
```

**Key Deliverables**:
- [ ] RAG architecture diagram
- [ ] Corpus definition (code + docs + tests)
- [ ] Chunking strategy specification
- [ ] Embeddings model selection (OpenAI text-embedding-3-small)
- [ ] Vector store config (ChromaDB)

**Day 2-3: Conflict Resolution Protocol (#1)**
```bash
gh issue view 1
gh issue edit 1 --add-assignee @me

# Create implementation file
touch core/orchestrator/conflict_resolver.py
```

**Key Deliverables**:
- [ ] ConflictResolver class
- [ ] Lead agent assignment logic
- [ ] Human escalation workflow
- [ ] Audit trail implementation

**Day 3-5: Security & Supply Chain (#2)**
```bash
gh issue view 2
gh issue edit 2 --add-assignee @me

# Install security tools
pip install cyclonedx-bom bandit gitleaks pre-commit
```

**Key Deliverables**:
- [ ] SBOM generation in CI
- [ ] Vulnerability scanning (trivy)
- [ ] Secret scanning (gitleaks)
- [ ] Pre-commit hooks configured

**Day 5-7: Observability Infrastructure (#19)**
```bash
gh issue view 19
gh issue edit 19 --add-assignee @me

# Install observability dependencies
pip install structlog opentelemetry-api opentelemetry-sdk
```

**Key Deliverables**:
- [ ] Structured JSON logging
- [ ] OpenTelemetry traces
- [ ] Metrics collection (latency, success rate, cost)

#### Week 2: RAG Implementation (Days 8-14)

**Day 8-12: RAG Retriever (#3) - BLOCKS EVERYTHING**
```bash
gh issue view 3
gh issue edit 3 --add-assignee @me

# Install RAG dependencies
pip install tree-sitter tree-sitter-python chromadb openai sentence-transformers

# Create RAG module
mkdir -p core/knowledge
touch core/knowledge/rag_retriever.py
touch core/knowledge/__init__.py
```

**Implementation Checklist**:
```python
# core/knowledge/rag_retriever.py
class RAGRetriever:
    # TODO: Implement based on spec from #18
    def __init__(self, collection_name: str):
        pass
    
    async def index_codebase(self, root_path: Path):
        # Parse with tree-sitter
        # Generate embeddings
        # Store in ChromaDB
        pass
    
    async def retrieve(self, query: str, k: int = 5):
        # Semantic search
        # Return relevant chunks
        pass
```

**Day 12-14: RAG Integration (#4)**
```bash
gh issue view 4
gh issue edit 4 --add-assignee @me

# Modify tri_agent_sdlc.py
code core/orchestrator/tri_agent_sdlc.py
```

**Integration Points**:
- [ ] Replace 1M token dumps in `_get_codebase_context()`
- [ ] Add fallback to full dump if RAG fails
- [ ] Measure token reduction (target: >70%)

**Day 10-14: Task Ledger (#5) - PARALLEL WORK**
```bash
gh issue view 5
gh issue edit 5 --add-assignee @me

touch core/orchestrator/task_ledger.py
```

**State Machine**:
```python
class TaskState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRY = "retry"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
```

#### Week 3: Routing & CI/CD (Days 15-21)

**Day 15-18: Enhanced Routing (#6)**
```bash
gh issue view 6
gh issue edit 6 --add-assignee @me

touch core/orchestrator/task_router.py
```

**Routing Formula**:
```python
fitness_score = (
    0.5 * capability_match +
    0.3 * (1 - normalized_load) +
    0.2 * historical_win_rate
)
```

**Day 18-21: CI/CD Pipeline (#7)**
```bash
gh issue view 7
gh issue edit 7 --add-assignee @me

# Create GitHub Actions workflow
mkdir -p .github/workflows
touch .github/workflows/ci.yml
touch .github/workflows/nightly.yml
```

**Pipeline Stages**:
- [ ] Pre-commit: ruff, mypy, bandit
- [ ] PR CI: unit tests, coverage gate
- [ ] Nightly: integration tests, chaos tests

## 📊 Sprint 1 Progress Tracking

### Update Issues Regularly
```bash
# Mark issue as in progress
gh issue edit <number> --add-label "status:in-progress"

# Add comments with progress updates
gh issue comment <number> --body "Completed RAG spec. Moving to implementation."

# Close issue when done (requires PR with tri-agent approval)
gh issue close <number> --comment "Completed via PR #X"
```

### Daily Standup Template
```markdown
## Daily Update - YYYY-MM-DD

**Yesterday**:
- Completed X
- Made progress on Y

**Today**:
- Working on issue #N
- Pairing with Agent X on issue #M

**Blockers**:
- Need design review for Z
- Waiting on dependency A
```

## 🎯 Sprint 1 Success Criteria

Before moving to Sprint 2, verify:

- [ ] All 11 Sprint 1 issues closed
- [ ] RAG retrieval working (<200ms p95)
- [ ] Conflict resolution tested with 10+ scenarios
- [ ] Security scans passing in CI
- [ ] Observability dashboard showing metrics
- [ ] Task ledger handling retries/timeouts
- [ ] Routing using fitness scores
- [ ] CI/CD pipeline green

## 🚨 Common Issues & Solutions

### Issue: RAG retrieval too slow
**Solution**: 
- Cache embeddings
- Use smaller chunk sizes
- Reduce k from 10 to 5

### Issue: Conflict resolution deadlocks
**Solution**:
- Add timeout (default: 5 minutes)
- Escalate to human after 3 rounds

### Issue: CI pipeline failing
**Solution**:
```bash
# Run locally first
pre-commit run --all-files
pytest tests/ -v
ruff check .
mypy .
```

## 📞 Getting Help

### Ask Codex
```bash
codex exec -m "gpt-5.1-codex-max" "Help me implement <feature>"
```

### Ask Gemini
```bash
python3 gemini_session_test.py "Review my implementation of <feature>"
```

### Ask Claude (me!)
Just ping me in the conversation!

## 🎉 Sprint 1 Demo

After completing all issues, prepare demo showing:

1. **RAG in action**: Query codebase, show <200ms retrieval
2. **Conflict resolution**: Trigger disagreement, show resolution
3. **Security scanning**: Show SBOM + vulnerability report
4. **Observability**: Show metrics dashboard
5. **CI/CD**: Show green pipeline with all checks passing

---

**Ready to start?** Pick up issue #18 (RAG Architecture Spec) and let's go! 🚀
