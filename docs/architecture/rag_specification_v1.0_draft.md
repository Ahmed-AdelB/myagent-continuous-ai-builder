# RAG (Retrieval-Augmented Generation) Architecture Specification

**Version**: 1.0
**Status**: Draft
**Last Updated**: 2025-11-20
**Owner**: Tri-Agent SDLC Team

## 📋 Executive Summary

This specification defines the Retrieval-Augmented Generation (RAG) system for the Continuous AI App Builder. RAG solves the "lost in the middle" problem when using large language models with 1M+ token contexts by retrieving only the most relevant code chunks instead of dumping entire codebases.

**Key Objectives**:
- Replace expensive 1M token dumps with targeted retrieval
- Achieve <200ms retrieval latency (p95)
- Reduce token usage by >70%
- Maintain or improve code comprehension accuracy
- Support offline operation for restricted environments

## 🎯 Problem Statement

### Current Issues
1. **Lost in the Middle**: LLMs struggle to find relevant information in large context windows (1M tokens)
2. **Cost**: Full codebase dumps are expensive (1M tokens × $0.003/1K = $3 per query)
3. **Latency**: Processing 1M tokens takes 15-30 seconds
4. **Accuracy**: Retrieval precision decreases with context size

### Success Metrics
- **Retrieval Latency**: <200ms at p95
- **Token Reduction**: >70% compared to full dumps
- **Precision@5**: >0.85 (5 out of 5 chunks relevant)
- **Recall@10**: >0.90 (captures 90% of relevant code)
- **Cost Savings**: >60% reduction in API costs

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐│
│  │ Indexing │───▶│ Embedding│───▶│  Vector  │───▶│Retrieval││
│  │ Pipeline │    │ Generation│    │  Store   │    │  API   ││
│  └──────────┘    └──────────┘    └──────────┘    └────────┘│
│       │               │                │               │     │
│       ▼               ▼                ▼               ▼     │
│  Parse Code      OpenAI API       ChromaDB        Semantic  │
│  with tree-      text-embedding   Persistent      Search    │
│  sitter          -3-small          Storage        k=5-10    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Indexing Pipeline
**Purpose**: Parse codebase and extract meaningful chunks

**Process**:
```
Codebase Files → tree-sitter Parser → Code Chunks → Metadata Extraction → Index
```

**Key Features**:
- AST-aware parsing (respects function boundaries)
- Incremental indexing (only re-index changed files)
- Multi-language support (Python, JavaScript, TypeScript)
- Git-aware (tracks file history for context)

#### 2. Embedding Generation
**Purpose**: Convert code chunks to semantic vectors

**Model Selection**: `text-embedding-3-small`
- **Dimensions**: 1536
- **Max tokens**: 8191
- **Cost**: $0.00002 per 1K tokens
- **Latency**: ~50ms per batch of 100 chunks

**Why this model**:
- Best cost/performance ratio
- Optimized for code understanding
- Fast inference (<50ms)
- Small storage footprint

#### 3. Vector Store
**Purpose**: Efficient similarity search over embeddings

**Database**: ChromaDB
- **Why ChromaDB**: Embedded, persistent, Python-native
- **Storage**: Local disk (persistence/storage/vector_db/)
- **Index**: HNSW (Hierarchical Navigable Small World)
- **Distance metric**: Cosine similarity

**Schema**:
```python
{
    "id": "file_path:chunk_id",
    "embedding": [1536 float values],
    "metadata": {
        "file_path": str,
        "language": str,
        "chunk_type": str,  # function, class, module, comment
        "start_line": int,
        "end_line": int,
        "function_name": str,
        "class_name": str,
        "imports": List[str],
        "git_hash": str,
        "last_modified": timestamp
    },
    "document": str  # actual code text
}
```

#### 4. Retrieval API
**Purpose**: Query interface for agents

**Methods**:
```python
async def retrieve(
    query: str,
    k: int = 5,
    filters: Dict[str, Any] = None,
    min_score: float = 0.7
) -> List[CodeChunk]
```

## 📦 Corpus Definition

### What to Index

**Primary Corpus** (always indexed):
1. **Production Code**: `core/`, `api/`, `frontend/src/`
2. **Tests**: `tests/` (for understanding behavior)
3. **Documentation**: `docs/`, `README.md`, `CLAUDE.md`
4. **Configuration**: `pyproject.toml`, `package.json`, `.github/workflows/`

**Secondary Corpus** (optional, based on storage):
1. **Dependencies**: `venv/lib/python3.*/site-packages/` (select packages only)
2. **Git History**: Commit messages and diffs (last 100 commits)
3. **Issues/PRs**: GitHub issues and PR descriptions (via API)

**Excluded**:
- Binary files (images, PDFs, etc.)
- Generated code (`__pycache__/`, `node_modules/`)
- Large data files (>10MB)
- Secrets/credentials

### Corpus Statistics (Estimated)

```
MyAgent Codebase:
├── Python files: ~150 files, ~25,000 LOC
├── JavaScript/React: ~50 files, ~8,000 LOC
├── Documentation: ~20 files, ~5,000 words
├── Tests: ~100 files, ~15,000 LOC
└── Total tokens: ~200,000 tokens

Chunked corpus:
├── Total chunks: ~1,500-2,000
├── Average chunk size: 100-150 tokens
├── Embeddings storage: ~12 MB (1500 × 1536 × 4 bytes)
└── Indexing time: ~2-3 minutes (full), ~5-10s (incremental)
```

## ✂️ Chunking Strategy

### Hierarchical Chunking

**Level 1: Module-Level Chunks**
- Entire file if <500 tokens
- Module docstring + imports
- Used for high-level understanding

**Level 2: Class-Level Chunks**
- Class definition + docstring
- All methods included
- Used for API understanding

**Level 3: Function-Level Chunks** (Primary)
- Individual functions with docstrings
- Context: imports, class name (if method)
- Size: 50-300 tokens per chunk

**Level 4: Comment Chunks**
- Standalone comment blocks (>3 lines)
- TODO/FIXME/NOTE annotations
- Architecture explanations

### Chunking Rules

```python
def chunk_code(file_path: str, language: str) -> List[CodeChunk]:
    """
    Chunking algorithm:

    1. Parse file with tree-sitter to get AST
    2. Extract top-level constructs (functions, classes)
    3. For each construct:
       a. If <500 tokens: keep as single chunk
       b. If 500-1000 tokens: split at method boundaries
       c. If >1000 tokens: split into logical sub-chunks
    4. Add context metadata (imports, parent class, etc.)
    5. Generate chunk ID: {file_path}:{start_line}-{end_line}
    """
```

**Overlap Strategy**:
- No overlap for function chunks (clean boundaries)
- 20-token overlap for split comments/docstrings
- Include parent context (class name) in metadata

**Token Budget**:
- Min chunk size: 20 tokens (avoid noise)
- Max chunk size: 500 tokens (fit in context)
- Target chunk size: 150 tokens (optimal for retrieval)

### Example Chunking

**Input Code**:
```python
# core/agents/coder_agent.py

class CoderAgent(PersistentAgent):
    """Implements features based on specifications."""

    async def initialize(self):
        """Initialize coder agent."""
        self.filesystem = FileSystemUtils()

    async def process_task(self, task: Task) -> TaskResult:
        """Process coding task."""
        # Implementation here
        pass
```

**Output Chunks**:
```json
[
    {
        "id": "core/agents/coder_agent.py:1-20",
        "chunk_type": "class",
        "document": "class CoderAgent(PersistentAgent):\n    \"\"\"Implements features...\"\"\"\n    async def initialize...",
        "metadata": {
            "class_name": "CoderAgent",
            "methods": ["initialize", "process_task"],
            "imports": ["PersistentAgent", "Task", "TaskResult"]
        }
    },
    {
        "id": "core/agents/coder_agent.py:5-8",
        "chunk_type": "function",
        "document": "async def initialize(self):\n    \"\"\"Initialize coder agent.\"\"\"\n    self.filesystem = FileSystemUtils()",
        "metadata": {
            "function_name": "initialize",
            "class_name": "CoderAgent",
            "is_async": true
        }
    },
    {
        "id": "core/agents/coder_agent.py:10-13",
        "chunk_type": "function",
        "document": "async def process_task(self, task: Task) -> TaskResult:\n    \"\"\"Process coding task.\"\"\"",
        "metadata": {
            "function_name": "process_task",
            "class_name": "CoderAgent",
            "parameters": ["task: Task"],
            "return_type": "TaskResult"
        }
    }
]
```

## 🔍 Retrieval Strategy

### Query Processing

**Step 1: Query Expansion**
```python
def expand_query(query: str) -> List[str]:
    """
    Generate multiple search queries from single input.

    Example:
    Input: "How does error handling work?"
    Output: [
        "How does error handling work?",
        "error handling implementation",
        "try except catch errors",
        "exception handling patterns"
    ]
    """
```

**Step 2: Semantic Search**
```python
# Generate embedding for query
query_embedding = await embedder.embed(query)

# Search vector DB
results = vector_store.query(
    query_embedding=query_embedding,
    n_results=k * 2,  # Retrieve 2x, then rerank
    where=filters  # Optional metadata filters
)
```

**Step 3: Reranking**
```python
# Rerank by multiple signals:
scores = [
    0.6 * cosine_similarity,     # Semantic match
    0.2 * recency_score,         # Prefer recent code
    0.1 * file_importance,       # Core vs. peripheral files
    0.1 * chunk_completeness     # Complete functions > fragments
]

top_k = rerank(results, scores, k=k)
```

**Step 4: Context Assembly**
```python
def assemble_context(chunks: List[CodeChunk]) -> str:
    """
    Format chunks for LLM consumption.

    Output format:
    ```
    # File: core/agents/coder_agent.py (lines 5-8)
    async def initialize(self):
        \"\"\"Initialize coder agent.\"\"\"
        self.filesystem = FileSystemUtils()

    # File: core/agents/coder_agent.py (lines 10-13)
    async def process_task(self, task: Task) -> TaskResult:
        \"\"\"Process coding task.\"\"\"
        ...
    ```
    """
```

### Retrieval Parameters

**Default Configuration**:
```python
RETRIEVAL_CONFIG = {
    "k": 5,                    # Top-k chunks
    "min_score": 0.70,         # Minimum cosine similarity
    "max_tokens": 2000,        # Max total context tokens
    "diversity_penalty": 0.2,  # Penalize duplicate information
    "recency_boost": 0.1,      # Boost recently modified files
    "filters": {
        "language": None,      # Filter by programming language
        "chunk_type": None,    # Filter by chunk type
        "file_pattern": None   # Filter by file path pattern
    }
}
```

**Adaptive k Selection**:
```python
# Adjust k based on query complexity
if "how does X work" in query.lower():
    k = 10  # Broader exploration
elif "implement X" in query.lower():
    k = 5   # Focused retrieval
elif "debug X" in query.lower():
    k = 8   # Include related code
```

## 📊 Evaluation Metrics

### Retrieval Quality

**Precision@k**: Percentage of retrieved chunks that are relevant
```
Precision@5 = relevant_chunks_in_top_5 / 5
Target: >0.85
```

**Recall@k**: Percentage of all relevant chunks retrieved
```
Recall@10 = relevant_chunks_retrieved / total_relevant_chunks
Target: >0.90
```

**Mean Reciprocal Rank (MRR)**: How early the first relevant chunk appears
```
MRR = 1 / rank_of_first_relevant_chunk
Target: >0.80
```

**NDCG@k** (Normalized Discounted Cumulative Gain): Relevance-weighted metric
```
NDCG@10 = DCG@10 / IDCG@10
Target: >0.85
```

### Performance Metrics

**Latency**:
- p50: <100ms
- p95: <200ms
- p99: <500ms

**Token Efficiency**:
- Baseline: 200,000 tokens (full codebase)
- Target: <60,000 tokens (70% reduction)
- Actual: Measure per query

**Cost**:
- Embedding cost: $0.00002/1K tokens × 200K = $0.004 per full index
- Storage: ~12 MB for full corpus
- Retrieval: Free (ChromaDB is embedded)

### Accuracy Metrics

**Agent Task Success Rate**:
- Baseline: 91.7% (with full context)
- Target: >91.0% (maintain quality)

**False Negative Rate**: Relevant code missed
```
FNR = missed_relevant_chunks / total_relevant_chunks
Target: <0.10
```

**False Positive Rate**: Irrelevant code retrieved
```
FPR = irrelevant_chunks_retrieved / k
Target: <0.15
```

## 🔧 Implementation Plan

### Phase 1: Core RAG Infrastructure (Week 1-2)

**Task 1.1: Indexing Pipeline** (3 days)
```python
# core/knowledge/indexer.py

class CodebaseIndexer:
    """Indexes codebase using tree-sitter."""

    async def index_directory(self, root_path: Path) -> IndexResult:
        """Index entire directory."""
        pass

    async def index_file(self, file_path: Path) -> List[CodeChunk]:
        """Index single file."""
        pass

    async def incremental_index(self, changed_files: List[Path]) -> IndexResult:
        """Re-index only changed files."""
        pass
```

**Task 1.2: Embedding Generation** (2 days)
```python
# core/knowledge/embedder.py

class CodeEmbedder:
    """Generates embeddings using OpenAI API."""

    def __init__(self):
        self.client = OpenAI()
        self.model = "text-embedding-3-small"
        self.cache = {}  # Local cache

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts."""
        pass
```

**Task 1.3: Vector Store Setup** (2 days)
```python
# core/knowledge/vector_store.py

class VectorStore:
    """ChromaDB wrapper for code embeddings."""

    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(
            path="persistence/storage/vector_db"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Add code chunks to vector store."""
        pass

    async def query(self, query_embedding: np.ndarray, k: int) -> List[CodeChunk]:
        """Semantic search for similar chunks."""
        pass
```

**Task 1.4: RAG Retriever** (3 days)
```python
# core/knowledge/rag_retriever.py

class RAGRetriever:
    """Main RAG interface for agents."""

    def __init__(self, project_name: str):
        self.indexer = CodebaseIndexer()
        self.embedder = CodeEmbedder()
        self.vector_store = VectorStore(f"{project_name}_code")

    async def index_codebase(self, root_path: Path) -> None:
        """Index entire codebase."""
        chunks = await self.indexer.index_directory(root_path)
        embeddings = await self.embedder.embed_batch([c.text for c in chunks])
        await self.vector_store.add_chunks(chunks, embeddings)

    async def retrieve(self, query: str, k: int = 5) -> List[CodeChunk]:
        """Retrieve relevant code chunks."""
        query_embedding = await self.embedder.embed(query)
        return await self.vector_store.query(query_embedding, k)
```

### Phase 2: Integration (Week 2)

**Task 2.1: TriAgentSDLC Integration** (2 days)
```python
# core/orchestrator/tri_agent_sdlc.py

class TriAgentSDLC:
    def __init__(self, project_name: str):
        self.rag_retriever = RAGRetriever(project_name)

    async def _get_codebase_context(self, query: str) -> str:
        """Get relevant codebase context via RAG."""
        try:
            chunks = await self.rag_retriever.retrieve(query, k=10)
            return self._format_chunks(chunks)
        except Exception as e:
            logger.warning(f"RAG failed: {e}. Falling back to full dump.")
            return await self._get_full_codebase()  # Fallback
```

**Task 2.2: CI/CD Integration** (1 day)
```yaml
# .github/workflows/rag_index.yml

name: RAG Indexing
on:
  push:
    branches: [main, master]

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Index codebase
        run: |
          python -m core.knowledge.rag_retriever index .
      - name: Upload embeddings
        uses: actions/upload-artifact@v3
        with:
          name: vector-db
          path: persistence/storage/vector_db/
```

### Phase 3: Evaluation (Week 3)

**Task 3.1: Evaluation Dataset** (2 days)
- Collect 50 representative queries
- Manual annotation of relevant chunks
- Baseline metrics with full context

**Task 3.2: Metrics Collection** (2 days)
```python
# tests/test_rag_retriever.py

async def test_retrieval_quality():
    """Test retrieval metrics."""
    retriever = RAGRetriever("test_project")

    for query, ground_truth in eval_dataset:
        results = await retriever.retrieve(query, k=10)

        precision = calculate_precision(results, ground_truth)
        recall = calculate_recall(results, ground_truth)

        assert precision > 0.85
        assert recall > 0.90
```

**Task 3.3: Performance Benchmarks** (1 day)
```python
async def benchmark_latency():
    """Benchmark retrieval latency."""
    retriever = RAGRetriever("myagent")

    latencies = []
    for _ in range(100):
        start = time.time()
        await retriever.retrieve("How does error handling work?")
        latencies.append(time.time() - start)

    assert np.percentile(latencies, 95) < 0.200  # <200ms p95
```

## 🌐 Offline Mode

### Requirements
- Support air-gapped environments
- Pre-computed embeddings bundled with release
- No internet access required after initial setup

### Implementation

**Step 1: Pre-compute Embeddings**
```bash
# During CI/CD build
python -m core.knowledge.rag_retriever index . --offline-bundle

# Generates:
# - persistence/storage/vector_db/ (ChromaDB)
# - offline_embeddings.tar.gz (compressed)
```

**Step 2: Offline Retrieval**
```python
class RAGRetriever:
    def __init__(self, project_name: str, offline_mode: bool = False):
        if offline_mode:
            # Use bundled embeddings
            self.vector_store = VectorStore.from_bundle(
                "offline_embeddings.tar.gz"
            )
            self.embedder = None  # No OpenAI calls
        else:
            # Online mode
            self.embedder = CodeEmbedder()
```

**Step 3: Query Processing (Offline)**
```python
async def retrieve_offline(self, query: str, k: int = 5) -> List[CodeChunk]:
    """
    Offline retrieval using pre-computed embeddings.

    Falls back to keyword search if embeddings not available.
    """
    if self.embedder is None:
        # Use BM25 keyword search as fallback
        return await self._keyword_search(query, k)
    else:
        # Normal RAG retrieval
        return await self.retrieve(query, k)
```

## 🔒 Security & Privacy

### Data Protection
- **PII Redaction**: Remove emails, API keys, credentials before indexing
- **Access Control**: Vector DB access restricted to authenticated agents
- **Audit Logging**: All queries logged with timestamp + agent ID
- **Retention**: Embeddings retained for 1 year, then deleted

### Compliance
- **GDPR**: User can request deletion of their code embeddings
- **SOC 2**: Audit trail for all access + encryption at rest
- **ISO 27001**: Regular security scans of vector DB

## 📈 Monitoring & Observability

### Metrics to Track

**Retrieval Metrics**:
- Queries per minute
- Latency (p50, p95, p99)
- Cache hit rate
- Error rate

**Quality Metrics**:
- Precision@5, Recall@10 (sampled)
- Agent task success rate
- False positive rate

**Cost Metrics**:
- Embedding API calls
- Storage usage (MB)
- Compute time for indexing

### Dashboards

```python
# Prometheus metrics
retrieval_latency = Histogram("rag_retrieval_latency_seconds")
retrieval_precision = Gauge("rag_precision_at_5")
retrieval_recall = Gauge("rag_recall_at_10")
embedding_cost = Counter("rag_embedding_cost_usd")
```

**Grafana Dashboard**:
- Latency over time (line chart)
- Precision/Recall over time (line chart)
- Cost per day (bar chart)
- Cache hit rate (gauge)

## 🚀 Rollout Plan

### Week 1: Development
- Implement core RAG components
- Unit tests for each module
- Integration tests with small corpus

### Week 2: Integration
- Integrate with TriAgentSDLC
- Add fallback to full dump
- CI/CD indexing pipeline

### Week 3: Evaluation
- Run evaluation dataset
- Benchmark performance
- Tune hyperparameters (k, min_score, etc.)

### Week 4: Rollout
- Deploy to staging environment
- A/B test vs. full context
- Monitor metrics for 1 week

### Week 5: Production
- Deploy to production
- Monitor closely for regressions
- Iterate based on feedback

## 📚 References

- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

## ✅ Acceptance Criteria

This specification is complete when:

- [x] Corpus scope defined (code + docs + tests)
- [x] Chunking strategy documented (hierarchical, token budget)
- [x] Embeddings model justified (text-embedding-3-small)
- [x] Vector store configured (ChromaDB, HNSW, cosine)
- [x] Retrieval parameters specified (k=5, min_score=0.7)
- [x] Evaluation metrics defined (Precision@5, Recall@10, latency)
- [x] Offline mode documented (pre-computed embeddings)
- [x] Architecture diagrams included
- [x] Implementation plan with timeline

## 🤝 Approval

**Reviewed by**:
- [ ] Claude (Sonnet 4.5) - Architecture review
- [ ] Codex (GPT-5.1) - Implementation feasibility
- [ ] Gemini (2.5 Pro) - Security & compliance

**Status**: Ready for implementation (Issue #3)

---

**Next Steps**: Proceed to issue #3 (Implement RAG Retriever) using this specification as the blueprint.
