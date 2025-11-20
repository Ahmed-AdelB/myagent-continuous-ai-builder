# RAG Specification Changelog

## Version 1.1 - Tri-Agent Review Revisions (2025-11-20)

### Tri-Agent Approval
- ✅ **Codex (GPT-5.1-Codex-Max)**: Technical feasibility review - 8 recommendations
- ✅ **Gemini (2.5 Pro)**: Strategic & security review - 8 recommendations
- ✅ **Claude (Sonnet 4.5)**: Implementation updates applied

### Critical Changes (HIGH Priority)

#### 1. Chunking Strategy Revised (Codex)
**Problem**: Original spec allowed 500-1000+ token blobs that violated the stated 500-token cap.

**Changes**:
- ❌ **Removed**: Class-level chunks with "all methods" aggregation
- ✅ **Updated**: Max chunk size from 500 → **350 tokens** (hard cap: 400)
- ✅ **Updated**: Min chunk size from 20 → **80 tokens** (eliminate noise)
- ✅ **Updated**: Target chunk size from 150 → **150-220 tokens**
- ✅ **Added**: 20-40 token overlap for long functions
- ✅ **Updated**: Split classes by method, not as monolithic blocks

**New Chunking Rules**:
```
- Min: 80 tokens
- Target: 150-220 tokens
- Max: 350 tokens (hard cap: 400 in rare cases)
- Overlap: 20-40 tokens for functions >250 tokens
```

#### 2. Performance Targets Adjusted (Codex)
**Problem**: <200ms p95 end-to-end latency not feasible with OpenAI API calls.

**Changes**:
- ✅ **Updated**: End-to-end latency targets:
  - p50: <100ms → **<300ms**
  - p95: <200ms → **<500ms** (was unrealistic)
  - p99: <500ms → **<800ms**

- ✅ **Clarified**: 200ms target applies to **retrieval-only** (embedding pre-computed), not query embedding + retrieval
- ✅ **Added**: Query/result caching strategy to achieve sub-200ms for repeated queries
- ✅ **Added**: Local embedding service option for latency-sensitive deployments

**Latency Breakdown**:
```
Query Embedding:     150-300ms (OpenAI API)
Vector Search:       20-50ms (ChromaDB HNSW)
Reranking:          10-20ms (local)
Context Assembly:    10-30ms
─────────────────────────────────────────
Total (p95):        ~500ms

With cache hit:     <50ms (bypasses embedding)
With local model:   ~200-300ms (no network calls)
```

### Important Changes (MEDIUM Priority)

#### 3. Token Budget Tightened (Codex)
**Changes**:
- ✅ **Updated**: Min chunk size 20 → **80 tokens**
- ✅ **Updated**: Max chunk size 500 → **350 tokens** (hard cap 400)
- ✅ **Added**: Separate handling for docstrings/comments (no tiny shards)

#### 4. Embeddings Strategy Enhanced (Codex + Gemini)
**Changes**:
- ✅ **Added**: Evaluation plan for `text-embedding-3-large` vs `-small`
- ✅ **Added**: Local OSS model option (nomic-embed-text or e5-large)
- ✅ **Added**: Client-side L2 normalization for cosine similarity
- ✅ **Updated**: Realistic latency expectations (150-400ms per API request)

#### 5. ChromaDB Configuration Explicit (Codex)
**Changes**:
- ✅ **Added**: Explicit HNSW parameters:
  ```python
  collection = client.get_or_create_collection(
      name=collection_name,
      metadata={
          "hnsw:space": "cosine",
          "hnsw:M": 64,                  # Links per node
          "hnsw:efConstruction": 200,    # Build-time search depth
          "hnsw:efSearch": 100           # Query-time search depth
      }
  )
  ```
- ✅ **Added**: Embedding normalization requirement before insert
- ✅ **Added**: Verification that `collection.query` stores embeddings

#### 6. Retrieval Parameters Relaxed (Codex)
**Changes**:
- ✅ **Updated**: Retrieval strategy:
  - ❌ **Old**: Retrieve k=5 directly with min_score=0.7
  - ✅ **New**: Retrieve n_results=20-30, rerank to top-k=5-8, then apply min_score

- ✅ **Added**: MMR (Maximal Marginal Relevance) for diversity
- ✅ **Added**: Rerank signals:
  ```python
  scores = [
      0.5 * cosine_similarity,
      0.2 * recency_score,
      0.1 * file_importance,
      0.1 * chunk_completeness,
      0.1 * code_graph_proximity  # NEW from Gemini
  ]
  ```

#### 7. Offline Mode Hardened (Codex + Gemini)
**Changes**:
- ❌ **Removed**: Simple BM25 fallback as primary offline strategy
- ✅ **Added**: Bundled quantized `all-MiniLM-L6-v2` model (80MB)
- ✅ **Added**: Local HNSW index build path for air-gapped updates
- ✅ **Added**: Semantic search on queries even without internet
- ✅ **Kept**: BM25 as last-resort fallback only

### Strategic Enhancements (Gemini)

#### 8. Security Architecture (Gemini)
**Problem**: Storing raw code in vector DB increases PII leakage risk.

**Changes**:
- ✅ **Added**: Decoupled storage architecture:
  ```
  Vector DB (ChromaDB):
    - Stores: embeddings + metadata + chunk_id
    - Does NOT store: raw code text

  Code Store (encrypted):
    - HashiCorp Vault or encrypted S3
    - Stores: raw code keyed by chunk_id
    - Retrieved only at context assembly step
  ```

- ✅ **Added**: Multi-layer PII redaction:
  - Regex patterns (emails, keys, SSN)
  - Entropy analysis for high-entropy strings
  - Optional: Fine-tuned LLM for contextual secrets

#### 9. Compliance Hardening (Gemini)
**Changes**:
- ✅ **Added**: GDPR deletion workflow:
  ```
  1. git blame → identify user's chunks
  2. Delete from vector DB (by chunk_id)
  3. Delete from code store
  4. No full re-index required
  ```

- ✅ **Added**: Tamper-evident audit logs:
  - Write-only log stream (AWS CloudWatch or dedicated service)
  - Cannot be altered by application
  - Logs: query, timestamp, agent_id, chunks_retrieved

#### 10. Cost Model Corrected (Gemini)
**Problem**: Baseline calculation used 1M tokens, but codebase is only 200k.

**Changes**:
- ✅ **Fixed**: Baseline cost calculation:
  - ❌ **Old**: 1M tokens × $0.003/1K = $3 per query
  - ✅ **New**: 200K tokens × $0.003/1K = **$0.60 per query**

- ✅ **Updated**: Target cost:
  - <60K tokens × $0.003/1K = **<$0.18 per query**
  - Savings: 70% (not 60% as originally claimed)

#### 11. Risk Management Section Added (Gemini)
**New Section**:
- ✅ **Added**: Formal risk assessment with mitigations:
  1. **Stale Index Risk**: Nightly full re-index mandatory
  2. **Semantic Drift**: Version embeddings with code_version metadata
  3. **Catastrophic Forgetting**: Never fine-tune production model
  4. **PII Leakage**: Multi-layer redaction + encrypted storage

#### 12. Scalability Plan Added (Gemini)
**Changes for 10x Growth** (2M tokens, 20k chunks):
- ✅ **Added**: ChromaDB as standalone service (not embedded)
- ✅ **Added**: Async indexing via Celery + Redis Queue
- ✅ **Added**: Migration path to Weaviate/Pinecone at scale
- ✅ **Updated**: Indexing from synchronous (3 min) → async (30 min acceptable)

### Minor Changes (LOW Priority)

#### 13. Implementation Timeline (Codex)
**Changes**:
- ✅ **Updated**: Week 1-2 tasks:
  - Added: Batching logic
  - Added: Embedding normalization
  - Added: Index tuning tasks
  - Added: Latency evaluation harness

- ✅ **Updated**: Total timeline:
  - ❌ **Old**: 3 weeks (optimistic)
  - ✅ **New**: **4-5 weeks** (includes eval + tuning + offline integration)

#### 14. Reranking Enhanced (Gemini)
**Changes**:
- ✅ **Added**: Code Graph Proximity score:
  ```python
  def calculate_code_graph_proximity(chunk, retrieved_chunks):
      """
      Boost score if chunk:
      - Calls functions in retrieved_chunks
      - Is called by functions in retrieved_chunks
      - Shares imports with retrieved_chunks
      """
      proximity_score = 0.0
      for other in retrieved_chunks:
          if chunk.calls(other) or other.calls(chunk):
              proximity_score += 0.3
          if chunk.shares_imports(other):
              proximity_score += 0.1
      return min(proximity_score, 1.0)
  ```

- ✅ **Added**: Pre-calculation during indexing:
  - AST analysis to extract call graph
  - Store in chunk metadata

#### 15. Evaluation Dashboard (Gemini)
**Changes**:
- ✅ **Added**: Human-in-the-loop (HITL) evaluation:
  - Sample 50-100 queries per week
  - Human experts mark chunk relevance
  - Track precision/recall drift over time
  - Use feedback to tune reranking weights

## Summary of Changes

**Total Recommendations**: 16
- **Critical (HIGH)**: 2 - Chunking, Performance
- **Important (MEDIUM)**: 5 - Token budget, Embeddings, ChromaDB, Retrieval, Offline
- **Strategic (Gemini)**: 8 - Security, Compliance, Cost, Risk, Scalability, Reranking, Evaluation
- **Minor (LOW)**: 1 - Implementation timeline

**Status**: Ready for implementation after v1.1 updates applied

## Action Items for Spec Update

1. [ ] Update Problem Statement (fix 1M → 200k tokens)
2. [ ] Update Chunking Strategy section
3. [ ] Update Performance Metrics section
4. [ ] Update Token Budget section
5. [ ] Update ChromaDB configuration code
6. [ ] Update Retrieval Parameters
7. [ ] Update Offline Mode section
8. [ ] Add Security Architecture section
9. [ ] Add Risk Management section
10. [ ] Add Scalability Plan section
11. [ ] Update Implementation Timeline
12. [ ] Add Code Graph Proximity to reranking
13. [ ] Update Cost Model calculations

## Next Steps

1. Apply all changes to `rag_specification.md` (create v1.1)
2. Get final tri-agent approval on updated spec
3. Proceed to Issue #3 (Implement RAG Retriever)
