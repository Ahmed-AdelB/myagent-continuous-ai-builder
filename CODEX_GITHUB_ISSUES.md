**Sprint 1 – RAG + Foundation**

- **Define RAG Architecture & Corpus Specification**
  - Description: Produce a concrete RAG spec covering corpus scope (code/docs/tests), chunking rules (AST + semantic, max token budget), embeddings model choice, vector store config, metadata schema, and retrieval parameters (k, filters). Include evaluation plan (precision@k, latency targets) and offline corpora fallback for restricted network.
  - Acceptance Criteria: Spec merged in docs with diagrams; corpus+chunking rules enumerated; embedding/vector store choices justified; evaluation metrics/SLAs defined; offline mode documented.
  - Labels: priority:critical, type:documentation, sprint:1
  - Effort: M
  - Dependencies: none

- **Implement `core/knowledge/rag_retriever.py`**
  - Description: Build RAGRetriever with AST-aware chunking, deterministic chunk IDs, embedding + Chroma (or pluggable) storage, metadata tagging (path, type, lines, hash), and query API to return ranked snippets. Include cost/latency logging hooks.
  - Acceptance Criteria: Module implemented with unit tests covering chunking, deduplication, query ranking, and failure paths; configurable k/filters; JSON logging of latency and cost per query.
  - Labels: priority:critical, type:feature, sprint:1
  - Effort: L
  - Dependencies: Define RAG Architecture & Corpus Specification

- **RAG Ingestion Pipeline (CI Job)**
  - Description: Add CLI/CI job to index the codebase nightly/on-demand using RAGRetriever. Handles incremental updates via file hashes, outputs snapshot metadata (version, timestamp, counts), and fails fast on missing embeddings. Supports offline mode using cached embeddings.
  - Acceptance Criteria: CLI command documented; CI job yaml drafted; incremental indexing demonstrated in tests; snapshot artifact generated and stored; offline flag tested.
  - Labels: priority:high, type:feature, sprint:1
  - Effort: M
  - Dependencies: Implement `core/knowledge/rag_retriever.py`

- **RAG Evaluation Suite**
  - Description: Create deterministic test corpus + benchmarks for retrieval quality and performance. Includes golden queries, expected file hits, precision@k targets, latency budget, and mutation/property-based checks for reproducibility.
  - Acceptance Criteria: Tests runnable via `pytest`; seeds fixed; thresholds enforced (precision@k, p95 latency); CI gate added; failure output is actionable.
  - Labels: priority:high, type:testing, sprint:1
  - Effort: M
  - Dependencies: Implement `core/knowledge/rag_retriever.py`, RAG Ingestion Pipeline

- **Integrate RAG into CodebaseAnalyzer**
  - Description: Wire CodebaseAnalyzer to use RAGRetriever for targeted queries (architecture, security, patterns), replacing 1M token reads. Generate RAG-backed `CODEBASE_INVENTORY.json` and Mermaid diagrams.
  - Acceptance Criteria: Analyzer uses RAG (no giant context loads); inventory JSON generated in local run; diagrams produced; smoke test passes; logs include retrieved snippets count and token usage.
  - Labels: priority:critical, type:feature, sprint:1
  - Effort: L
  - Dependencies: RAG Ingestion Pipeline, RAG Evaluation Suite

- **Conflict Resolution Protocol & Lead-Agent Tie-Breaker**
  - Description: Implement ConflictResolver with lead-agent ownership, quorum/timeout rules, human escalation path, and audit trail. Expose metrics (conflict rate, resolution time).
  - Acceptance Criteria: Code merged; unit tests for quorum/timeout/escalation; audit log persisted; metrics emitted; documented runbook for human escalation.
  - Labels: priority:high, type:feature, sprint:1
  - Effort: M
  - Dependencies: none

- **Task Ledger & Orchestration Hardening**
  - Description: Add durable task ledger with states (queued/running/retry/blocked/done), retry + jitter, timeouts, cancellation, and circuit breakers for routing. Include dependency tracking to avoid deadlocks.
  - Acceptance Criteria: Ledger data model implemented; API covers CRUD + dependency graph; tests for retries/backoff/timeout; circuit breaker toggles routing; JSON logs for state transitions.
  - Labels: priority:critical, type:feature, sprint:1
  - Effort: L
  - Dependencies: Conflict Resolution Protocol & Lead-Agent Tie-Breaker

- **GeminiSession Wrapper: Persistence & Redaction**
  - Description: Implement session wrapper with persistent history, key rotation, PII/secrets redaction, and configurable retention. Provide hooks for routing to prefer session continuity.
  - Acceptance Criteria: Wrapper in code; tests for history resume, redaction, and key rotation; config documented; logging avoids sensitive data.
  - Labels: priority:high, type:feature, sprint:1
  - Effort: M
  - Dependencies: Task Ledger & Orchestration Hardening

- **Security & Supply Chain Baseline**
  - Description: Pin dependencies, add SBOM generation, enable vulnerability + secret scanning (trivy/gitleaks), and add pre-commit hooks. Document network/offline scan fallback.
  - Acceptance Criteria: SBOM produced in CI; scans run in CI with fail criteria; dependency pins updated; offline scan instructions documented.
  - Labels: priority:high, type:chore, sprint:1
  - Effort: M
  - Dependencies: none

- **Run Phase 0 Research & Produce RESEARCH_SYNTHESIS.json**
  - Description: Execute autonomous research (with network-safe constraints), consolidate into RESEARCH_SYNTHESIS.json and individual agent reports, citing sources and tagging topic coverage.
  - Acceptance Criteria: JSON and markdown reports generated; citations included; coverage of all listed topics; stored in repo; summary note linked in README/START_HERE.
  - Labels: priority:medium, type:documentation, sprint:1
  - Effort: M
  - Dependencies: Security & Supply Chain Baseline (to ensure scan before ingest), GeminiSession Wrapper (for continuity)

- **RAG-Based Codebase Inventory Generation**
  - Description: Run RAG-backed analysis to produce `CODEBASE_INVENTORY.json` with architecture, security hotspots, and component metadata; verify outputs stored and repeatable.
  - Acceptance Criteria: Inventory JSON generated via documented command; outputs validated against sample queries; artifacts archived; links added to docs.
  - Labels: priority:critical, type:feature, sprint:1
  - Effort: M
  - Dependencies: Integrate RAG into CodebaseAnalyzer

---

**Sprint 2 – Gap Analysis, Routing, Validation**

- GapAnalyzer Implementation (consensus scoring, parallel runs) – Labels: priority:high, type:feature, sprint:2; Effort: M; Dependencies: RAG-Based Codebase Inventory Generation  
- Enhanced Router with multi-signal fitness (capability + cost + latency + load + history) – Labels: priority:critical, type:feature, sprint:2; Effort: L; Dependencies: Task Ledger & Orchestration Hardening  
- Validation Pipeline (4-layer: self-check, peer, tests, static analysis) – Labels: priority:critical, type:feature, sprint:2; Effort: L; Dependencies: Enhanced Router  
- CI Testing Gates (pre-commit → PR → nightly → chaos) – Labels: priority:high, type:testing, sprint:2; Effort: M; Dependencies: Validation Pipeline  
- Security/Supply Chain Expansion (policy-as-code, infra scans) – Labels: priority:high, type:chore, sprint:2; Effort: M; Dependencies: Security & Supply Chain Baseline  
- Observability Foundations (structured JSON logs, traces, metrics, SLOs) – Labels: priority:critical, type:feature, sprint:2; Effort: L; Dependencies: Task Ledger & Orchestration Hardening  
- Data Governance Controls (redaction, retention, dataset versioning for RAG) – Labels: priority:high, type:feature, sprint:2; Effort: M; Dependencies: RAG Ingestion Pipeline, Observability Foundations  

---

**Sprint 3 – First Improvement Cycle, Rollouts, Safety Nets**

- Execute Gap Remediation Cycle #1 (top gaps from GAPS_ANALYSIS) – Labels: priority:high, type:feature, sprint:3; Effort: L; Dependencies: GapAnalyzer Implementation  
- Feature Flags & Rollback Plan for agent-driven changes – Labels: priority:high, type:feature, sprint:3; Effort: M; Dependencies: Task Ledger & Orchestration Hardening  
- Non-Functional Test Suite (performance/load/latency budgets) – Labels: priority:medium, type:testing, sprint:3; Effort: M; Dependencies: CI Testing Gates  
- Routing Optimization via Outcomes (feedback loop to adjust weights) – Labels: priority:medium, type:enhancement, sprint:3; Effort: M; Dependencies: Enhanced Router, Observability Foundations  
- SDLC Completion Report & executive-ready metrics – Labels: priority:medium, type:documentation, sprint:3; Effort: S; Dependencies: All prior sprint deliverables