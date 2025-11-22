"""
Knowledge Management Module

Combines semantic knowledge graphs with RAG-based code retrieval for
comprehensive knowledge representation and semantic code search.

Components:
- Knowledge Graph Manager: Semantic reasoning and ontology management
- RAG Retriever: Retrieval-Augmented Generation for code search
"""

# Knowledge Graph components
from .knowledge_graph_manager import (
    KnowledgeGraphManager,
    KnowledgeEntity,
    KnowledgeRelation,
    Ontology,
    SemanticQuery,
    InferenceRule,
    GraphAnalytics,
    EntityType,
    RelationType,
    ConfidenceLevel,
    ReasoningEngine
)

# RAG components (new - Issue #3)
try:
    from .rag_retriever import RAGRetriever, CodeChunk
    from .code_embedder import CodeEmbedder
    from .vector_store import VectorStore
    from .codebase_indexer import CodebaseIndexer
    _rag_available = True
except ImportError:
    # RAG components not yet implemented
    _rag_available = False

__all__ = [
    # Knowledge Graph
    'KnowledgeGraphManager',
    'KnowledgeEntity',
    'KnowledgeRelation',
    'Ontology',
    'SemanticQuery',
    'InferenceRule',
    'GraphAnalytics',
    'EntityType',
    'RelationType',
    'ConfidenceLevel',
    'ReasoningEngine',
]

# Add RAG components if available
if _rag_available:
    __all__.extend([
        'RAGRetriever',
        'CodeChunk',
        'CodeEmbedder',
        'VectorStore',
        'CodebaseIndexer'
    ])