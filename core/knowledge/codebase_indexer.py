"""
CodebaseIndexer - Parses and indexes Python code using tree-sitter AST parsing.

Implements tri-agent validated chunking strategy:
- Token budget: 80-350 tokens (target: 150-220)
- 20-40 token overlap for long functions
- Metadata extraction: function_name, class_name, imports, git info

Security (Gemini recommendations):
- TODO: Add PII redaction before indexing
- TODO: Audit logging for all indexed chunks
- TODO: Agent ID validation

Based on: docs/architecture/rag_specification.md (v1.1)
Issue: #3
Implemented by: Codex (GPT-5.1-Codex-Max)
Security review: Gemini (2.5 Pro)
Integration: Claude (Sonnet 4.5)
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# tree-sitter imports with graceful fallback
try:
    from tree_sitter import Parser
    from tree_sitter_languages import get_language, get_parser
except Exception:
    Parser = None
    get_language = None
    get_parser = None
    logger.warning("tree-sitter not available; will use Python ast fallback")


@dataclass
class CodeChunk:
    """Represents a parsed code chunk with metadata."""
    id: str
    file_path: str
    chunk_type: str  # function | method | class_fragment | module
    start_line: int
    end_line: int
    text: str
    metadata: Dict[str, Any]
    score: float = 0.0


@dataclass
class IndexResult:
    """Aggregated result of an indexing run."""
    chunks: List[CodeChunk] = field(default_factory=list)
    files_indexed: int = 0
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class CodebaseIndexer:
    """
    Indexes codebase using tree-sitter AST parsing for Python files.

    Features:
    - Tree-sitter AST parsing with Python ast fallback
    - Token budget: 80-350 tokens (v1.1 spec)
    - Function/class metadata extraction
    - Smart directory filtering (.git, __pycache__, venv)
    - 20-40 token overlap for long functions

    Security (TODO - Gemini recommendations):
    - PII redaction before indexing
    - Audit logging
    - Agent ID validation
    """

    def __init__(
        self,
        min_tokens: int = 80,
        target_min_tokens: int = 150,
        target_max_tokens: int = 220,
        max_tokens: int = 350,
        hard_max_tokens: int = 400,
        overlap_tokens: int = 30,
    ) -> None:
        """
        Initialize indexer with token budget (v1.1 spec).

        Args:
            min_tokens: Minimum chunk size (avoid noise)
            target_min_tokens: Target minimum for quality chunks
            target_max_tokens: Target maximum for quality chunks
            max_tokens: Maximum chunk size
            hard_max_tokens: Hard cap (rarely reached)
            overlap_tokens: Overlap for long functions
        """
        self.min_tokens = min_tokens
        self.target_min_tokens = target_min_tokens
        self.target_max_tokens = target_max_tokens
        self.max_tokens = max_tokens
        self.hard_max_tokens = hard_max_tokens
        self.overlap_tokens = overlap_tokens
        self.parser = self._init_parser()
        self.language = getattr(self.parser, "language", None)

    def _init_parser(self) -> Optional[Parser]:
        """Initialize tree-sitter parser for Python with fallback."""
        if Parser is None:
            logger.warning("tree-sitter not available; falling back to ast parsing")
            return None

        # Try prebuilt language from tree_sitter_languages
        if get_parser:
            try:
                parser = get_parser("python")
                logger.info("Initialized tree-sitter parser via tree_sitter_languages")
                return parser
            except Exception as exc:
                logger.warning(f"tree_sitter_languages parser unavailable: {exc}")

        # Fallback: manually set language
        if get_language:
            try:
                lang = get_language("python")
                parser = Parser()
                parser.set_language(lang)
                logger.info("Initialized tree-sitter parser via get_language")
                return parser
            except Exception as exc:
                logger.warning(f"Failed to bind tree-sitter language: {exc}")

        logger.warning("tree-sitter initialization failed; using ast fallback")
        return None

    async def index_directory(self, root_path: Path) -> IndexResult:
        """
        Index all Python files under root_path.

        Args:
            root_path: Root directory to index

        Returns:
            IndexResult with all chunks, stats, and errors
        """
        start = time.monotonic()
        result = IndexResult()

        files = list(self._iter_python_files(root_path))
        tasks = [self.index_file(path) for path in files]

        for coro in asyncio.as_completed(tasks):
            try:
                chunks = await coro
                result.chunks.extend(chunks)
                result.files_indexed += 1
            except Exception as exc:
                msg = f"Failed to index file: {exc}"
                logger.exception(msg)
                result.errors.append(msg)

        elapsed = time.monotonic() - start
        result.stats = {
            "elapsed_seconds": round(elapsed, 3),
            "total_chunks": len(result.chunks),
            "files_indexed": result.files_indexed,
            "errors": len(result.errors),
        }

        logger.info(
            f"Indexed {result.files_indexed} files, "
            f"{len(result.chunks)} chunks in {elapsed:.2f}s"
        )

        return result

    async def index_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Index a single Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of CodeChunk objects
        """
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning(f"Failed to read {file_path}: {exc}")
            return []

        # Extract imports for metadata
        imports = self._extract_imports(source)

        # Use tree-sitter if available, otherwise fallback to ast
        if self.parser:
            chunks = await self._index_with_tree_sitter(file_path, source, imports)
        else:
            chunks = await self._index_with_ast(file_path, source, imports)

        # TODO (Gemini): Add PII redaction here before returning chunks
        # TODO (Gemini): Add audit logging for indexed chunks

        return chunks

    async def _index_with_tree_sitter(
        self, file_path: Path, source: str, imports: List[str]
    ) -> List[CodeChunk]:
        """Index using tree-sitter AST parsing."""
        tree = self.parser.parse(source.encode("utf-8"))
        root = tree.root_node

        # Find all function definitions with their enclosing class
        defs = self._iter_defs_with_class(root)

        chunks = []
        for node, class_name in defs:
            func_name = self._node_name(node, source.encode("utf-8"))
            start_byte = node.start_byte
            end_byte = node.end_byte
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            text = source[start_byte:end_byte]

            metadata = {
                "function_name": func_name,
                "class_name": class_name,
                "imports": imports,
                "language": "python",
                "parser": "tree-sitter",
            }

            # Split if too large
            if self._count_tokens(text) > self.max_tokens:
                chunks.extend(
                    self._split_large_chunk(
                        file_path, text, start_line, "function", metadata
                    )
                )
            else:
                chunk = self._build_chunk(
                    file_path, "function", start_line, end_line, text, metadata
                )
                chunks.append(chunk)

        return chunks

    async def _index_with_ast(
        self, file_path: Path, source: str, imports: List[str]
    ) -> List[CodeChunk]:
        """Fallback indexing using Python's built-in ast module."""
        import ast

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.warning(f"Syntax error in {file_path}: {exc}")
            return []

        chunks = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            func_name = node.name
            class_name = self._enclosing_class_name(node)
            start_line = node.lineno
            end_line = node.end_lineno or start_line

            lines = source.splitlines()
            text = "\n".join(lines[start_line - 1 : end_line])

            metadata = {
                "function_name": func_name,
                "class_name": class_name,
                "imports": imports,
                "language": "python",
                "parser": "ast",
            }

            if self._count_tokens(text) > self.max_tokens:
                chunks.extend(
                    self._split_large_chunk(
                        file_path, text, start_line, "function", metadata
                    )
                )
            else:
                chunk = self._build_chunk(
                    file_path, "function", start_line, end_line, text, metadata
                )
                chunks.append(chunk)

        return chunks

    def _split_large_chunk(
        self,
        file_path: Path,
        text: str,
        start_line: int,
        chunk_type: str,
        base_meta: Dict[str, Any],
    ) -> List[CodeChunk]:
        """Split large chunks to fit token budget."""
        lines = text.splitlines(keepends=True)
        tokens_per_line = [self._count_tokens(line) for line in lines]

        chunks = []
        idx = 0

        while idx < len(lines):
            # Accumulate lines up to target_max_tokens
            end = idx
            token_total = 0

            while end < len(lines) and token_total < self.target_max_tokens:
                token_total += tokens_per_line[end]
                end += 1

            # Trim back if over max_tokens
            while end > idx and token_total > self.max_tokens:
                end -= 1
                token_total -= tokens_per_line[end]

            if end == idx:  # Safety: ensure progress
                end = min(idx + 1, len(lines))
                token_total = tokens_per_line[idx]

            # Extend if below min_tokens
            while end < len(lines) and token_total < self.min_tokens:
                token_total += tokens_per_line[end]
                end += 1

            chunk_lines = lines[idx:end]
            chunk_start_line = start_line + idx
            chunk_end_line = start_line + end - 1

            chunk_meta = dict(base_meta)
            chunk_meta.update(
                {"start_line": chunk_start_line, "end_line": chunk_end_line}
            )

            chunk = CodeChunk(
                id=f"{file_path.as_posix()}:{chunk_start_line}-{chunk_end_line}",
                file_path=file_path.as_posix(),
                chunk_type=chunk_type,
                start_line=chunk_start_line,
                end_line=chunk_end_line,
                text="".join(chunk_lines),
                metadata=chunk_meta,
            )
            chunks.append(chunk)

            # Overlap only for longer functions
            if token_total >= 250 and self.overlap_tokens > 0:
                overlap_lines = self._lines_for_overlap(chunk_lines, self.overlap_tokens)
                idx = max(end - overlap_lines, idx + 1)
            else:
                idx = end

        return chunks

    def _build_chunk(
        self,
        file_path: Path,
        chunk_type: str,
        start_line: int,
        end_line: int,
        text: str,
        metadata: Dict[str, Any],
    ) -> CodeChunk:
        """Build a CodeChunk from components."""
        chunk_id = f"{file_path.as_posix()}:{start_line}-{end_line}"
        return CodeChunk(
            id=chunk_id,
            file_path=file_path.as_posix(),
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
            text=text,
            metadata=metadata,
        )

    def _iter_python_files(self, root_path: Path):
        """Yield Python files while skipping common generated/large directories."""
        skip_dirs = {".git", "__pycache__", "node_modules", "venv", ".pytest_cache"}
        for path in root_path.rglob("*.py"):
            if any(part in skip_dirs for part in path.parts):
                continue
            yield path

    def _extract_imports(self, source: str) -> List[str]:
        """Lightweight import extraction via regex."""
        import_pattern = re.compile(
            r"^(from\s+[^\s]+\s+import\s+[^\n]+|import\s+[^\n]+)", re.MULTILINE
        )
        return [line.strip() for line in import_pattern.findall(source)]

    def _iter_defs_with_class(self, root) -> List[Tuple[Any, Optional[str]]]:
        """DFS to collect function_definition nodes and their enclosing class name."""
        stack: List[Tuple[Any, Optional[str]]] = [(root, None)]
        defs: List[Tuple[Any, Optional[str]]] = []

        while stack:
            node, class_name = stack.pop()
            if node.type == "class_definition":
                cls_name = self._node_name(
                    node, node.tree.source.decode("utf-8", errors="ignore").encode()
                )
                for child in reversed(node.children):
                    stack.append((child, cls_name))
            elif node.type == "function_definition":
                defs.append((node, class_name))
            else:
                for child in reversed(node.children or []):
                    stack.append((child, class_name))

        return defs

    def _node_name(self, node, source_bytes: bytes) -> str:
        """Extract node name from tree-sitter node."""
        name_field = getattr(node, "child_by_field_name", lambda _: None)("name")
        if name_field:
            return source_bytes[
                name_field.start_byte : name_field.end_byte
            ].decode("utf-8")
        return "[anonymous]"

    def _lines_for_overlap(self, lines: List[str], overlap_tokens: int) -> int:
        """Calculate how many trailing lines to overlap based on token budget."""
        tokens = 0
        count = 0
        for line in reversed(lines):
            tokens += self._count_tokens(line)
            count += 1
            if tokens >= overlap_tokens:
                break
        return count

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (good enough for chunk sizing)."""
        return len(re.findall(r"\w+|[^\s\w]", text))

    def _enclosing_class_name(self, node: Any) -> Optional[str]:
        """AST fallback helper to find nearest enclosing class."""
        parent = getattr(node, "parent", None)
        while parent:
            if parent.__class__.__name__ == "ClassDef":
                return getattr(parent, "name", None)
            parent = getattr(parent, "parent", None)
        return None
