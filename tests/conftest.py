"""
Shared fixtures for adapter integration tests.

This module provides:
- DB reachability guard (skip-all when Memgraph is down)
- A reusable loader fixture `load_repo_into_memgraph` that:
    * Creates a temporary repo (files you specify)
    * Runs the real pipeline `build_knowledge_graph_and_insert_db`
    * Clears the DB before load for deterministic tests
"""

import os, uuid, pytest
from pathlib import Path
from contextlib import closing
from typing import Callable
import mgclient
from src.code_graph_rag.utils.logging_setup import setup_logging, set_corr_id, pipeline_context
from src.code_graph_rag.pipeline.build_knowledge_graph import (
    build_knowledge_graph_and_insert_db,
)
from src.code_graph_rag.parser.ast_parser import ASTParser

# One-time logging init for THIS FILE
setup_logging(level="DEBUG", log_file="logs/test.log", force=True, is_pytest=True)
set_corr_id("TEST-" + uuid.uuid4().hex[:6])


def _can_connect() -> bool:
    """Quick connectivity check to Memgraph using MGCLIENT."""
    host = "127.0.0.1"
    port = 7687
    try:
        with closing(mgclient.connect(host=host, port=port)):
            return True
    except Exception:
        return False


@pytest.fixture(autouse=True, scope="session")
def _skip_all_if_no_db():
    """
    Skip the whole adapter integration test session if Memgraph is not reachable.

    WHY:
        These are real integration tests. We avoid long failures and tell the
        user early that a running Memgraph instance is required.
    """
    if not _can_connect():
        pytest.skip("Memgraph is not reachable; set MEMGRAPH_HOST/PORT and start it.")


@pytest.fixture
def load_repo_into_memgraph(tmp_path: Path) -> Callable[[dict[str, str], str | None], Path]:
    """
    Factory fixture to build & load a temporary repo into Memgraph via the
    real pipeline.

    Usage in tests:
        repo_dir = load_repo_into_memgraph(
            {
                "app.py": "def foo(): ...",
                "pkg/__init__.py": "",
                "pkg/a.py": "import pkg.b",
                ...
            },
            project_name="sample"  # optional
        )

    Returns:
        Path to the created repo directory.

    Notes:
        - The DB is cleared (drop all nodes/edges) before loading for
          deterministic, isolated test runs.
        - We do NOT bootstrap indexes by default to keep tests fast. If your
          dataset needs them, turn it on from here.
    """
    def _loader(files: dict[str, str], project_name: str | None = None) -> Path:
        repo_dir = tmp_path / (project_name or "repo")
        # Write files to the repo dir
        for rel, content in files.items():
            p = repo_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        export_path = tmp_path / "export.cypherl"
        host = "127.0.0.1"
        port = 7687

        build_knowledge_graph_and_insert_db(
            repo_path=str(repo_dir),
            export_path=str(export_path),
            host=host,
            port=port,
            clear_db=True,          # isolate each test
            bootstrap_schema=False, # keep fast; adapters use small scans
            bootstrap_file=None,    # no DDL needed for this test
        )
        return repo_dir

    return _loader

@pytest.fixture
def make_parser(tmp_path: Path) -> Callable[[str, str, dict[str, str] | None], tuple[ASTParser, str]]:
    """
    Returns a function:

        (parser) = make_parser(code, extra_files=None)

    It will:
      1) Create a temporary repo whose directory name == project_name == 'proj'
         so that module qualified names match the legacy tests (e.g., 'proj.mod').
      2) Write `mod.py` with given `code`. If `extra_files` provided, write them too
         (e.g., {'mypkg/utils.py': '...'} for absolute import cases).
      3) Create an ASTParser(project_root=repo_dir) and _walk_files_and_dirs
         so we can assert parser edges.

    Returns:
      (ASTParser)
    """
    def _fn(
        code: str,
        extra_files: dict[str, str] | None = None,
    ) -> tuple[ASTParser, str]:
        # 1) Create a repo whose basename is 'proj' → qnames like 'proj.mod'
        repo_dir = tmp_path / "proj"
        repo_dir.mkdir(parents=True, exist_ok=True)

        # 2) Write primary module file
        (repo_dir / "mod.py").write_text(code, encoding="utf-8")

        #    Write any additional files requested (packages, helpers, etc.)
        for rel, content in (extra_files or {}).items():
            p = repo_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        # 3) Run AST parser for assertions (like legacy unit tests)
        parser = ASTParser(project_root=str(repo_dir))
        parser._walk_files_and_dirs()
        return parser

    return _fn

class FakeChain:
    """Minimal stub that mimics a chain with an ``invoke`` method.

    Attributes:
        _payloads: Queue of JSON strings to return on each ``invoke``.
        name: Optional name for debugging output.
        calls: Recorded call history for assertions.
    """

    def __init__(self, payloads, name: str = "main"):
        """Initialize the fake chain with a list of JSON payloads.

        Args:
            payloads: Iterable of JSON-serializable strings to pop per call.
            name: An identifier used in error messages.
        """
        # payloads: list[str] (mỗi phần tử là JSON string)
        self._payloads = list(payloads)
        self.name = name
        self.calls = []  # lưu lịch sử invoke

    def invoke(self, *args, **kwargs):
        """Return the next payload and record call metadata.

        Raises:
            AssertionError: If the fake runs out of payloads.
        """
        self.calls.append({"args": args, "kwargs": kwargs})
        if not self._payloads:
            raise AssertionError(
                f"FakeChain[{self.name}] ran out of payloads.\n"
                f"Calls so far: {self.calls}\n"
                f"Hint: provide 2 payloads if auto-repair is triggered."
            )
        return self._payloads.pop(0)


@pytest.fixture
def patch_chain(monkeypatch):
    """Patch the prompt→LLM→parser pipeline to return controlled JSON.

    This fixture returns a builder that constructs an operator-pipe-compatible
    object where ``.invoke()`` will yield pre-seeded JSON strings. It bypasses
    ``ChatPromptTemplate``, ``StrOutputParser``, and ``get_cypher_generate_model``.

    Args:
        monkeypatch: Pytest monkeypatch fixture injected by pytest.

    Returns:
        Callable[[list[str]], object]: A function that accepts a list of JSON
        strings and returns a pipeline stub compatible with ``prompt | llm |
        StrOutputParser``.
    """
    from src.code_graph_rag.agent import intent as intent_mod

    class _Builder:
        def __init__(self, chain: FakeChain):
            self._chain = chain

        def __or__(self, _):  # support ``prompt | llm | parser`` chaining
            return self

        def invoke(self, *a, **k):
            return self._chain.invoke(*a, **k)

    def _make_chain(payloads):
        # Bypass heavy components. Tests only assert JSON plumbing.
        monkeypatch.setattr(
            intent_mod,
            "ChatPromptTemplate",
            type("X", (object,), {"from_messages": staticmethod(lambda m: None)}),
        )
        monkeypatch.setattr(intent_mod, "StrOutputParser", lambda: None)
        monkeypatch.setattr(intent_mod, "get_cypher_generate_model", lambda: None)
        return _Builder(FakeChain(payloads))

    return _make_chain

@pytest.fixture(autouse=True)
def _pipeline_per_test():
    # Mặc định mỗi test chạy trong context "tests"
    with pipeline_context("tests"):
        yield
