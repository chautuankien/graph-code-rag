"""Pytest fixtures and helpers for agent intent tests.

This module provides a lightweight chain stub and a fixture to patch the
LLM pipeline so tests can deterministically control model outputs.
"""

import os, uuid, logging, pytest
from src.code_graph_rag.utils.logging_setup import setup_logging, set_corr_id, pipeline_context

# One-time logging init for THIS FILE
setup_logging(level="DEBUG", log_file="logs/test.log", force=True, is_pytest=True)
set_corr_id("TEST-" + uuid.uuid4().hex[:6])

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
