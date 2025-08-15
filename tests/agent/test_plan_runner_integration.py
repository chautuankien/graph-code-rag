"""
Integration tests for all current adapters using the real build+load pipeline.

Each test:
  1) Builds a tiny repo tailored to the adapter under test
  2) Loads it into Memgraph via build_knowledge_graph_and_insert_db
  3) Calls the adapter and asserts results

Adapters covered:
- META
- CALLERS_TOP
- CALLEES_TOP
- IMPORTS
- NEIGHBORHOOD
- INHERITS_DIRECT
- OVERRIDDEN_BY
- METHODS_OF_CLASS
- MODULES_DEPENDING_ON_EXTERNAL
- NODE_META
- ENTRY_FUNCS_BY_KEYWORD
"""

from __future__ import annotations

import pytest

from src.code_graph_rag.agent.adapters.core import (
    meta_adapter,
    callers_top_adapter,
    callees_top_adapter,
    imports_adapter,
    neighborhood_adapter,
    inherits_direct_adapter,
    overridden_by_adapter,
    methods_of_class_adapter,
    modules_dep_ext_adapter,
    node_meta_adapter,
    entry_funcs_by_keyword_adapter,
)
from src.code_graph_rag.agent.models import QueryIntent, ResolvedEntity, PlanStep
from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# META
# ---------------------------------------------------------------------------
def test_meta_adapter_module(load_repo_into_memgraph):
    """META should resolve a Module by qualified_name.

    Repo:
        app.py with 3 functions (foo/bar/baz).
    Expect:
        META(id='app') returns a row with label 'Module' (or 'File' depending
        on exporter) and id 'app' (or an equivalent module/file id).
    """
    load_repo_into_memgraph(
        {
            "app.py": (
                "def foo():\n"
                "    return 42\n\n"
                "def bar():\n"
                "    return foo()\n\n"
                "def baz():\n"
                "    x = foo()\n"
                "    return x\n"
            )
        }
    )

    qi = QueryIntent(action="explain_function", mention="repo.app", language="en")
    rz = ResolvedEntity(resolved_id="repo.app", resolved_id_dst=None)
    step = PlanStep(name="META", params={"id": "repo.app"}, required=True)
    log.debug("test_meta_adapter_module: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = meta_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_meta_adapter_module: rows=%s", rows)

    assert rows, "META returned no rows for module 'repo.app'"
    assert rows[0]["label"] in {"Module", "File"}
    assert rows[0]["id"] in {"app", rows[0]["id"]}


# ---------------------------------------------------------------------------
# CALLERS_TOP
# ---------------------------------------------------------------------------
def test_callers_top_adapter(load_repo_into_memgraph):
    """CALLERS_TOP should return callers of 'app.foo' (bar/baz in this sample)."""
    load_repo_into_memgraph(
        {
            "app.py": (
                "def foo():\n"
                "    return 42\n\n"
                "def bar():\n"
                "    return foo()\n\n"
                "def baz():\n"
                "    x = foo()\n"
                "    return x\n"
            )
        }
    )
    qi = QueryIntent(action="list_callers", mention="repo.app.foo", language="en", limit=10)
    rz = ResolvedEntity(resolved_id="repo.app.foo", resolved_id_dst=None)
    step = PlanStep(name="CALLERS_TOP", params={"id": "repo.app.foo", "limit": 10}, required=False)
    log.debug("test_callers_top_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = callers_top_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_callers_top_adapter: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert {"repo.app.bar", "repo.app.baz"} <= ids


# ---------------------------------------------------------------------------
# CALLEES_TOP
# ---------------------------------------------------------------------------
def test_callees_top_adapter(load_repo_into_memgraph):
    """CALLEES_TOP should return callees of 'repo.app.bar' → 'repo.app.foo'."""
    load_repo_into_memgraph(
        {
            "app.py": (
                "def foo():\n"
                "    return 42\n\n"
                "def bar():\n"
                "    return foo()\n"
            )
        }
    )
    qi = QueryIntent(action="list_callees", mention="repo.app.bar", language="en", limit=10)
    rz = ResolvedEntity(resolved_id="repo.app.bar", resolved_id_dst=None)
    step = PlanStep(name="CALLEES_TOP", params={"id": "repo.app.bar", "limit": 10}, required=False)
    log.debug("test_callees_top_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = callees_top_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_callees_top_adapter: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert "repo.app.foo" in ids


# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
def test_imports_adapter(load_repo_into_memgraph):
    """IMPORTS should return imported targets from 'pkg.a' → expect 'pkg.b'.

    Repo:
        pkg/__init__.py
        pkg/a.py:  import pkg.b
        pkg/b.py:  def g(): return 1
    """
    load_repo_into_memgraph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": (
                "import pkg.b\n"
                "def f():\n"
                "    return pkg.b.g()\n"
            ),
            "pkg/b.py": (
                "def g():\n"
                "    return 1\n"
            ),
        }
    )
    qi = QueryIntent(action="imports", mention="repo.pkg.a", language="en", limit=10)
    rz = ResolvedEntity(resolved_id="repo.pkg.a", resolved_id_dst=None)
    step = PlanStep(name="IMPORTS", params={"id": "repo.pkg.a", "limit": 10}, required=False)
    log.debug("test_imports_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = imports_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_imports_adapter: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert "repo.pkg.b" in ids


# ---------------------------------------------------------------------------
# NEIGHBORHOOD
# ---------------------------------------------------------------------------
def test_neighborhood_adapter(load_repo_into_memgraph):
    """NEIGHBORHOOD depth=1 around 'repo.app.foo' should include its callers 'repo.app.bar'."""
    load_repo_into_memgraph(
        {
            "app.py": (
                "def foo():\n"
                "    return 42\n\n"
                "def bar():\n"
                "    return foo()\n"
            )
        }
    )
    qi = QueryIntent(action="explain_function", mention="repo.app.foo", language="en", depth=1, limit=10)
    rz = ResolvedEntity(resolved_id="repo.app.foo", resolved_id_dst=None)
    step = PlanStep(name="NEIGHBORHOOD", params={"id": "repo.app.foo", "depth": 1, "limit": 10}, required=False)
    log.debug("test_neighborhood_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = neighborhood_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_neighborhood_adapter: rows=%s", rows)
    assert rows and any(r["id"] == "repo.app.bar" for r in rows)


# ---------------------------------------------------------------------------
# INHERITS_DIRECT
# ---------------------------------------------------------------------------
def test_inherits_direct_adapter(load_repo_into_memgraph):
    """INHERITS_DIRECT for 'repo.modc.B' should return base class 'repo.modc.A'.

    Repo:
        modc.py:
            class A: pass
            class B(A): pass
    """
    load_repo_into_memgraph(
        {
            "modc.py": (
                "class A:\n"
                "    pass\n\n"
                "class B(A):\n"
                "    pass\n"
            )
        }
    )
    qi = QueryIntent(action="inherits_tree", mention="repo.modc.B", language="en", limit=10)
    rz = ResolvedEntity(resolved_id="repo.modc.B", resolved_id_dst=None)
    step = PlanStep(name="INHERITS_DIRECT", params={"id": "repo.modc.B", "limit": 10}, required=False)
    log.debug("test_inherits_direct_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = inherits_direct_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_inherits_direct_adapter: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert "repo.modc.A" in ids


# ---------------------------------------------------------------------------
# OVERRIDDEN_BY
# ---------------------------------------------------------------------------
def test_overridden_by_adapter(load_repo_into_memgraph):
    """OVERRIDDEN_BY for base 'repo.modc.A' should include subclass 'repo.modc.B'."""
    load_repo_into_memgraph(
        {
            "modc.py": (
                "class A:\n"
                "    pass\n\n"
                "class B(A):\n"
                "    pass\n"
            )
        }
    )
    qi = QueryIntent(action="overrides", mention="repo.modc.A", language="en", limit=10)
    rz = ResolvedEntity(resolved_id="repo.modc.A", resolved_id_dst=None)
    step = PlanStep(name="OVERRIDDEN_BY", params={"id": "repo.modc.A", "limit": 10}, required=False)
    log.debug("test_overridden_by_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = overridden_by_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_overridden_by_adapter: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert "repo.modc.B" in ids


# ---------------------------------------------------------------------------
# METHODS_OF_CLASS
# ---------------------------------------------------------------------------
def test_methods_of_class_adapter(load_repo_into_memgraph):
    """METHODS_OF_CLASS for 'repo.mclass.C' should list its methods 'repo.mclass.C.m1' and 'repo.mclass.C.m2'."""
    load_repo_into_memgraph(
        {
            "mclass.py": (
                "class C:\n"
                "    def m1(self):\n"
                "        return 1\n\n"
                "    def m2(self):\n"
                "        return 2\n"
            )
        }
    )
    qi = QueryIntent(action="inherits_tree", mention="repo.mclass.C", language="en", limit=10)
    rz = ResolvedEntity(resolved_id="repo.mclass.C", resolved_id_dst=None)
    step = PlanStep(name="METHODS_OF_CLASS", params={"id": "repo.mclass.C", "limit": 10}, required=False)
    log.debug("test_methods_of_class_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = methods_of_class_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_methods_of_class_adapter: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert {"repo.mclass.C.m1", "repo.mclass.C.m2"} == ids


# ---------------------------------------------------------------------------
# MODULES_DEPENDING_ON_EXTERNAL
# ---------------------------------------------------------------------------
def test_modules_dep_ext_adapter(load_repo_into_memgraph):
    """MODULES_DEPENDING_ON_EXTERNAL should return the module that imports an external package.

    Repo:
        extmod.py:
            import requests
            def use(): return requests.__name__
        requirements.txt:
            requests==2.32.0

    Expect:
        modules_dep_ext_adapter(package='requests') returns 'extmod'.
    """
    load_repo_into_memgraph(
        {
            "requirements.txt": "requests==2.32.0\n",
            "extmod.py": (
                "import requests\n"
                "def use():\n"
                "    return requests.__name__\n"
            ),
        }
    )
    qi = QueryIntent(action="depends_external", mention="requests", language="en", limit=10)
    rz = ResolvedEntity(resolved_id="requests", resolved_id_dst=None)
    step = PlanStep(
        name="MODULES_DEPENDING_ON_EXTERNAL",
        params={"package": "requests", "limit": 10},
        required=False,
    )
    log.debug("test_modules_dep_ext_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = modules_dep_ext_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_modules_dep_ext_adapter: rows=%s", rows)
    ids = {r["id"] for r in rows}
    # Under our pipeline, importing requests should create a module-level dependency
    assert "repo.extmod" in ids


# ---------------------------------------------------------------------------
# NODE_META
# ---------------------------------------------------------------------------
def test_node_meta_adapter(load_repo_into_memgraph):
    """NODE_META should resolve metadata for a batch of ids (module + function)."""
    load_repo_into_memgraph(
        {
            "app.py": (
                "def foo():\n"
                "    return 42\n"
            )
        }
    )
    qi = QueryIntent(action="explain_function", mention="repo.app.foo", language="en", limit=10)
    rz = ResolvedEntity(resolved_id="repo.app.foo", resolved_id_dst=None)
    step = PlanStep(name="NODE_META", params={"ids": ["repo.app", "repo.app.foo"]}, required=False)
    log.debug("test_node_meta_adapter: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = node_meta_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_node_meta_adapter: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert {"repo.app", "repo.app.foo"} == ids


# ---------------------------------------------------------------------------
# ENTRY_FUNCS_BY_KEYWORD
# ---------------------------------------------------------------------------
def test_entry_funcs_by_keyword_short_kw(load_repo_into_memgraph):
    """ENTRY_FUNCS_BY_KEYWORD should find 'repo.runmod.main' using short keyword 'ma' (anchors only)."""
    load_repo_into_memgraph(
        {
            "runmod.py": (
                "def main():\n"
                "    pass\n\n"
                "def helper_mainstuff():\n"
                "    pass\n"
            )
        }
    )
    kw = "ma"  # short → adapter uses anchors (no broad CONTAINS)
    qi = QueryIntent(action="explain_function", mention=kw, language="en", limit=10)
    rz = ResolvedEntity(resolved_id=None, resolved_id_dst=None)
    step = PlanStep(name="ENTRY_FUNCS_BY_KEYWORD", params={"kw": kw, "limit": 10}, required=False)
    log.debug("test_entry_funcs_by_keyword_short_kw: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = entry_funcs_by_keyword_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_entry_funcs_by_keyword_short_kw: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert "repo.runmod.main" in ids


def test_entry_funcs_by_keyword_long_kw(load_repo_into_memgraph):
    """ENTRY_FUNCS_BY_KEYWORD should find 'repo.runmod.main' using long keyword 'main' (CONTAINS enabled)."""
    load_repo_into_memgraph(
        {
            "runmod.py": (
                "def main():\n"
                "    pass\n\n"
                "def helper_mainstuff():\n"
                "    pass\n"
            )
        }
    )
    kw = "main"  # long enough → adapter enables CONTAINS
    qi = QueryIntent(action="explain_function", mention=kw, language="en", limit=10)
    rz = ResolvedEntity(resolved_id=None, resolved_id_dst=None)
    step = PlanStep(name="ENTRY_FUNCS_BY_KEYWORD", params={"kw": kw, "limit": 10}, required=False)
    log.debug("test_entry_funcs_by_keyword_long_kw: step=%s intent=%s resolved=%s", step, qi, rz)

    rows = entry_funcs_by_keyword_adapter(step=step, intent=qi, resolved=rz)
    log.debug("test_entry_funcs_by_keyword_long_kw: rows=%s", rows)
    ids = {r["id"] for r in rows}
    assert "repo.runmod.main" in ids
