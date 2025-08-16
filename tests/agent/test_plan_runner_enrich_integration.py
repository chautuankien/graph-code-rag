import pytest
from src.code_graph_rag.agent.models import ExplainPlan, PlanStep, QueryIntent, ResolvedEntity
from src.code_graph_rag.agent.plan_runner import run_plan, PlanExecutionError
from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

def _mk_intent(mention: str = "") -> QueryIntent:
    """Create a minimal QueryIntent used by adapters during tests.

    Only the fields required by the adapter pipeline are filled.
    The specific node to query is provided via step.params["id"],
    so "mention" is informational only.
    """
    return QueryIntent(
        action="explain_function",
        mention=mention,
        language="en",
        depth=1,
        limit=50,
        k_paths=3,
    )

def _mk_plan(step_name: str, node_id: str, required: bool = True) -> ExplainPlan:
    """Build a single-step ExplainPlan targeting a specific node id.

    Args:
        step_name: Adapter step to execute (e.g., "META").
        node_id: Qualified name or path to resolve in the graph.
        required: If True, the pipeline must return rows or raise.
    """
    return ExplainPlan(
        steps=[
            PlanStep(
                name=step_name,
                params={"id": node_id},
                required=required,
            )
        ],
        knobs={},
    )

def _normalize_results(results):
    """Normalize run_plan outputs to a list of dicts with a 'step' key."""
    if isinstance(results, dict):
        out = []
        for step, rows in results.items():
            for r in rows:
                d = dict(r)
                d.setdefault("step", step)
                out.append(d)
        return out
    out = []
    for r in results:
        if hasattr(r, "model_dump"):
            d = r.model_dump()
        elif isinstance(r, dict):
            d = dict(r)
        else:
            # best-effort
            d = getattr(r, "__dict__", {})
        out.append(d)
    return out

def _filter_step(rows, step_name: str):
    """Return only rows that match a specific step name."""
    return [r for r in rows if r.get("step") == step_name]


def test_run_plan_with_static_enrich(load_repo_into_memgraph):
    # 1. Load mini repo vào Memgraph thật
    repo_dir = load_repo_into_memgraph({
        "app.py": '''
"""This is app module docstring"""

def hello(name):
    """Say hello"""
    return f"Hello {name}"
'''
    }, project_name="proj")

    # 2. Tạo ExplainPlan có step META
    plan = ExplainPlan(
        steps=[
            PlanStep(
                name="META",
                params={"id": "proj.app.hello"},  # qualified_name module
                required=True
            )
        ],
        knobs={}
    )
    log.debug("test_run_plan_with_static_enrich.plan=%s", plan)

    # 3. Fake intent + resolved entity
    intent = QueryIntent(
        action="explain_function",
        mention="hello",
        mention_dst=None,
        language="en",
        depth=1,
        limit=50,
        k_paths=3
    )
    log.debug("test_run_plan_with_static_enrich.intent=%s", intent)
    resolved = ResolvedEntity(
        resolved_id="proj.app.hello",
        resolved_label="Function"
    )
    log.debug("test_run_plan_with_static_enrich.resolved=%s", resolved)

    # 4. Chạy run_plan() (phiên bản phase 3.5)
    outputs = run_plan(plan=plan, intent=intent, resolved=resolved, repo_root=str(repo_dir))
    log.debug("test_run_plan_with_static_enrich.outputs=%s", outputs)

    # 5. Assert output có enrich metadata + snippet
    assert len(outputs) == 1
    result = outputs[0]
    assert result.step == "META"
    assert result.id == "proj.app.hello"
    assert result.label == "Function"
    assert result.path == "app.py"
    assert result.snippet is not None
    assert "def hello" in result.snippet
    assert "Say hello" in (result.docstring or "") or "This is app" in (result.docstring or "")


def test_01_meta_function_snippet(load_repo_into_memgraph):
    """Integration: META returns enriched metadata and snippet for a Function.

    Setup:
      - Create a tiny repo with a module docstring and a top-level function
        `hello(name)` with its own docstring.
      - Build and insert the graph into Memgraph (fixture).

    Assertions:
      - Row is labeled as Function with the expected qualified id.
      - File `path` and `start_line`/`end_line` are present.
      - `snippet` includes the function header `def hello`.
    """
    code = '''
"""module doc"""

def hello(name):
    """Say hello"""
    return f"Hello {name}"
'''
    repo = load_repo_into_memgraph(
        {"app.py": code},
    )
    qid = "repo.app.hello"
    plan = _mk_plan("META", qid, required=True)
    intent = _mk_intent(qid)
    log.debug("test_01_meta_function_snippet.plan=%s", plan)
    log.debug("test_01_meta_function_snippet.intent=%s", intent)

    res = run_plan(plan=plan, intent=intent, resolved=None, repo_root=repo)
    rows = _normalize_results(res)
    metas = _filter_step(rows, "META")
    log.debug("test_01_meta_function_snippet.metas=%s", metas)

    assert len(metas) == 1, metas
    r = metas[0]
    assert r.get("label") == "Function"
    assert r.get("id") == qid
    assert r.get("path", "").endswith("app.py")
    assert r.get("start_line") and r.get("end_line")
    # snippet should include function header
    assert r.get("snippet"), "Expected snippet on Function"
    assert "def hello" in r["snippet"]

def test_02_meta_method_snippet(load_repo_into_memgraph):
    """Integration: META returns enriched metadata and snippet for a Method.

    Setup:
      - Create a repo with `class C` defining a single method `m(self)`.

    Assertions:
      - Row is labeled as Method with the expected qualified id.
      - `snippet` includes `def m` and the file `path` ends with `mod.py`.
    """
    code = '''
class C:
    def m(self):
        return 1
'''
    repo = load_repo_into_memgraph({"mod.py": code})
    qid = "repo.mod.C.m"
    plan = _mk_plan("META", qid, required=True)
    intent = _mk_intent(qid)
    log.debug("test_02_meta_method_snippet.plan=%s", plan)
    log.debug("test_02_meta_method_snippet.intent=%s", intent)

    res = run_plan(plan=plan, intent=intent, resolved=None, repo_root=repo)
    rows = _normalize_results(res)
    metas = _filter_step(rows, "META")
    log.debug("test_02_meta_method_snippet.metas=%s", metas)

    assert len(metas) == 1
    r = metas[0]
    assert r.get("label") == "Method"
    assert r.get("id") == qid
    assert r.get("path", "").endswith("mod.py")
    assert "def m" in (r.get("snippet") or "")

def test_03_meta_class_snippet(load_repo_into_memgraph):
    """Integration: META returns a bounded snippet for a Class definition.

    Setup:
      - Create a class with two small methods (`a`, `b`).

    Assertions:
      - Row is labeled as Class with qualified id for the class.
      - `snippet` is present and begins with `class C` (after trimming).
    """
    code = '''
class C:
    def a(self): return 1
    def b(self): return 2
'''
    repo = load_repo_into_memgraph({"mod.py": code})
    qid = "repo.mod.C"
    plan = _mk_plan("META", qid, required=True)
    intent = _mk_intent(qid)
    log.debug("test_03_meta_class_snippet.plan=%s", plan)
    log.debug("test_03_meta_class_snippet.intent=%s", intent)

    res = run_plan(plan=plan, intent=intent, resolved=None, repo_root=repo)
    rows = _normalize_results(res)
    metas = _filter_step(rows, "META")
    log.debug("test_03_meta_class_snippet.metas=%s", metas)

    assert len(metas) == 1
    r = metas[0]
    assert r.get("label") == "Class"
    assert r.get("id") == qid
    assert r.get("path", "").endswith("mod.py")
    assert r.get("snippet")  # has some bounded lines
    assert r["snippet"].lstrip().startswith("class C")

# def test_04_decorators_and_async(load_repo_into_memgraph):
#     """Integration: snippet extraction for decorators and async defs.

#     Setup:
#       - Class with `@staticmethod` and `@classmethod` methods.
#       - A top-level `async def af()`.

#     Assertions:
#       - Snippets contain function headers, and async snippet includes `async def`.
#     -> Now, not recognize decorators and async yet, need to fix this later.
#     """
#     code = '''
# class C:
#     @staticmethod
#     def s():
#         return 1

#     @classmethod
#     def c(cls):
#         return 2

# async def af():
#     return 3
# '''
#     repo = load_repo_into_memgraph({"mod.py": code})
#     # staticmethod / classmethod / async function
#     for qid in ("repo.mod.C.s", "repo.mod.C.c"):
#         plan = _mk_plan("META", qid, required=True)
#         intent = _mk_intent(qid)
#         log.debug("test_04_decorators_and_async.plan=%s", plan)
#         log.debug("test_04_decorators_and_async.intent=%s", intent)

#         res = run_plan(plan=plan, intent=intent, resolved=None, repo_root=repo)
#         rows = _normalize_results(res)
#         metas = _filter_step(rows, "META")
#         log.debug("test_04_decorators_and_async.metas=%s", metas)

#         r = metas[0]
#         snip = r.get("snippet") or ""
#         assert "def " in snip
#         if qid.endswith(".s") or qid.endswith(".c"):
#             assert "@staticmethod" in snip or "@classmethod" in snip
#         if qid.endswith(".af"):
#             assert "async def af" in snip


# def test_05_nested_function_and_nested_class(load_repo_into_memgraph):
#     """Integration: nested class/method and nested function handling.

#     Setup:
#       - `Outer.Inner.im` method inside a nested class (should work).
#       - A nested function `inner()` declared inside `outer()` (may be unsupported).

#     Assertions:
#       - Snippet for `Outer.Inner.im` contains `def im`.
#       - Snippet for `outer.inner` is expected to fail on some parsers, hence xfail.
#     """
#     code = '''class Outer:
#     class Inner:
#         def im(self): pass

# def outer():
#     def inner(): return 1
#     return inner()
# '''
#     repo = load_repo_into_memgraph({"mod.py": code})
#     # Nested class method: should work
#     qid1 = "repo.mod.Outer.Inner.im"
#     plan1 = _mk_plan("META", qid1, required=True)
#     res1 = run_plan(plan=plan1, intent=_mk_intent(qid1), resolved=None, repo_root=repo)
#     r1 = _filter_step(_normalize_results(res1), "META")[0]
#     assert "def im" in (r1.get("snippet") or "")
    
#     # Nested function: may or may not be captured depending on parser support
#     qid2 = "repo.mod.outer.inner"
#     plan2 = _mk_plan("META", qid2, required=True)
#     res2 = run_plan(plan=plan2, intent=_mk_intent(qid2), resolved=None, repo_root=repo)
#     r2 = _filter_step(_normalize_results(res2), "META")[0]
#     assert "def inner" in (r2.get("snippet") or "")

def test_06_required_step_empty_raises(load_repo_into_memgraph):
    """Integration: required=True step with empty result must raise.

    Setup:
      - Repo with a simple function `ok()`.
      - Plan queries a non-existent id with required=True.

    Assertions:
      - The pipeline raises PlanExecutionError (strict mode behavior).
    """
    repo = load_repo_into_memgraph({"mod.py": "def ok(): return 1\n"})
    plan = _mk_plan("META", "repo.mod.DOES_NOT_EXIST", required=True)
    with pytest.raises(PlanExecutionError):
        run_plan(plan=plan, intent=_mk_intent("repo.mod.DOES_NOT_EXIST"), resolved=None, repo_root=repo)
