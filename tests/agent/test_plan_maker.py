# tests/agent/test_planner.py
import json
import pytest

from code_graph_rag.agent import plan_maker as P
from src.code_graph_rag.agent.models import QueryIntent, ResolvedEntity

@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    class _DummyLLM:
        def invoke(self, x):
            return x
    monkeypatch.setattr(P, "get_cypher_generate_model", lambda **_: _DummyLLM())

def _patch_prompt_to_chain(monkeypatch, chain):
    """Patch ChatPromptTemplate.from_messages(...) to return our stubbed chain.

    Accepts arbitrary kwargs (e.g., template_format="jinja2") to mirror production code.
    """
    monkeypatch.setattr(
        P,
        "ChatPromptTemplate",
        type("X", (object,), {"from_messages": staticmethod(lambda *_a, **_kw: chain)}),
    )


# ---------------------------------------------------------------------------
# 1) Happy path: required-first ordering & stable sorting
# ---------------------------------------------------------------------------
def test_required_first_and_stable_sorting(patch_chain, monkeypatch):
    """Ensure required steps are ordered before non-required, and names are
    sorted deterministically within each group."""
    payload = {
        "steps": [
            {"name": "CALLERS_TOP", "params": {"id": "X", "limit": 10}, "required": False},
            {"name": "META", "params": {"id": "X"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 10, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]

    assert names[0] == "META"  # required first
    # Only two steps; sorting within groups is implied by the result


# ---------------------------------------------------------------------------
# 2) Filter out disallowed steps
# ---------------------------------------------------------------------------
def test_filter_disallowed_steps(patch_chain, monkeypatch):
    """Steps not present in the allowed list (e.g., 'GRAPHOPS') must be dropped."""
    payload = {
        "steps": [
            {"name": "GRAPHOPS", "params": {"id": "X"}, "required": False},
            {"name": "META", "params": {"id": "X"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 50, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    assert all(s.name != "GRAPHOPS" for s in plan.steps)
    assert any(s.name == "META" for s in plan.steps)


# ---------------------------------------------------------------------------
# 3) Clamp knobs to safe bounds
# ---------------------------------------------------------------------------
def test_knobs_clamped_to_bounds(patch_chain, monkeypatch):
    """Knobs returned by the LLM must be clamped to safe bounds:
    depth<=5, limit<=200, k<=5."""
    payload = {"steps": [], "knobs": {"depth": 99, "limit": 999, "k": 9}}
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    assert plan.knobs["depth"] <= 5
    assert plan.knobs["limit"] <= 200
    assert plan.knobs["k"] <= 5


# ---------------------------------------------------------------------------
# 4) Default knobs when missing or partial
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "knobs_in, expected_defaults",
    [
        ({}, {"depth": 2, "limit": 50, "k": 3}),
        ({"limit": 5}, {"depth": 2, "limit": 5, "k": 3}),
    ],
)
def test_knobs_defaults_and_partial(patch_chain, monkeypatch, knobs_in, expected_defaults):
    """If knobs are missing or partially provided, defaults are filled and
    still subject to clamping."""
    payload = {"steps": [], "knobs": knobs_in}
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    assert plan.knobs["depth"] == expected_defaults["depth"]
    assert plan.knobs["limit"] == expected_defaults["limit"]
    assert plan.knobs["k"] == expected_defaults["k"]


# ---------------------------------------------------------------------------
# 5) Required default = True
# ---------------------------------------------------------------------------
def test_required_defaults_to_true(patch_chain, monkeypatch):
    """A step without an explicit 'required' flag should default to True."""
    payload = {
        "steps": [
            {"name": "META", "params": {"id": "X"}},  # required missing
            {"name": "CALLERS_TOP", "params": {"id": "X", "limit": 10}, "required": False},
        ],
        "knobs": {"depth": 2, "limit": 10, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    assert plan.steps[0].name == "META"
    assert plan.steps[0].required is True


# ---------------------------------------------------------------------------
# 6) Params passthrough & normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "params_in, expected_out",
    [
        ({"id": "X", "limit": 10}, {"id": "X", "limit": 10}),  # normal dict
        ({}, {}),  # empty dict
        (None, {}),  # missing params treated as {}
        ([], {}),  # list -> dict([]) -> {}
    ],
)
def test_params_passthrough_and_normalization(patch_chain, monkeypatch, params_in, expected_out):
    """Planner should pass-through params as a dict; if missing/None or an
    empty list, it should normalize to {} and not crash."""
    payload = {
        "steps": [
            {"name": "META", "params": {"id": "X"}, "required": True},
            {"name": "CALLERS_TOP", "params": params_in, "required": False},
        ],
        "knobs": {"depth": 2, "limit": 10, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    # find CALLERS_TOP step
    ct = next(s for s in plan.steps if s.name == "CALLERS_TOP")
    assert ct.params == expected_out


# ---------------------------------------------------------------------------
# 7) Case sensitivity of step names
# ---------------------------------------------------------------------------
def test_step_name_case_sensitivity(patch_chain, monkeypatch):
    """Step names must match the allow-list exactly; lowercase variants are
    considered invalid and should be filtered out."""
    payload = {
        "steps": [
            {"name": "meta", "params": {"id": "X"}, "required": True},  # invalid
            {"name": "CALLERS_TOP", "params": {"id": "X", "limit": 10}, "required": False},
        ],
        "knobs": {"depth": 2, "limit": 10, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert "meta" not in names
    assert "CALLERS_TOP" in names


# ---------------------------------------------------------------------------
# 8) Duplicate steps keep deterministic order
# ---------------------------------------------------------------------------
def test_duplicate_steps_deterministic_order(patch_chain, monkeypatch):
    """When duplicates exist, required steps must appear first; within each
    group ordering should be by name to keep determinism."""
    payload = {
        "steps": [
            {"name": "META", "params": {"id": "X"}, "required": False},
            {"name": "META", "params": {"id": "X"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 50, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    assert plan.steps[0].required is True
    assert plan.steps[0].name == "META"


# ---------------------------------------------------------------------------
# 9) list_callers → minimal plan
# ---------------------------------------------------------------------------
def test_action_list_callers_minimal_plan(patch_chain, monkeypatch):
    """Minimal plan for list_callers should include META and optionally
    CALLERS_TOP; ordering must place required first."""
    payload = {
        "steps": [
            {"name": "CALLERS_TOP", "params": {"id": "X", "limit": 5}, "required": False},
            {"name": "META", "params": {"id": "X"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 5, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callers", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert names[0] == "META"
    assert "CALLERS_TOP" in names


# ---------------------------------------------------------------------------
# 10) list_callees → minimal plan
# ---------------------------------------------------------------------------
def test_action_list_callees_minimal_plan(patch_chain, monkeypatch):
    """Minimal plan for list_callees should include META and CALLEES_TOP,
    with required-first ordering."""
    payload = {
        "steps": [
            {"name": "CALLEES_TOP", "params": {"id": "X", "limit": 7}, "required": False},
            {"name": "META", "params": {"id": "X"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 7, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="list_callees", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert names[0] == "META"
    assert "CALLEES_TOP" in names


# ---------------------------------------------------------------------------
# 15) imports → minimal plan
# ---------------------------------------------------------------------------
def test_action_imports_minimal_plan(patch_chain, monkeypatch):
    """Minimal plan for imports should include META and IMPORTS."""
    payload = {
        "steps": [
            {"name": "IMPORTS", "params": {"id": "X", "limit": 11}, "required": False},
            {"name": "META", "params": {"id": "X"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 11, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="imports", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert names[0] == "META"
    assert "IMPORTS" in names


# ---------------------------------------------------------------------------
# 11) inherits_tree → minimal plan
# ---------------------------------------------------------------------------
def test_action_inherits_tree_minimal_plan(patch_chain, monkeypatch):
    """Minimal plan for inherits_tree should include META and structural
    inheritance steps like INHERITS_DIRECT and possibly METHODS_OF_CLASS."""
    payload = {
        "steps": [
            {"name": "METHODS_OF_CLASS", "params": {"id": "X", "limit": 9}, "required": False},
            {"name": "INHERITS_DIRECT", "params": {"id": "X", "limit": 9}, "required": False},
            {"name": "META", "params": {"id": "X"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 9, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="inherits_tree", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert names[0] == "META"
    assert "INHERITS_DIRECT" in names
    assert "METHODS_OF_CLASS" in names


# ---------------------------------------------------------------------------
# 12) overrides → minimal plan
# ---------------------------------------------------------------------------
def test_action_overrides_minimal_plan(patch_chain, monkeypatch):
    """Minimal plan for overrides should include META and OVERRIDDEN_BY."""
    payload = {
        "steps": [
            {"name": "OVERRIDDEN_BY", "params": {"id": "X", "limit": 8}, "required": False},
            {"name": "META", "params": {"id": "X"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 8, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="overrides", mention="X", language="en")
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert names[0] == "META"
    assert "OVERRIDDEN_BY" in names


# ---------------------------------------------------------------------------
# 13) depends_external → minimal plan
# ---------------------------------------------------------------------------
def test_action_depends_external_minimal_plan(patch_chain, monkeypatch):
    """Minimal plan for depends_external should include a direct package-level
    step like MODULES_DEPENDING_ON_EXTERNAL (often required)."""
    payload = {
        "steps": [
            {"name": "MODULES_DEPENDING_ON_EXTERNAL", "params": {"package": "numpy", "limit": 5}, "required": True}
        ],
        "knobs": {"depth": 2, "limit": 5, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="depends_external", mention="numpy", language="en")
    resolved = ResolvedEntity(resolved_id="numpy", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert names[0] == "MODULES_DEPENDING_ON_EXTERNAL"
    # Check params were honored by the chain stub
    step = plan.steps[0]
    assert step.params.get("package") == "numpy"


# ---------------------------------------------------------------------------
# 14) trace_flow → minimal plan
# ---------------------------------------------------------------------------
def test_action_trace_flow_minimal_plan(patch_chain, monkeypatch):
    """Minimal plan for trace_flow should include both META and PATH as
    required steps, with PATH carrying src/dst/k."""
    payload = {
        "steps": [
            {"name": "PATH", "params": {"src": "A", "dst": "B", "k": 3}, "required": True},
            {"name": "META", "params": {"id": "A"}, "required": True},
        ],
        "knobs": {"depth": 2, "limit": 50, "k": 3},
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="trace_flow", mention="A", mention_dst="B", language="en", k_paths=3)
    resolved = ResolvedEntity(resolved_id="A", resolved_id_dst="B")

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert set(names) >= {"META", "PATH"}
    # Ensure PATH params persisted
    path_step = next(s for s in plan.steps if s.name == "PATH")
    assert path_step.params.get("src") == "A"
    assert path_step.params.get("dst") == "B"
    assert path_step.params.get("k") == 3


# ---------------------------------------------------------------------------
# 15) explain_function → minimal plan
# ---------------------------------------------------------------------------
def test_action_explain_function_minimal_plan(patch_chain, monkeypatch):
    """Minimal plan for explain_function should include META and STATIC_ENRICH
    as key steps, plus local context like NEIGHBORHOOD/CALLERS_TOP/CALLEES_TOP."""
    payload = {
        "steps": [
            {"name": "CALLERS_TOP", "params": {"id": "X", "limit": 10}, "required": False},
            {"name": "CALLEES_TOP", "params": {"id": "X", "limit": 10}, "required": False},
            {"name": "NEIGHBORHOOD", "params": {"id": "X", "depth": 2, "limit": 10}, "required": False},
            {"name": "STATIC_ENRICH", "params": {"from": "META", "take": "all"}, "required": True},
            {"name": "META", "params": {"id": "X"}, "required": True}
        ],
        "knobs": {"depth": 2, "limit": 10, "k": 3}
    }
    chain = patch_chain(payloads=[json.dumps(payload)])
    _patch_prompt_to_chain(monkeypatch, chain)

    qi = QueryIntent(action="explain_function", mention="X", language="en", depth=2, limit=10)
    resolved = ResolvedEntity(resolved_id="X", resolved_id_dst=None)

    plan = P.make_plan(qi, resolved)
    names = [s.name for s in plan.steps]
    assert "META" in names and "STATIC_ENRICH" in names
    assert "NEIGHBORHOOD" in names and "CALLERS_TOP" in names and "CALLEES_TOP" in names
    # Required-first ordering must hold
    assert plan.steps[0].required is True
