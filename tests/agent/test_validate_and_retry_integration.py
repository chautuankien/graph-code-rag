import pytest
from pathlib import Path

# Core models & runner
from src.code_graph_rag.agent.models import (
    QueryIntent,
    ResolvedEntity,
    ExplainPlan,
    PlanStep,
    Action,
)
from src.code_graph_rag.agent.plan_runner import run_plan
import src.code_graph_rag.agent.graph_agent as ga

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

GraphState = ga.GraphState

@pytest.mark.integration
def test_01_validate_and_retry_node_happy_path(load_repo_into_memgraph):
    """
    Integration: real Memgraph + real plan_runner + validator node (phase 3.6)

    Dataset:
        repo (project_name='proj')
        - app.py
            from utils import helper
            def main():
                helper()
        - utils.py
            def helper():
                return 42

    Expectation:
        - run_plan trả về rows cho META và CALLEES_TOP của 'proj.app.main'
        - validate_and_retry_node tạo validated_rows != []
        - mỗi row sau validate có label & id (schema-ok), không có drop
    """
    # 1) Build a tiny repo and load into Memgraph (real pipeline)
    repo_dir = load_repo_into_memgraph(
        {
            "app.py": (
                "from utils import helper\n"
                "def main():\n"
                "    helper()\n"
            ),
            "utils.py": (
                "def helper():\n"
                "    return 42\n"
            ),
        },
        project_name="proj",
    )

    # 2) Craft intent & resolved target → function 'proj.app.main'
    intent = QueryIntent(
        action=Action.list_callees,
        mention="proj.app.main",
        mention_dst=None,
        language="en",
        depth=1,
        limit=10,
        k_paths=3,
    )
    resolved = ResolvedEntity(
        label="Function",
        resolved_id="proj.app.main",
        name="main",
    )
    log.debug("test_validate_and_retry_node_happy_path.intent: %s", intent)
    log.debug("test_validate_and_retry_node_happy_path.resolved: %s", resolved)

    # 3) Make a deterministic plan (không phụ thuộc LLM)
    #    - META: enrich static info of the function
    #    - CALLEES_TOP: top callees from CALLS edges
    plan = ExplainPlan(
        steps=[
            PlanStep(name="META", params={"id": resolved.resolved_id}, required=True),
            PlanStep(
                name="CALLEES_TOP",
                params={"id": resolved.resolved_id, "limit": 10},
                required=True,
            ),
        ],
        knobs={},
    )
    log.debug("Plan: %s", plan)

    # 4) Run plan against the real DB
    plan_rows = run_plan(
        plan=plan, intent=intent, resolved=resolved, repo_root=str(repo_dir)
    )
    log.debug("test_validate_and_retry_node_happy_path.plan_outputs: %s", plan_rows)

    # 5) Prepare GraphState for the node
    state = GraphState(
        repo_root=str(repo_dir),
        question="Who does main() call?",
        intent=intent,
        resolve=resolved,
        plan=plan,
        plan_outputs=plan_rows,
    )

    # 5) Node under test
    new_state = ga.validate_and_retry_node(state)
    log.debug("test_validate_and_retry_node_happy_path.validated_rows: %s", new_state.validated_rows)
    log.debug("test_validate_and_retry_node_happy_path.validation_report: %s", new_state.validation_report)

    # 6) Assertions: validated rows must exist & be schema-cleaned
    validated = getattr(new_state, "validated_rows", None)
    assert validated, "validated_rows should be non-empty after validation"

    for row in validated:
        assert row.label, "each row must have a label"
        assert row.id, "each row must have an id"
        # snippet/docstring may be optional, đừng assert cứng

    report = getattr(new_state, "validation_report", None)
    if report:
        assert getattr(report, "dropped", 0) == 0, f"unexpected drops: {report}"

@pytest.mark.integration
def test_02_dedupe_keeps_single_row_for_duplicate_symbol(load_repo_into_memgraph):
    """
    Dedupe: when two steps return the same (label, id), validator must keep a single row.

    Setup:
      - main() calls helper()
      - CALLEES_TOP(main) returns Function helper
      - META(helper) returns the same Function helper => duplicate

    Assert:
      - Exactly one (Function, proj.utils.helper) remains after validate_and_retry_node.
    """
    repo_dir = load_repo_into_memgraph(
        {
            "app.py": (
                "from utils import helper\n"
                "def main():\n"
                "    helper()\n"
            ),
            "utils.py": (
                "\"\"\"utility module\"\"\"\n"
                "def helper():\n"
                "    \"\"\"docstring makes META richer (if adapters surface it)\"\"\"\n"
                "    return 42\n"
            ),
        },
        project_name="proj",
    )

    intent = QueryIntent(
        action=Action.list_callees,
        mention="proj.app.main",
        mention_dst=None,
        language="en",
        depth=1,
        limit=10,
        k_paths=3,
    )
    resolved = ResolvedEntity(label="Function", resolved_id="proj.app.main", name="main")
    log.debug("test_02_dedupe_keeps_single_row_for_duplicate_symbol.intent: %s", intent)
    log.debug("test_02_dedupe_keeps_single_row_for_duplicate_symbol.resolved: %s", resolved)

    plan = ExplainPlan(
        steps=[
            # returns helper as a callee
            PlanStep(name="CALLEES_TOP", params={"id": resolved.resolved_id, "limit": 10}, required=True),
            # returns helper explicitly (duplicate on purpose)
            PlanStep(name="META", params={"id": "proj.utils.helper"}, required=True),
        ],
        knobs={},
    )

    plan_outputs = run_plan(plan=plan, intent=intent, resolved=resolved, repo_root=str(repo_dir))
    log.debug("test_02_dedupe_keeps_single_row_for_duplicate_symbol.plan_outputs: %s", plan_outputs)

    state = GraphState(
        repo_root=str(repo_dir),
        question="Check dedupe for helper",
        intent=intent,
        resolve=resolved,
        plan=plan,
        plan_outputs=plan_outputs,
    )

    new_state = ga.validate_and_retry_node(state)
    log.debug("test_02_dedupe_keeps_single_row_for_duplicate_symbol.validated_rows: %s", new_state.validated_rows)
    log.debug("test_02_dedupe_keeps_single_row_for_duplicate_symbol.validation_report: %s", new_state.validation_report)

    validated = new_state.validated_rows
    assert validated, "validated_rows should not be empty"

    # Count occurrences of (Function, proj.utils.helper) after validation
    helper_id = "proj.utils.helper"
    count = sum(1 for r in validated if r.label == "Function" and r.id == helper_id)
    assert count == 1, f"Expected exactly 1 row for helper after dedupe, got {count}"

@pytest.mark.integration
def test_03_drop_invalid_row_missing_core_fields(load_repo_into_memgraph, monkeypatch):
    """
    Validation: rows missing core fields (label or id) must be dropped.

    We inject an optional step 'BROKEN' whose adapter returns a dict missing 'id'.
    Required step 'META(main)' ensures the plan still succeeds overall.
    """
    from src.code_graph_rag.agent import plan_runner as pr

    repo_dir = load_repo_into_memgraph(
        {
            "app.py": (
                "def main():\n"
                "    return 1\n"
            ),
        },
        project_name="proj",
    )

    intent = QueryIntent(
        action=Action.explain_function,
        mention="proj.app.main",
        mention_dst=None,
        language="en",
        depth=1,
        limit=10,
        k_paths=1,
    )
    resolved = ResolvedEntity(label="Function", resolved_id="proj.app.main", name="main")
    log.debug("test_03_drop_invalid_row_missing_core_fields.intent: %s", intent)
    log.debug("test_03_drop_invalid_row_missing_core_fields.resolved: %s", resolved)

    # Use PlanStep.model_construct to bypass Pydantic validation for the custom step name.
    # This allows us to create a step named 'BROKEN_SPAN' without touching the production schema.
    broken_step = PlanStep.model_construct(name="BROKEN", params={}, required=False)
    plan = ExplainPlan(
        steps=[
            PlanStep(name="META", params={"id": resolved.resolved_id}, required=True),
            broken_step,
        ],
        knobs={},
    )

    # Monkeypatch plan_runner.get_adapter to serve a broken adapter for 'BROKEN'
    real_get = pr.get_adapter

    def fake_get(name: str):
        if name == "BROKEN":
            def broken_adapter(*, step, intent, resolved):
                # Missing 'id' on purpose -> should be dropped by validator
                return [{"label": "Function"}]
            return broken_adapter
        return real_get(name)

    monkeypatch.setattr(pr, "get_adapter", fake_get)

    plan_outputs = run_plan(plan=plan, intent=intent, resolved=resolved, repo_root=str(repo_dir))
    log.debug("test_03_drop_invalid_row_missing_core_fields.plan_outputs: %s", plan_outputs)

    state = GraphState(
        repo_root=str(repo_dir),
        question="Drop invalid rows missing core fields",
        intent=intent,
        resolve=resolved,
        plan=plan,
        plan_outputs=plan_outputs,
    )

    new_state = ga.validate_and_retry_node(state)
    log.debug("test_03_drop_invalid_row_missing_core_fields.validated_rows: %s", new_state.validated_rows)
    log.debug("test_03_drop_invalid_row_missing_core_fields.validation_report: %s", new_state.validation_report)

    report = getattr(new_state, "validation_report", None)
    assert report is not None
    assert getattr(report, "dropped", 0) >= 1, "Expected at least one row to be dropped as invalid"
    assert getattr(report, "reasons", {}).get("missing-core-fields", 0) == 1, "Expected 'missing-core-fields' reason to be recorded"
    assert new_state.validated_rows, "Validated rows should still contain META(main) output"

@pytest.mark.integration
def test_04_drop_row_with_bad_span(load_repo_into_memgraph, monkeypatch):
    """
    Validation: rows with invalid source span (start_line/end_line) must be dropped.

    We inject an optional step 'BROKEN_SPAN' returning a row for utils.helper with
    start_line > end_line and a valid file path. Validator must discard it.
    """
    from src.code_graph_rag.agent import plan_runner as pr

    repo_dir = load_repo_into_memgraph(
        {
            "app.py": (
                "from utils import helper\n"
                "def main():\n"
                "    helper()\n"
            ),
            "utils.py": (
                "def helper():\n"
                "    \"\"\"docstring makes META richer (if adapters surface it)\"\"\"\n"
                "    return 42\n"
            ),
        },
        project_name="proj",
    )

    file_path = str(Path(repo_dir, "utils.py"))
    intent = QueryIntent(
        action=Action.list_callees,
        mention="proj.app.main",
        mention_dst=None,
        language="en",
        depth=1,
        limit=10,
        k_paths=1,
    )
    resolved = ResolvedEntity(label="Function", resolved_id="proj.app.main", name="main")
    log.debug("test_04_drop_row_with_bad_span.intent: %s", intent)
    log.debug("test_04_drop_row_with_bad_span.resolved: %s", resolved)

    # Use PlanStep.model_construct to bypass Pydantic validation for the custom step name.
    # This allows us to create a step named 'BROKEN_SPAN' without touching the production schema.
    broken_step = PlanStep.model_construct(name="BROKEN_SPAN", params={}, required=False)
    plan = ExplainPlan(
        steps=[
            PlanStep(name="META", params={"id": resolved.resolved_id}, required=True),
            broken_step,  # custom step added without validation
        ],
        knobs={},
    )

    real_get = pr.get_adapter

    def fake_get(name: str):
        if name == "BROKEN_SPAN":
            def broken_span_adapter(*, step, intent, resolved):
                # Invalid span: start_line > end_line
                return [{
                    "label": "Function",
                    "id": "repo.utils.helper",  # fake qualified_name
                    "name": "helper",
                    "path": file_path,
                    "start_line": 21,
                    "end_line": 11,
                }]
            return broken_span_adapter
        return real_get(name)

    monkeypatch.setattr(pr, "get_adapter", fake_get, raising=True)

    plan_outputs = run_plan(plan=plan, intent=intent, resolved=resolved, repo_root=str(repo_dir))
    log.debug("test_04_drop_row_with_bad_span.plan_outputs: %s", plan_outputs)

    state = GraphState(
        repo_root=str(repo_dir),
        question="Drop rows with bad span",
        intent=intent,
        resolve=resolved,
        plan=plan,
        plan_outputs=plan_outputs,
    )

    new_state = ga.validate_and_retry_node(state)
    log.debug("test_04_drop_row_with_bad_span.validated_rows: %s", new_state.validated_rows)
    log.debug("test_04_drop_row_with_bad_span.validation_report: %s", new_state.validation_report)

    report = getattr(new_state, "validation_report", None)

    assert report is not None
    assert getattr(report, "dropped", 0) >= 1, "Expected at least one row to be dropped for bad span"
    assert getattr(report, "reasons", {}).get("bad-span", 0) == 1, "Expected 'bad_span' reason to be recorded"
    # Ensure no validated row carries the invalid helper span
    validated = new_state.validated_rows
    bad = [r for r in validated if r.label == "Function" and r.id == "repo.utils.helper" and getattr(r, "start_line", 0) > getattr(r, "end_line", 0)]
    assert not bad, "Invalid span row should not survive validation"


