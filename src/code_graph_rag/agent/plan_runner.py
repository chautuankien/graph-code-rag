# src/code_graph_rag/agent/plan_runner.py
from __future__ import annotations
from typing import Any

from src.code_graph_rag.agent.models import ExplainPlan, QueryIntent, ResolvedEntity, PlanStep
from src.code_graph_rag.agent.adapters import core  # noqa: F401 ensure adapters imported
from src.code_graph_rag.agent.adapters.registry import get as get_adapter
from src.code_graph_rag.utils.logging_setup import get_logger

log = get_logger(__name__)

class PlanExecutionError(RuntimeError):
    """Raised when a required step fails or returns empty (strict mode)."""

def run_plan(*, plan: ExplainPlan, intent: QueryIntent, resolved: ResolvedEntity) -> dict[str, list[dict[str, Any]]]:
    """Execute an ExplainPlan by dispatching to registered adapters.

    Args:
        plan: ExplainPlan.
        intent: Original QueryIntent (for adapter params).
        resolved: ResolvedEntity (ids).

    Returns:
        Dict mapping step name → list of row dicts (adapter outputs).
        Later phases may transform this into richer context objects.

    Raises:
        PlanExecutionError: if a required step fails or returns empty.
    """
    outputs: dict[str, list[dict[str, Any]]] = {}
    for step in plan.steps:
        log.debug("run_plan.step name=%s params=%s required=%s", step.name, step.params, step.required)
        name = step.name
        fn = get_adapter(name)
        log.debug("run_plan.adapter fn=%s", fn.__name__)
        try:
            rows = fn(step=step, intent=intent, resolved=resolved) or []
            log.debug("run_plan.adapter.rows name=%s rows=%s", name, rows)
            outputs.setdefault(name, []).extend(rows)
            if step.required and not rows:
                raise PlanExecutionError(f"Required step produced no rows: {name}")
        except Exception as e:
            log.error("adapter.failed name=%s err=%s", name, e)
            if step.required:
                raise PlanExecutionError(f"Required step failed: {name}") from e
            # non-required → continue
            outputs.setdefault(name, [])  # ensure key exists
    return outputs
