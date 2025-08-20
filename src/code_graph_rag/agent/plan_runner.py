# src/code_graph_rag/agent/plan_runner.py
from __future__ import annotations
from typing import Any

from src.code_graph_rag.agent.models import ExplainPlan, QueryIntent, ResolvedEntity, PlanStep, PlanExecutionResult
from src.code_graph_rag.agent.adapters import core  # noqa: F401 ensure adapters imported
from src.code_graph_rag.agent.adapters.registry import get as get_adapter
from src.code_graph_rag.agent.utils.utils import enrich_with_static_info
from src.code_graph_rag.agent.utils.code_slice import load_code_slice
from src.code_graph_rag.utils.logging_setup import get_logger

log = get_logger(__name__)

class PlanExecutionError(RuntimeError):
    pass

def run_plan(*, plan: ExplainPlan, intent: QueryIntent, resolved: ResolvedEntity, repo_root: str) -> list[PlanExecutionResult]:
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
    outputs: list[PlanExecutionResult] = []
    for step in plan.steps:
        log.debug("run_plan.step name=%s params=%s required=%s", step.name, step.params, step.required)
        name = step.name
        fn = get_adapter(name)
        log.debug("run_plan.adapter fn=%s", fn.__name__)
        try:
            rows = fn(step=step, intent=intent, resolved=resolved) or []
            log.debug("run_plan.adapter.rows name=%s rows=%s", name, rows)

            for row in rows:
                row = enrich_with_static_info(row)
                log.debug("run_plan.adapter.name=%s meta_row=%s", name, row)
                if (
                    row.get("label") in {"Function", "Method", "Class"}
                    and row.get("path") and row.get("start_line") and row.get("end_line")
                ):
                    snippet = load_code_slice(
                        path=row["path"],
                        start=row["start_line"],
                        end=row["end_line"],
                        repo_root=repo_root,
                    )
                    row["snippet"] = snippet
                # Append as typed model
                outputs.append(PlanExecutionResult(step=step.name, **row))
                
            if step.required and not rows:
                raise PlanExecutionError(f"Required step produced no rows: {name}")
        except Exception as e:
            log.error("adapter.failed name=%s err=%s", name, e)
            if step.required:
                raise PlanExecutionError(f"Required step failed: {name}") from e
            # non-required → continue
            outputs.setdefault(name, [])  # ensure key exists
    return outputs
