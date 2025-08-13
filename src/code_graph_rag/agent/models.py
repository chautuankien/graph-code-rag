from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from enum import Enum

class Action(str, Enum):
    list_callers = "list_callers"
    list_callees = "list_callees"
    imports = "imports"
    inherits_tree = "inherits_tree"
    overrides = "overrides"
    depends_external = "depends_external"
    explain_function = "explain_function"
    trace_flow = "trace_flow"
    impact_analysis = "impact_analysis"

Language = Literal["vi", "en"]

Route = Literal["fast", "plan"]


class QueryIntent(BaseModel):
    """Normalized user intent for routing.

    Args:
        action: Intent action (FACT vs EXPLAIN/FLOW/IMPACT families).
        mention: Primary textual mention (e.g., qualified name or symbol).
        mention_dst: Destination mention for flow queries.
        language: Interface/prompt language ("vi" or "en").
        depth: Traversal depth (bounded by router rules).
        limit: Row limit per adapter (bounded by router rules).
        k_paths: Number of alternative paths for trace/flow (bounded).

    Notes:
        - This model is the contract emitted by the intent parser.
        - Bounds are enforced via `clamp_*` during normalization.
    """

    action: Action
    mention: str | None = None
    mention_dst: str | None = None
    language: Language = "vi"
    depth: int = 2
    limit: int = 50
    k_paths: int = Field(default=3, alias="k_paths")

    @field_validator("language")
    @classmethod
    def lang_norm(cls, v: str) -> str:
        return "vi" if v not in ("vi", "en") else v

    @field_validator("depth", "limit", "k_paths")
    @classmethod
    def clamp(cls, v: int, info):
        bounds = {"depth": (1, 5), "limit": (1, 200), "k_paths": (1, 5)}
        lo, hi = bounds[info.field_name]
        return min(max(v, lo), hi)

class Route(str, Enum):
    FAST = "FAST"
    PLAN = "PLAN"


def clamp_int(value: int, low: int, high: int) -> int:
    """Clamp an integer to [low, high]."""
    return max(low, min(high, value))
