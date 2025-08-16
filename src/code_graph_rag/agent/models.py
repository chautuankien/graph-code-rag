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

CandidateLabel = Literal[
    "Project", "Package", "Module", "Class", "Function", "Method", "File", "Folder", "ExternalPackage"
]

class Candidate(BaseModel):
    """One possible mapping for a textual mention."""
    label: CandidateLabel
    id: str                    # canonical identifier (qname / package name / path)
    score: float               # 0..1
    display: str               # human-friendly text

class ResolvedEntity(BaseModel):
    """Final resolution (and ranked alternatives) for source/destination mentions."""
    resolved_label: CandidateLabel | None = None
    resolved_id: str | None = None
    confidence: float = 0.0
    candidates: list[Candidate] = []
    assumption: str | None = None

    resolved_label_dst: CandidateLabel | None = None
    resolved_id_dst: str | None = None
    confidence_dst: float = 0.0
    candidates_dst: list[Candidate] = []

# 1) Allow-list step names (Literal keeps planner outputs type-safe)
StepName = Literal[
    "META", "CALLERS_TOP", "CALLEES_TOP", "IMPORTS", "NEIGHBORHOOD", "PATH",
    "NODE_META", "METHODS_OF_CLASS", "INHERITS_DIRECT", "OVERRIDDEN_BY",
    "ENTRY_FUNCS_BY_KEYWORD", "MODULE_OF_SYMBOL", "MODULES_DEPENDING_ON_EXTERNAL",
    "PROJECT_EXTERNALS", "STATIC_ENRICH"
]

class PlanStep(BaseModel):
    """One atomic retrieval/explain step chosen by the planner.

    Args:
        name: Step identifier from the allow-list.
        params: Bounded parameter bag. No strings of Cypher.
        required: If True, missing result will trigger retry/degrade in 3.4.

    Example:
        PlanStep(name="META", params={"id": "proj.app.main"}, required=True)
    """
    name: StepName
    params: dict[str, Any] = Field(default_factory=dict)
    required: bool = True

class ExplainPlan(BaseModel):
    """Planner output consumed by Phase 3.4.

    Fields:
        steps: Ordered list of steps (deterministic).
        knobs: Global controls with tight bounds (depth/limit/k).

    Notes:
        - Bounds mirror QueryIntent clamps.
        - Deterministic sort expected for test snapshots.
    """
    steps: list[PlanStep]
    knobs: dict[str, int] = Field(default_factory=lambda: {"depth": 2, "limit": 50, "k": 3})

    @field_validator("knobs")
    @classmethod
    def clamp_knobs(cls, v: dict[str, int]) -> dict[str, int]:
        d = dict(v or {})
        def clamp(x, lo, hi): return max(lo, min(hi, int(x)))
        d["depth"] = clamp(d.get("depth", 2), 1, 5)
        d["limit"] = clamp(d.get("limit", 50), 1, 200)
        d["k"] = clamp(d.get("k", 3), 1, 5)
        return d

class PlanExecutionResult(BaseModel):
    """Unified output model for run_plan(), enriched with static metadata and snippet."""

    step: str                      # Step name in ExplainPlan
    label: str | None = None       # Node type (Function, Class, Module, File, ...)
    id: str | None = None          # qualified_name or path
    name: str | None = None        # Short name or path

    # Phase 3.5 static metadata
    docstring: str | None = None
    signature: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    # Code snippet
    snippet: str | None = None

    # For step-specific or adapter-specific extra fields
    extra: dict[str, Any] = {}