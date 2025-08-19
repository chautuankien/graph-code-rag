from __future__ import annotations

from typing import Any, Literal, Union

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

class Language(str, Enum):
    """UI/prompt language for the agent."""
    vi = "vi"
    en = "en"

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
    "META", "CALLERS_TOP", "CALLEES_TOP", "IMPORTS", "NEIGHBORHOOD",
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

class ValidationReport(BaseModel):
    kept: int = 0
    dropped: int = 0
    reasons: dict[str, int] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


# Base item types cho các action khác nhau
class EntityReference(BaseModel):
    """Reference to a graph entity (Function, Class, etc.)"""
    label: str  # "Function", "Method", "Class", etc.
    id: str     # qualified_name hoặc path
    name: str   # display name

class CallerCalleeItem(BaseModel):
    """Item for list_callers/list_callees actions"""
    caller: EntityReference
    callee: EntityReference
    why: str  # explanation of relationship
    call_count: int | None = None  # optional: frequency

class RelationshipItem(BaseModel):
    """Item for imports/inherits_tree/overrides/depends_external"""
    source: EntityReference
    relationship: str  # "IMPORTS", "INHERITS", "OVERRIDES", etc.
    target: EntityReference
    why: str
    details: dict[str, Any] = {}  # extra metadata

class ExplainFunctionItem(BaseModel):
    """Item for explain_function action"""
    summary: str
    purpose: str
    inputs: list[str]
    outputs: str | None = None
    side_effects: list[str] = []
    exceptions: list[str] = []
    complexity_hint: str | None = None
    snippet_excerpt: str | None = None
    dependencies: list[EntityReference] = []

class FlowStepItem(BaseModel):
    """Single step in trace_flow/impact_analysis"""
    step_id: str
    entity: EntityReference
    operation: str  # "calls", "defines", "imports", etc.
    why: str
    order: int

class FlowPathItem(BaseModel):
    """Complete path for trace_flow/impact_analysis"""
    path_id: str
    steps: list[FlowStepItem]
    confidence: float  # 0.0 - 1.0
    description: str

# Union type cho tất cả item types
SynthesisItem = Union[
    CallerCalleeItem,
    RelationshipItem, 
    ExplainFunctionItem,
    FlowPathItem
]

class Citation(BaseModel):
    """Reference to evidence used in synthesis"""
    label: str
    id: str
    snippet_lines: tuple[int, int] | None = None  # (start, end) if applicable

class SynthesisOutput(BaseModel):
    """Main output model for structured synthesis"""
    action: Action
    language: Language
    answer: str  # concise markdown-safe summary
    items: list[dict[str, Any]]  # structured per action (will validate separately)
    citations: list[Citation]
    notes: str | None = None  # warnings, limitations, etc.
    confidence: float = 1.0  # overall confidence in results
    
    # Metadata
    evidence_count: int = 0
    processing_time_ms: int | None = None

class SynthesisError(BaseModel):
    """Error information when synthesis fails"""
    error_type: str  # "parse_error", "validation_error", "llm_error"
    message: str
    raw_response: str | None = None
    repair_attempted: bool = False