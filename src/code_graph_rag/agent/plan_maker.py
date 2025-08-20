"""Plan maker: build an ExplainPlan via LLM JSON with guardrails.

This module assembles a constrained prompt (schema, catalog, graph rules),
invokes the LLM to get a structured plan, validates via Pydantic, then
applies an allow-list and deterministic ordering for idempotency.
"""

from __future__ import annotations
import json

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Type
from pydantic import BaseModel, ValidationError

from src.code_graph_rag.agent.llm import run_llm_json
from src.code_graph_rag.agent.models import (
    QueryIntent, ResolvedEntity, ExplainPlan, PlanStep
)
from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

ALLOWED_STEPS: tuple[str, ...] = (
    "META", "CALLERS_TOP", "CALLEES_TOP", "IMPORTS", "NEIGHBORHOOD", "PATH",
    "NODE_META", "METHODS_OF_CLASS", "INHERITS_DIRECT", "OVERRIDDEN_BY",
    "ENTRY_FUNCS_BY_KEYWORD", "MODULE_OF_SYMBOL", "MODULES_DEPENDING_ON_EXTERNAL",
    "PROJECT_EXTERNALS", "STATIC_ENRICH"
)

SCHEMA = """
ExplainPlan JSON Schema:
{
  "steps": [
    {"name": "<one of ALLOWED_STEPS>", "params": {...}, "required": true|false}
  ],
  "knobs": {"depth": 1, "limit": 50, "k": 3}
}

CRITICAL RULES:
1. Return ONLY valid JSON - no explanations, no markdown, no code fences
2. Use EXACT step names from ALLOWED_STEPS list only
3. Set params as key-value objects, do NOT write raw Cypher queries
4. Mark evidence steps (META, ENTRY_FUNCS_BY_KEYWORD, STATIC_ENRICH, PATH) as required=true when they provide core evidence
5. Respect strict bounds: depth≤5, limit≤200, k≤5
6. Required steps should come first, then sort by name alphabetically
"""

CATALOG = """
ALLOWED_STEPS with required parameters:

Get Metadata for a node by ID:
- META{id}: Get metadata for entity
- NODE_META{ids}: Batch metadata for multiple nodes


Relationship Analysis:
- CALLERS_TOP{id, limit}: Find who calls this function/method
- CALLEES_TOP{id, limit}: Find what this function/method calls
- IMPORTS{id, limit}: Find import relationships
- NEIGHBORHOOD{id, depth, limit}: Explore local graph neighborhood

Path & Flow:
- PATH{src, dst, k?}: Find paths between source and destination

Class Structure:
- METHODS_OF_CLASS{id, limit}: Get methods of a class
- INHERITS_DIRECT{id, limit}: Direct inheritance relationships
- OVERRIDDEN_BY{id, limit}: Methods that override this one

Search & Discovery:
- ENTRY_FUNCS_BY_KEYWORD{kw, limit}: Find functions by keyword
- MODULE_OF_SYMBOL{id}: Find containing module
- MODULES_DEPENDING_ON_EXTERNAL{package, limit}: External dependencies
- PROJECT_EXTERNALS{project, limit}: External packages used

Code Content:
- STATIC_ENRICH{path, start_line, end_line}: Get source code content
- STATIC_ENRICH{from, take}: Alternative parameter format
"""

GRAPH_RULES = """
Graph Schema:
- Node Labels: Project, Package, Module, Class, Function, Method, File, Folder, ExternalPackage
- Key Relationships: DEFINES, DEFINES_METHOD, CALLS, IMPORTS, INHERITS, OVERRIDES, DEPENDS_ON_EXTERNAL
- Each node has properties like: name, qualified_name, id, type
"""

def _build_repair_aware_prompt() -> ChatPromptTemplate:
    """Build a prompt template that handles repair hints elegantly."""
    
    system_template = """{instructions}

{repair_section}"""

    user_template = """Query Context:
{user_input}

Generate the ExplainPlan JSON now:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def _format_repair_section(repair_hints: str) -> str:
    """Format repair hints in a prominent way."""
    if not repair_hints or repair_hints.strip() == "":
        return ""
    
    return f"""
🔧 REPAIR REQUIRED - Previous attempt failed:
{repair_hints}

Please fix these issues and ensure your JSON response is valid."""

def make_plan(intent: QueryIntent, resolved: ResolvedEntity, *, provider: str = "openai") -> ExplainPlan:
    """Generate an ExplainPlan via LLM JSON and enforce guardrails.

    Builds a constrained system prompt (schema + catalog + graph rules),
    calls the LLM to produce a structured plan, validates it, then filters to
    the allow-list and sorts deterministically for idempotency.

    Args:
      intent: Parsed query intent.
      resolved: Entity resolution for the question.
      provider: LLM provider identifier used by ``run_llm_json``.

    Returns:
      ExplainPlan: A validated, filtered, and deterministically ordered plan.
    """

    # Build comprehensive system instructions
    system_instructions = "\n".join([
        SCHEMA,
        "-----",
        "STEP CATALOG:",
        CATALOG,
        "-----",
        "GRAPH SCHEMA:",
        GRAPH_RULES,
        "-----",
        "RESPONSE FORMAT: Return only the JSON object with no additional text."
    ])

    # Prepare user context
    user_context = {
        "intent": intent.model_dump(),
        "resolved": resolved.model_dump(),
    }

    # Build prompt template with repair awareness
    prompt = _build_repair_aware_prompt()

    # Custom payload handler that integrates repair hints
    def build_payload(repair_hints: str = "") -> dict[str, Any]:
        return {
            "instructions": system_instructions,
            "repair_section": _format_repair_section(repair_hints),
            "user_input": json.dumps(user_context, ensure_ascii=False, indent=2),
            "_repair_hints": repair_hints  # This will be used by run_llm_json internally
        }

    # 1) LLM structured → ExplainPlan (validated by Pydantic)
    plan: ExplainPlan = run_llm_json(
        prompt=prompt,
        payload=build_payload(),
        schema=ExplainPlan,
        provider=provider,
        max_retries=1,
        max_tokens=2000,
    )
    log.debug("make_plan.raw_plan: %s", plan.model_dump())

    # 2) Apply safety guardrails and deterministic ordering
    plan = _apply_guardrails(plan)
    
    log.info("make_plan.final_plan: %s", plan.model_dump())
    return plan

def _apply_guardrails(plan: ExplainPlan) -> ExplainPlan:
    """Apply safety filtering and deterministic ordering to the plan.
    
    Args:
        plan: Raw plan from LLM
        
    Returns:
        Cleaned and sorted plan
    """
    # Filter and normalize steps
    cleaned_steps: list[PlanStep] = []
    
    for step in plan.steps:
        # Skip disallowed steps
        if step.name not in ALLOWED_STEPS:
            log.warning(f"Filtering disallowed step: {step.name}")
            continue
            
        # Normalize step properties
        step.params = dict(step.params or {})
        step.required = bool(step.required)
        
        # Validate parameter bounds
        if "limit" in step.params:
            step.params["limit"] = min(int(step.params["limit"]), 200)
        if "depth" in step.params:
            step.params["depth"] = min(int(step.params["depth"]), 5)
        if "k" in step.params:
            step.params["k"] = min(int(step.params["k"]), 5)
            
        cleaned_steps.append(step)
    
    # Deterministic sorting: required steps first, then by name
    plan.steps = sorted(cleaned_steps, key=lambda s: (not s.required, s.name))
    
    # Validate and clamp knobs
    if plan.knobs:
        plan.knobs["depth"] = min(max(plan.knobs.get("depth", 2), 1), 5)
        plan.knobs["limit"] = min(max(plan.knobs.get("limit", 50), 1), 200)
        plan.knobs["k"] = min(max(plan.knobs.get("k", 3), 1), 5)
    else:
        plan.knobs = {"depth": 2, "limit": 50, "k": 3}
    
    return plan