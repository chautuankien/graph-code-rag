from __future__ import annotations
import json

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.code_graph_rag.agent.llm import get_cypher_generate_model
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
Return ONLY a JSON object:

{
  "steps": [
    {"name": "<one of ALLOWED_STEPS>", "params": {...}, "required": true|false}
  ],
  "knobs": {"depth": 1, "limit": 50, "k": 3}
}

Rules:
- JSON only (no prose, no code fences).
- Use only names from ALLOWED_STEPS.
- Bind params, do NOT write Cypher.
- Mark evidence-producing steps (META, ENTRY_FUNCS_BY_KEYWORD, STATIC_ENRICH, PATH) as required when appropriate.
- Respect bounds: depth<=5, limit<=200, k<=5.
- Deterministic ordering: required steps first; then by name ASC.
"""

CATALOG = """
ALLOWED_STEPS:
- META{id}
- CALLERS_TOP{id, limit}
- CALLEES_TOP{id, limit}
- IMPORTS{id, limit}
- NEIGHBORHOOD{id, depth, limit}
- PATH{src, dst, k?}
- NODE_META{ids}
- METHODS_OF_CLASS{id, limit}
- INHERITS_DIRECT{id, limit}
- OVERRIDDEN_BY{id, limit}
- ENTRY_FUNCS_BY_KEYWORD{kw, limit}
- MODULE_OF_SYMBOL{id}
- MODULES_DEPENDING_ON_EXTERNAL{package, limit}
- PROJECT_EXTERNALS{project, limit}
- STATIC_ENRICH{path, start_line, end_line} OR {from, take}
"""

GRAPH_RULES = "Labels: Project, Package, Module, Class, Function, Method, File, Folder, ExternalPackage. Edges (subset): DEFINES, DEFINES_METHOD, CALLS, IMPORTS, INHERITS, OVERRIDES, DEPENDS_ON_EXTERNAL."


def make_plan(intent: QueryIntent, resolved: ResolvedEntity) -> ExplainPlan:
    """Generate an ExplainPlan using an LLM with strict post-validation.

    The function constructs system and user prompts, invokes the configured
    LLM to obtain a JSON plan, then validates and normalizes the result. It
    never emits Cypher; only JSON metadata is produced.

    Args:
      intent: Parsed query intent provided by the intent router.
      resolved: Entity resolution results (e.g., qualified_name) for symbols.

    Returns:
      ExplainPlan: A plan with allowed steps sorted deterministically and
      optional knobs carried through.

    Example:
      plan = make_plan(intent, resolved)
      for step in plan.steps:
          ...
    """
    system_prompt = "\n".join([
        SCHEMA,
        "-----",
        "CATALOG:",
        CATALOG,
        "-----",
        "GRAPH_RULES:",
        GRAPH_RULES,
    ])
    user_payload = json.dumps({
        "intent": intent.model_dump(),
        "resolved": resolved.model_dump(),
    }, ensure_ascii=False)

    llm = get_cypher_generate_model(temperature=0.0, seed=42, json_mode=True, max_tokens=800)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{{ u }}")],
        template_format="jinja2",
    )
    # log.debug("make_plan.prompt:\n %s", prompt.messages[1].content)
    raw = (prompt | llm | StrOutputParser()).invoke({"u": user_payload})
    log.debug("make_plan.raw: %s", raw)

    # SECURITY/DETERMINISM: Filter to the allow-list, drop unknown steps,
    # coerce params to dicts, default required, then sort with required
    # steps first and name ASC to keep outputs stable.
    # TODO(maintainers): Clamp knobs to bounds (depth<=5, limit<=200, k<=5).
    data = json.loads(raw)
    # Hard safety: filter/normalize
    steps = data.get("steps") or []
    cleaned: list[PlanStep] = []
    for s in steps:
        name = str(s.get("name", "")).strip()
        if name not in ALLOWED_STEPS:
            continue
        params = dict(s.get("params") or {})
        required = bool(s.get("required", True))
        cleaned.append(PlanStep(name=name, params=params, required=required))

    knobs = dict(data.get("knobs") or {})
    plan = ExplainPlan(
        steps=sorted(cleaned, key=lambda s: (not s.required, s.name)),
        knobs=knobs,
    )
    log.debug("make_plan.explain_plan: %s", plan.model_dump())
    return plan
