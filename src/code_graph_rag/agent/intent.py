"""Intent parsing and routing helpers for the code graph QA agent.

This module provides utilities to transform a natural-language question into a
structured QueryIntent and to decide routing strategy for execution.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import model helpers and types used by the intent parser and router.
from .models import Action, QueryIntent, Route

# Import LLM helper used to generate JSON intents.
from ..agent.llm import get_cypher_generate_model  # reuse your helper

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

JSON_INSTRUCTIONS = """
You are an intent parser for a code graph QA agent.
Return ONLY a valid JSON object with this schema (no prose, no code fences):

{{
  "action": "<one of: list_callers, list_callees, imports, inherits_tree, overrides, depends_external, explain_function, trace_flow, impact_analysis>",
  "mention": "<symbol-or-none>",
  "mention_dst": "<symbol-or-none>",
  "language": "vi|en",
  "depth": 1,
  "limit": 50,
  "k_paths": 3
}}

Rules:
- Infer action from the question.
- If the question is Vietnamese, set "language":"vi", else "en".
- If destination symbol is implied (trace/impact), fill "mention_dst".
- Depth/limit/k_paths: choose sensible defaults if unspecified.
- Do not include any extra fields or comments.
"""

REPAIR_HINT = """
The previous JSON was invalid: {error}.
Please output a corrected JSON that strictly matches the schema.
No prose, no code fences — JSON only.
"""

def _detect_language(text: str) -> str:
    """Detect the likely UI language from the given text.

    Uses a lightweight heuristic that checks for Vietnamese diacritics.

    Args:
        text: The user question or free-form text.

    Returns:
        'vi' if Vietnamese is detected; otherwise 'en'.
    """
    return (
        "vi"
        if any(
            ch in text
            for ch in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        )
        else "en"
    )


def llm_parse_intent(question: str) -> QueryIntent:
    """Parse a natural-language question into a QueryIntent with one retry.

    The function prompts a small model to return a strict JSON object. If the
    first attempt is invalid, it performs a single repair pass with the parse
    error included as a hint.

    Args:
        question: The user's natural language question.

    Returns:
        A validated QueryIntent parsed from the model output.

    Raises:
        json.JSONDecodeError: If both attempts return non-JSON text.
        pydantic.ValidationError: If the JSON does not match the schema.
    """
    log.debug("intent.question: %s", question)
    llm = get_cypher_generate_model()  # small/cheap model is OK
    prompt = ChatPromptTemplate.from_messages(
        [("system", JSON_INSTRUCTIONS), ("user", "{q}")]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"q": question})
    log.debug("intent.raw_first: %s", raw)

    try:
        obj: dict[str, Any] = json.loads(raw)
        # language fallback if the model missed it
        obj.setdefault("language", _detect_language(question))

        qi = QueryIntent.model_validate(obj)
        log.debug("intent.validated_first: %s", qi.model_dump())

        return qi
    except Exception as e:
        log.error("intent.parse_error: %s", e)
        # one-shot repair with error hints
        repair = ChatPromptTemplate.from_messages(
            [("system", JSON_INSTRUCTIONS + REPAIR_HINT), ("user", "{q}")]
        )
        fixed = (repair | llm | StrOutputParser()).invoke(
            {"q": question, "error": str(e)}
        )
        log.debug("intent.raw_repair: %s", fixed)

        obj = json.loads(fixed)
        obj.setdefault("language", _detect_language(question))
        
        qi: QueryIntent = QueryIntent.model_validate(obj)
        log.debug("intent.validated_repair: %s", qi.model_dump())

        return qi

def decide_route(intent: QueryIntent) -> Route:
    """Choose a fast or plan route based on the intent's action.

    Args:
        intent: The parsed query intent.

    Returns:
        Route: FAST for simple graph lookups; PLAN otherwise.
    """
    fast_actions = {
        Action.list_callers,
        Action.list_callees,
        Action.imports,
        Action.inherits_tree,
        Action.overrides,
        Action.depends_external,
    }
    return Route.FAST if intent.action in fast_actions else Route.PLAN
