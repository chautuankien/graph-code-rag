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
from ..agent.llm import run_llm_json

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

# Generate action list dynamically from the Action enum
def _get_action_values() -> str:
    """Get all valid action values from the Action enum."""
    try:
        return ", ".join([action.value for action in Action])
    except Exception:
        # Fallback to hardcoded list if enum introspection fails
        return "list_callers, list_callees, imports, inherits_tree, overrides, depends_external, explain_function, trace_flow, impact_analysis"

JSON_INSTRUCTIONS = f"""
You are an intent parser for a code graph QA agent.
Analyze the user's question and return ONLY a valid JSON object matching this EXACT schema:

{{
  "action": "<one of: {_get_action_values()}>",
  "mention": "<symbol-name-or-null>",
  "mention_dst": "<destination-symbol-or-null>", 
  "language": "vi|en",
  "depth": <number>,
  "limit": <number>,
  "k_paths": <number>
}}

CRITICAL RULES:
1. Return ONLY the JSON object - no explanations, no markdown, no code fences
2. Choose the most appropriate action based on what the user is asking
3. Set language to "vi" if question contains Vietnamese text, otherwise "en"
4. For flow/trace questions, set both mention (source) and mention_dst (destination)
5. Use null (not "null" string) for missing symbols
6. Set reasonable defaults: depth=2, limit=50, k_paths=3
7. Bounds: depth≤5, limit≤200, k_paths≤10

ACTION GUIDE:
- list_callers: "Who calls this function?"
- list_callees: "What does this function call?"
- imports: "What does this module import?"
- inherits_tree: "Class inheritance hierarchy"
- overrides: "Method overriding relationships"
- depends_external: "External dependencies"
- explain_function: "How does this function work?"
- trace_flow: "How does data flow from A to B?"
- impact_analysis: "What would be affected if I change this?"
"""

def _detect_language(text: str) -> str:
    """Detect the likely UI language from the given text.

    Uses a lightweight heuristic that checks for Vietnamese diacritics.

    Args:
        text: The user question or free-form text.

    Returns:
        'vi' if Vietnamese is detected; otherwise 'en'.
    """
    vietnamese_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    return "vi" if any(ch in text.lower() for ch in vietnamese_chars) else "en"

def _build_intent_prompt() -> ChatPromptTemplate:
    """Build a prompt template optimized for intent parsing with repair support."""
    
    system_template = """{instructions}

{repair_section}"""

    user_template = """Question: {question}

Parse this into a QueryIntent JSON:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def _format_repair_section(repair_hints: str) -> str:
    """Format repair hints for intent parsing errors."""
    if not repair_hints or repair_hints.strip() == "":
        return ""
    
    return f"""
⚠️ REPAIR NEEDED - Previous JSON was invalid:
{repair_hints}

Please correct the JSON format and ensure it matches the schema exactly."""

def llm_parse_intent(question: str, provider: str = "nvidia") -> QueryIntent:
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
    log.debug("llm_parse_intent.question: %s", question)

    # Build the prompt template
    prompt = _build_intent_prompt()
    
    # Custom payload builder for repair integration
    def build_payload(repair_hints: str = "") -> dict[str, Any]:
        return {
            "instructions": JSON_INSTRUCTIONS,
            "repair_section": _format_repair_section(repair_hints),
            "question": question,
            "_repair_hints": repair_hints  # Used by run_llm_json internally
        }
    
    # Parse intent with structured output
    intent = run_llm_json(
        prompt=prompt,
        payload=build_payload(),
        schema=QueryIntent,
        provider=provider,  # use the specified provider
        temperature=0.0,  # deterministic output
        max_tokens=300,  # enough for a full intent
        max_retries=2,  # two attempts with repair
    )
    log.debug("llm_parse_intent.raw_intent: %s", intent.model_dump())

    # Apply post-processing and validation
    intent = _post_process_intent(intent, question)
    
    log.info("llm_parse_intent.final_intent: %s", intent.model_dump())
    return intent

def _post_process_intent(intent: QueryIntent, original_question: str) -> QueryIntent:
    """Apply post-processing and validation to the parsed intent.
    
    Args:
        intent: Raw intent from LLM
        original_question: Original user question for fallback detection
        
    Returns:
        Processed and validated intent
    """
    # Language fallback detection
    if not getattr(intent, "language", None) or intent.language not in ["vi", "en"]:
        detected_lang = _detect_language(original_question)
        log.debug(f"Language fallback: {intent.language} -> {detected_lang}")
        intent.language = detected_lang

    # Validate and clamp bounds
    if hasattr(intent, 'depth'):
        intent.depth = max(1, min(intent.depth, 5))
    if hasattr(intent, 'limit'):
        intent.limit = max(1, min(intent.limit, 200))
    if hasattr(intent, 'k_paths'):
        intent.k_paths = max(1, min(intent.k_paths, 10))

    # Normalize mentions (handle common LLM mistakes)
    if hasattr(intent, 'mention') and intent.mention:
        intent.mention = str(intent.mention).strip()
        if intent.mention.lower() in ["none", "null", ""]:
            intent.mention = None
            
    if hasattr(intent, 'mention_dst') and intent.mention_dst:
        intent.mention_dst = str(intent.mention_dst).strip()
        if intent.mention_dst.lower() in ["none", "null", ""]:
            intent.mention_dst = None

    return intent

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
