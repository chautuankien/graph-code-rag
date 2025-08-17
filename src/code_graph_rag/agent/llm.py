"""LLM helpers for intent parsing, plan making, and JSON-structured IO.

Provides small wrappers to configure chat models and to invoke them with
structured (Pydantic) outputs. Defaults favor determinism and safety.
"""

from __future__ import annotations
import os
import json
from typing import Type, Any
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel

from dotenv import load_dotenv
load_dotenv()

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

class LLMJsonError(RuntimeError):
    """Raised when structured JSON output fails after all retries."""

def get_json_llm(
    *,
    provider: str = "nvidia",
    temperature: float = 0.0,
    max_tokens: int | None = 512,
    seed: int | None = 42,
    json_mode: bool = True,
) -> BaseChatModel:
    """Return a chat model configured for JSON-oriented tasks.

    Supports OpenAI and NVIDIA backends. The caller can further constrain the
    output using ``with_structured_output(schema)``.

    Args:
      provider: Backend provider identifier ("openai" or "nvidia").
      temperature: Sampling temperature.
      max_tokens: Optional response token cap.
      seed: Random seed, when supported.
      json_mode: If True, request JSON-formatted responses when available.

    Returns:
      BaseChatModel: Configured model for JSON tasks.

    Raises:
      AssertionError: If ``provider='nvidia'`` but NVIDIA endpoints are missing.
    """
    if provider == "nvidia":
        assert ChatNVIDIA is not None, "langchain_nvidia_ai_endpoints not installed"
        model = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b")
        
        # NVIDIA-specific configuration
        model_kwargs: dict[str, Any] = {}
        if seed is not None:
            model_kwargs["seed"] = seed
        # Note: NVIDIA endpoints may not support response_format like OpenAI
        
        return ChatNVIDIA(
            model=model, 
            temperature=temperature,
            max_tokens=max_tokens, 
            model_kwargs=model_kwargs if model_kwargs else None
        )
    
    elif provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-nano")  # Valid model name
        
        # OpenAI-specific configuration
        model_kwargs: dict[str, Any] = {}
        if seed is not None:
            model_kwargs["seed"] = seed
        if json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}
        
        return ChatOpenAI(
            model=model, 
            temperature=temperature,
            max_tokens=max_tokens, 
            model_kwargs=model_kwargs if model_kwargs else None
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def _extract_content(response: Any) -> str:
    """Extract text content from LLM response, handling different response formats."""
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    elif isinstance(response, str):
        return response
    else:
        return str(response)

def _parse_json_response(content: str, schema: Type[BaseModel]) -> BaseModel:
    """Parse JSON content and validate against Pydantic schema.
    
    Args:
        content: Raw text response from LLM
        schema: Target Pydantic model
        
    Returns:
        Validated Pydantic model instance
        
    Raises:
        ValidationError: If JSON parsing or validation fails
    """
    content = content.strip()
    
    # Extract JSON from markdown code blocks if present
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end != -1:
            content = content[start:end].strip()
    
    # Extract JSON object if embedded in text
    elif "{" in content and "}" in content:
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx + 1]
    
    try:
        json_data = json.loads(content)
        return schema.model_validate(json_data)
    except json.JSONDecodeError as e:
        raise ValidationError.from_exception_data(
            "JSONDecodeError",
            [{
                "type": "json_invalid",
                "input": content[:200] + "..." if len(content) > 200 else content,
                "ctx": {"error": str(e)}
            }]
        )

def run_llm_json(
    *,
    prompt: BasePromptTemplate,
    payload: dict[str, Any],
    schema: Type[BaseModel],
    provider: str = "openai",           # "openai" | "nvidia"
    temperature: float = 0.0,
    max_tokens: int | None = 1000,
    seed: int | None = 42,
    max_retries: int = 2,
) -> BaseModel:
    """Invoke a chat model and parse structured JSON into a Pydantic model.

    Primary path uses ``with_structured_output(schema)``. On validation error,
    a minimal one-pass repair injects parse errors back into the prompt.

    Args:
      prompt: Assembled prompt template.
      payload: Variables for the prompt.
      schema: Target Pydantic model for structured parsing.
      provider: Backend provider ("openai" or "nvidia").
      temperature: Sampling temperature.
      max_tokens: Optional response token cap.
      seed: Random seed, when supported.
      max_retries: Max attempts before failing.

    Returns:
      BaseModel: Parsed instance of ``schema``.

    Raises:
      LLMJsonError: If parsing fails after the retry budget is exhausted.
    """
    log.debug("run_llm_json.payload: %s", payload)

    llm = get_json_llm(provider=provider, temperature=temperature,
                    max_tokens=max_tokens, seed=seed, json_mode=True)
    chain: Runnable = prompt | llm

    errs: list[str] = []
    for attempt in range(max_retries):
        try:
            log.debug(f"Attempt {attempt + 1}/{max_retries}")

            # Get response from LLM
            response = chain.invoke(payload)
            content = _extract_content(response)
            log.debug(f"Raw response content: {content[:500]}...")
            
            # Parse JSON manually
            parsed = _parse_json_response(content, schema)
            log.debug("Successfully parsed JSON: %s", parsed)
            return parsed

        except ValidationError as ve:
            error_msg = str(ve.errors())
            errs.append(error_msg)
            log.debug(f"Validation error on attempt {attempt + 1}: {ve.errors()}")
            
            # Add repair hints to payload for next attempt
            payload = {
                **payload, 
                "_repair_hints": f"Previous attempt failed with errors: {error_msg}. Please fix these issues in your JSON response."
            }
            last_err = ve
            
        except Exception as e:
            log.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            last_err = e
            # For unexpected errors, also add hints
            payload = {
                **payload,
                "_repair_hints": f"Previous attempt failed with error: {str(e)}. Please ensure your response is valid JSON."
            }

    raise LLMJsonError(
        f"JSON parsing failed after {max_retries} retries. "
        f"Provider: {provider}, Schema: {schema.__name__}, "
        f"Last error: {last_err}, All validation errors: {errs}"
    )