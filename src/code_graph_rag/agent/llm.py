from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_cypher_generate_model(
    *,
    temperature: float = 0.0,
    max_tokens: int | None = 512,
    seed: int | None = 42,
    json_mode: bool = False,
):
    """Return a configured ChatOpenAI client for intent/Cypher tasks.

    Deterministic defaults per Phase 3.1 DoD.
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    model_kwargs = {}
    if seed is not None:
        model_kwargs["seed"] = seed
    if json_mode:
        # Best-effort JSON mode for models that support it
        model_kwargs["response_format"] = {"type": "json_object"}

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs or None,
    )
    return llm