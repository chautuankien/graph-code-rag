# src/code_graph_rag/agent/adapters/registry.py
from __future__ import annotations
from typing import Callable, Protocol, Any

from src.code_graph_rag.agent.models import QueryIntent, ResolvedEntity, PlanStep
from src.code_graph_rag.utils.logging_setup import get_logger

log = get_logger(__name__)

class Adapter(Protocol):
    """Adapter protocol: build & execute a parameterized Cypher."""
    def __call__(
        self,
        *,
        step: PlanStep,
        intent: QueryIntent,
        resolved: ResolvedEntity,
    ) -> list[dict[str, Any]]: ...

_REGISTRY: dict[str, Adapter] = {}

def register(name: str):
    """Decorator to register an adapter by step name (allow-list members)."""
    def _wrap(fn: Adapter) -> Adapter:
        if name in _REGISTRY:
            raise ValueError(f"Adapter already registered: {name}")
        _REGISTRY[name] = fn
        return fn
    return _wrap

def get(name: str) -> Adapter:
    fn = _REGISTRY.get(name)
    if not fn:
        raise KeyError(f"No adapter for step: {name}")
    return fn

def available() -> list[str]:
    return sorted(_REGISTRY)
