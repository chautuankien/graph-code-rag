"""Tests for parsing intent and deciding routes for the agent router."""

import json, logging
import pytest
from src.code_graph_rag.agent.models import Action, Route, QueryIntent
from src.code_graph_rag.agent import intent as intent_mod
from src.code_graph_rag.utils.logging_setup import get_logger


@pytest.mark.parametrize(
    "q,payload,expect_action,expect_route,expect_depth,expect_limit,expect_k",
    [
        (
            "Ai gọi foo?",
            {"action":"list_callers","mention":"foo","depth":1,"limit":20,"k_paths":3,"language":"vi"},
            Action.list_callers, Route.FAST, 1, 20, 3
        ),
        (
            "Trace từ preprocess sang train",
            {"action":"trace_flow","mention":"preprocess","mention_dst":"train","depth":4,"limit":100,"k_paths":2,"language":"vi"},
            Action.trace_flow, Route.PLAN, 4, 100, 2
        ),
        (
            "Explain load_data",
            {"action":"explain_function","mention":"load_data","depth":99,"limit":999,"k_paths":0,"language":"en"},
            Action.explain_function, Route.PLAN, 5, 200, 1  # clamp về biên an toàn
        ),
    ]
)
def test_parse_and_route_happy(
    patch_chain,
    monkeypatch,
    q,
    payload,
    expect_action,
    expect_route,
    expect_depth,
    expect_limit,
    expect_k,
):
    """Ensure intent is parsed and routed as expected with stubbed pipeline."""
    # Arrange: the chain returns the provided payload as JSON.
    chain = patch_chain(payloads=[json.dumps(payload)])

    # Patch the intent pipeline to use the fake chain for both attempts.
    monkeypatch.setattr(
        intent_mod,
        "ChatPromptTemplate",
        type("X", (object,), {"from_messages": staticmethod(lambda m: chain)}),
    )

    # Language heuristic fallback matches payload or default.
    monkeypatch.setattr(
        intent_mod, "_detect_language", lambda *_, **__: payload.get("language", "vi")
    )

    # Act
    qi = intent_mod.llm_parse_intent(q)
    route = intent_mod.decide_route(qi)

    # Assert
    assert isinstance(qi, QueryIntent)
    assert qi.action is expect_action
    assert qi.depth == expect_depth
    assert qi.limit == expect_limit
    assert qi.k_paths == expect_k
    assert route is expect_route

def test_auto_repair_once(patch_chain, monkeypatch):
    bad = "{ action: list_callers, mention: 'foo' }"  # thiếu quote keys → lỗi json
    good = json.dumps({"action":"list_callers","mention":"foo","depth":1,"limit":10,"k_paths":3,"language":"vi"})
    chain = patch_chain(payloads=[bad, good])  # lần 1 lỗi, lần 2 đúng

    monkeypatch.setattr(intent_mod, "ChatPromptTemplate",
                        type("X",(object,),{"from_messages": staticmethod(lambda m: chain)}))
    monkeypatch.setattr(intent_mod, "_detect_language", lambda *_: "vi")

    qi = intent_mod.llm_parse_intent("Ai gọi foo?")
    calls = getattr(chain, "_chain", getattr(chain, "calls", []))
    if hasattr(calls, "calls"):  # chain là _Builder, còn ._chain mới là FakeChain
        calls = calls.calls

    assert qi.action is Action.list_callers, f"Expected action 'list_callers', got {qi.action}"
    assert qi.mention == "foo", f"Expected mention 'foo', got {qi.mention}"
    assert len(calls) == 2, f"expect 2 invokes, got {len(calls)}; calls={calls}"

@pytest.mark.parametrize("q,lang", [
    ("Ai gọi hàm main?", "vi"),
    ("Who calls main?", "en"),
])
def test_language_detection_and_defaults(patch_chain, monkeypatch, q, lang):
    # Payload thiếu language/depth/limit/k_paths → phải tự set default + detect lang
    minimal = {"action":"list_callees","mention":"main"}
    chain = patch_chain(payloads=[json.dumps(minimal)])

    monkeypatch.setattr(intent_mod, "ChatPromptTemplate",
                        type("X",(object,),{"from_messages": staticmethod(lambda m: chain)}))
    monkeypatch.setattr(intent_mod, "_detect_language", lambda text: lang)

    qi = intent_mod.llm_parse_intent(q)
    assert qi.language == lang
    assert qi.depth == 2 and qi.limit == 50 and qi.k_paths == 3