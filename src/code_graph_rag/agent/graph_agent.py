from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable

from langchain.prompts import ChatPromptTemplate

from pydantic import BaseModel

from src.code_graph_rag.agent.models import (
    QueryIntent, Route, 
    ResolvedEntity, 
    ExplainPlan, PlanStep,
    PlanExecutionResult, ValidationReport,
    SynthesisOutput
)
from src.code_graph_rag.agent.utils.utils import run_cypher_query
from src.code_graph_rag.agent.intent import llm_parse_intent, decide_route
from src.code_graph_rag.agent.resolver import resolve_entity
from src.code_graph_rag.agent.plan_maker import make_plan
from src.code_graph_rag.agent.plan_runner import run_plan
from src.code_graph_rag.agent.validator import validate_and_retry, make_retry_cb
from src.code_graph_rag.agent.synthesis import synthesize

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

class GraphState(BaseModel):
    repo_root: str | None = None  # Default repo root
    question: str | None = None
    intent: QueryIntent | None = None
    route: str | None = None
    resolve: ResolvedEntity | None = None

    plan: ExplainPlan | None = None
    plan_outputs: list[PlanExecutionResult] | None = None

    validated_rows: list[PlanExecutionResult] | None = None
    validation_report: ValidationReport | None = None

    synthesis_output: SynthesisOutput | None = None

    code_snippets: list[str] | None = None
    answer: str | None = None

def user_question_node(state: GraphState):
    return {"question": state.question}

def parse_intent_node(state: GraphState) -> GraphState:
    intent = llm_parse_intent(state.question or "", provider="nvidia")
    state.intent = intent
    return state

def router_node(state: GraphState) -> GraphState:
    route = decide_route(state.intent) if state.intent else Route.FAST
    state.route = route.value
    return state

def resolve_entity_node(state: GraphState) -> GraphState:
    if state.intent:
        state.resolve = resolve_entity(state.intent)
    return state

def make_plan_node(state: GraphState) -> GraphState:
    if not (state.intent and state.resolve):
        return state
    state.plan = make_plan(state.intent, state.resolve, provider="openai")
    return state

def run_plan_node(state: GraphState) -> GraphState:
    if state.plan and state.intent and state.resolve:
        state.plan_outputs = run_plan(plan=state.plan, intent=state.intent, resolved=state.resolve, repo_root=state.repo_root)
    return state

def validate_and_retry_node(state: GraphState) -> GraphState:
    if not (state.plan_outputs and state.intent and state.resolve):
        return state
    outs = state.plan_outputs or []

    # 1) Tập hợp required steps từ ExplainPlan
    required_steps = {s.name for s in state.plan.steps if s.required}

    # 2) Tạo runner wrapper cho retry_cb
    def _runner(steps: list[PlanStep], intent: QueryIntent, resolved: ResolvedEntity, repo_root: str | None):
        # NOTE: dùng run_plan cho 1 hoặc nhiều step bằng cách tạo ExplainPlan tạm
        tmp = ExplainPlan(steps=steps, knobs=state.plan.knobs if state.plan else {})
        return run_plan(plan=tmp, intent=intent, resolved=resolved, repo_root=repo_root or "")

    retry_cb = make_retry_cb(
        plan_steps=state.plan.steps,
        runner=_runner,
        intent=state.intent,
        resolved=state.resolve,
        repo_root=state.repo_root,
    )

    # 3) Validate + Retry
    cleaned, report = validate_and_retry(
        rows=outs,
        required_steps=required_steps,
        retry_cb=retry_cb,
        max_retries=2,  # Tối đa 2 lần retry
    )

    state.validated_rows = cleaned
    state.validation_report = report
    return state

def synthesis_node(state: GraphState) -> GraphState:
    if not (state.validated_rows and state.intent):
        log.warning("simple_synthesis_node: Missing validated_rows or intent")
        return state
    
    try:

        output = synthesize(
            intent=state.intent,
            results=state.validated_rows,
            provider="nvidia" 
        )
        
        state.synthesis_output = output

        log.info(f"simple_synthesis_node: Success - evidence_count={output.evidence_count}, "
                 f"confidence={output.confidence:.2f}")

        return state
        
    except Exception as e:
        log.error(f"simple_synthesis_node: Failed - {e}")
        
        # Create minimal fallback
        fallback_output = SynthesisOutput(
            action=state.intent.action.value if state.intent else "unknown",
            answer=f"Synthesis failed: {str(e)}",
            items=[],
            evidence_count=len(state.validated_rows) if state.validated_rows else 0,
            confidence=0.1
        )
        
        state.simple_synthesis_output = fallback_output
        return state

def context_retrieval_node(state: GraphState):
    if not state.plan:
        return {"matched_node": [], "code_snippets": []}
    
    results = run_cypher_query(state.cypher_query)

    # Lấy nội dung code nếu có field "source" hoặc "code"
    snippets = []
    for r in results:
        for v in r.values():
            if isinstance(v, dict):
                code = v.get("source") or v.get("code")
                if code:
                    snippets.append(code)
    return {
        "matched_nodes": results,
        "code_snippets": snippets
    }

def answer_generation_node(state: GraphState):
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers developer questions about a codebase using the given context."),
        ("user", "Question: {question}\nContext:\n{context}\n\nAnswer:"),
    ])
    
    model = get_cypher_generate_model()
    chain = answer_prompt | model

    context = "\n---\n".join(state.code_snippets or [])

    result = chain.invoke({
        "question": state.question,
        "context": context
    })

    return {"answer": result.content.strip()}

builder = StateGraph(GraphState)

builder.add_node("UserQuestion", user_question_node)
builder.add_node("ParseIntent", parse_intent_node)
builder.add_node("Router", router_node)
builder.add_node("ResolveEntity", resolve_entity_node)
builder.add_node("MakePlan", make_plan_node)
builder.add_node("RunPlan", run_plan_node)
builder.add_node("ValidateAndRetry", validate_and_retry_node)
builder.add_node("Synthesis", synthesis_node)

builder.set_entry_point("UserQuestion")
builder.add_edge("UserQuestion", "ParseIntent")
builder.add_edge("ParseIntent", "Router")
builder.add_edge("Router", "ResolveEntity")
builder.add_edge("ResolveEntity", "MakePlan")
builder.add_edge("MakePlan", "RunPlan")
builder.add_edge("RunPlan", "ValidateAndRetry")
builder.add_edge("ValidateAndRetry", "Synthesis")
builder.set_finish_point("Synthesis")

graph = builder.compile()

# final_output = graph.invoke({"question": "What is the relationship between (c:Class {name: 'Trainer'}) and (m:Method {name: 'train'})?"})
# print(final_output)
