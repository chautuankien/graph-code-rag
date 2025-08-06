from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable

from langchain.prompts import ChatPromptTemplate

from pydantic import BaseModel

from src.code_graph_rag.agent.llm import get_cypher_generate_model
from src.code_graph_rag.agent.utils.utils import run_cypher_query

class GraphState(BaseModel):
    question: str | None = None
    cypher_query: str | None = None
    matched_node: list[dict] | None = None
    code_snippets: list[str] | None = None
    answer: str | None = None

def user_question_node(state: GraphState):
    return {"question": state.question}

def graph_query_node(state: GraphState):
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a codebase assistant. Your job is to translate user questions into Cypher queries for a code knowledge graph."),
        ("user", "Question: {question}\nReturn only the Cypher query")
    ])

    model = get_cypher_generate_model()
    chain = query_prompt | model
    result = chain.invoke({"question": state.question})

    return {"cypher_query": result.content.strip()}

def context_trieval_node(state: GraphState):
    if not state.cypher_query:
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
builder.add_node("GraphQuery", graph_query_node)
builder.add_node("ContextRetrieval", context_trieval_node)
builder.add_node("AnswerGeneration", answer_generation_node)

builder.set_entry_point("UserQuestion")
builder.add_edge("UserQuestion", "GraphQuery")
builder.add_edge("GraphQuery", "ContextRetrieval" )
builder.add_edge("ContextRetrieval", "AnswerGeneration")
builder.set_finish_point("AnswerGeneration")

graph = builder.compile()

# final_output = graph.invoke({"question": "What is the relationship between (c:Class {name: 'Trainer'}) and (m:Method {name: 'train'})?"})
# print(final_output)
