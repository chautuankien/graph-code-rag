import pytest
from src.code_graph_rag.agent.graph_agent import graph, GraphState

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

@pytest.mark.integration
def test_simple_synthesis_end_to_end(load_repo_into_memgraph):
    """
    Test complete SimpleSynthesizer pipeline end-to-end với real database.
    
    Flow: Question → GraphAgent → SimpleSynthesizer → Response
    """
    
    # 1) Load test repo into Memgraph
    repo_dir = load_repo_into_memgraph(
        {
            "main.py": """
def main():
    result = helper()
    return result

def process():
    helper()
""",
            "utils.py": """
def helper():
    return "done"
"""
        },
        project_name="proj"
    )
    
    # 2) Test simple question → response
    question = "Who calls the helper()?"
    
    state = GraphState(
        question=question,
        repo_root=str(repo_dir)
    )
    
    response = graph.invoke(state)
    response = GraphState.model_validate(response)
    log.debug("test_simple_synthesis_end_to_end: response=%s", response)

    # 3) Validate response
    assert response.synthesis_output is not None
    
    output = response.synthesis_output
    assert output.action == "list_callers"
    assert output.evidence_count > 0
    assert 0.0 <= output.confidence <= 1.0
    assert len(output.answer) > 0
    
    log.info(f"✅ End-to-end test passed: confidence={output.confidence:.2f}, items={len(output.items)}")
    log.debug(f"Answer: {output.answer}")
    log.debug(f"Items: {output.items}")