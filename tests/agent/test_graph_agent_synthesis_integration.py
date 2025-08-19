import pytest
from src.code_graph_rag.agent.graph_agent import graph, GraphState
from src.code_graph_rag.agent.models import QueryIntent, Action, Language

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

@pytest.mark.integration
def test_graphagent_with_synthesis_pipeline(load_repo_into_memgraph):
    """
    Test complete GraphAgent pipeline với synthesis integration.
    """
    # Setup test repo
    repo_dir = load_repo_into_memgraph(
        {
            "app.py": """
def main():
    result = helper()
    return result
""",
            "utils.py": """
def helper():
    return "done"
"""
        },
        project_name="proj"
    )
    
    # Test full pipeline
    initial_state = GraphState(
        repo_root=str(repo_dir),
        question="Who calls the helper?"
    )
    
    # Run complete graph
    final_state = graph.invoke(initial_state)
    log.debug("test_graphagent_with_synthesis_pipeline: Final state: %s", final_state)
    
    # Validate synthesis output exists
    assert final_state["synthesis_output"] is not None
    assert final_state["synthesis_output"].action == Action.list_callers
    assert final_state["synthesis_output"].confidence > 0
    assert len(final_state["synthesis_output"].items) >= 0


    log.info(f"✅ Full pipeline synthesis: {final_state['synthesis_output'].confidence:.2%} confidence")
