import pytest
from src.code_graph_rag.agent.models import QueryIntent, PlanExecutionResult, Action, SynthesisOutput
from src.code_graph_rag.agent.synthesis import synthesize

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

@pytest.mark.integration
def test_01_simple_synthesis_with_real_db(load_repo_into_memgraph):
    """
    Test SimpleSynthesizer với real database connection.
    
    Flow: Load repo → Create evidence → Run SimpleSynthesizer → Validate output
    """
    
    # 1) Load simple test repo into Memgraph
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
    
    # 2) Create query intent
    intent = QueryIntent(
        action=Action.list_callers,
        mention="proj.utils.helper",
        language="en"
    )
    
    # 3) Mock evidence results (as would come from plan execution)
    validated_results = [
        PlanExecutionResult(
            step="CALLERS_TOP",
            label="Function",
            id="proj.app.main",
            name="main",
            snippet="def main():\n    result = helper()\n    return result"
        ),
        PlanExecutionResult(
            step="META",
            label="Function",
            id="proj.utils.helper", 
            name="helper",
            snippet="def helper():\n    return \"done\""
        )
    ]
    
    # 4) Run SimpleSynthesizer
    output = synthesize(intent, validated_results, provider="openai")
    log.debug("test_01_simple_synthesis_with_real_db output: %s", output)

    # 5) Basic validation
    assert isinstance(output, SynthesisOutput)
    assert output.action == "list_callers"
    assert output.evidence_count == 2
    assert 0.0 <= output.confidence <= 1.0
    assert len(output.answer) > 0
    
    log.info(f"✅ Synthesizer test passed: confidence={output.confidence:.2f}, items={len(output.items)}")
    log.debug(f"Answer: {output.answer}")
    log.debug(f"Items: {output.items}")