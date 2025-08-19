import pytest
from pathlib import Path

from src.code_graph_rag.agent.models import (
    QueryIntent, Action, Language, PlanExecutionResult
)
from code_graph_rag.agent.synthesis.synthesis_engine import SynthesisEngine, create_synthesis_engine
from src.code_graph_rag.agent.synthesis.synthesis_context import ContextPreparationConfig
from src.code_graph_rag.utils.logging_setup import get_logger

log = get_logger(__name__)

@pytest.mark.integration
def test_01_synthesis_list_callers_basic(load_repo_into_memgraph):
    """
    Basic integration test: synthesis engine với real evidence.
    
    Tests: load simple repo → create evidence → run synthesis → validate output
    """
    
    # 1) Setup: Simple repo với clear caller relationship
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

    # 2) Create intent
    intent = QueryIntent(
        action=Action.list_callers,
        mention="proj.utils.helper",
        language=Language.en
    )

    # 3) Mock evidence (normally từ plan execution)
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

    # 4) Run synthesis
    engine = create_synthesis_engine(provider="nvidia")
    synthesis_output = engine.synthesize(intent, validated_results)
    log.debug("test_synthesis_list_callers_basic: Synthesis output: %s", synthesis_output)

    # 5) Basic validation
    assert synthesis_output.action == Action.list_callers
    assert synthesis_output.language == Language.en
    assert len(synthesis_output.items) >= 1
    assert synthesis_output.confidence > 0.0
    
    log.info(f"✅ Synthesis test passed: {len(synthesis_output.items)} items, confidence={synthesis_output.confidence:.2f}")

def test_02_synthesis_list_callers_real_evidence(load_repo_into_memgraph):
    """
    Test synthesis cho list_callers action với real evidence từ database.
    
    Scenario:
    - Load simple repo với function calls
    - Run full plan execution pipeline
    - Test synthesis engine với real evidence
    - Validate output structure và content quality
    """
    # 1) Setup: Create repo với clear caller-callee relationships
    repo_dir = load_repo_into_memgraph(
        {
            "app.py": """
def main():
'''Main entry point for the application'''
config = load_config()
result = process_data(config)
return result

def load_config():
'''Load application configuration'''
return {"debug": True, "port": 8000}
""",
            "utils.py": """
def process_data(config):
'''Process data based on configuration'''
if config.get("debug"):
    log_debug("Processing in debug mode")
return helper_function(config)

def helper_function(config):
'''Helper function for data processing'''
return {"processed": True, "config": config}

def log_debug(message):
'''Log debug message'''
print(f"DEBUG: {message}")
"""
        },
        project_name="proj"
    )

    # 2) Setup: Create intent để find callers của process_data
    intent = QueryIntent(
        action=Action.list_callers,
        mention="proj.utils.process_data",
        language=Language.en,
        limit=10
    )

    # 3) Create mock evidence (normally từ plan execution)
    # Simulate evidence từ CALLERS_TOP và META adapters
    validated_results = [
        # Evidence 1: main() calls process_data()
        PlanExecutionResult(
            step="CALLERS_TOP",
            label="Function",
            id="proj.app.main",
            name="main",
            path="app.py",
            start_line=2,
            end_line=6,
            docstring="Main entry point for the application",
            signature="main() -> None",
            snippet="""def main():
'''Main entry point for the application'''
config = load_config()
result = process_data(config)
return result"""
        ),
        
        # Evidence 2: Target function metadata
        PlanExecutionResult(
            step="META",
            label="Function", 
            id="proj.utils.process_data",
            name="process_data",
            path="utils.py",
            start_line=1,
            end_line=5,
            docstring="Process data based on configuration",
            signature="process_data(config) -> dict",
            snippet="""def process_data(config):
'''Process data based on configuration'''
if config.get("debug"):
    log_debug("Processing in debug mode")
return helper_function(config)"""
        )
    ]

    # 4) Run synthesis
    engine = create_synthesis_engine(provider="nvidia", temperature=0.0)
    
    log.info(f"test_synthesis_list_callers: Running synthesis với {len(validated_results)} evidence items")
    
    try:
        synthesis_output = engine.synthesize(
            intent=intent,
            validated_results=validated_results
        )
        log.debug("test_02_synthesis_list_callers_real_evidence: Synthesis output: %s", synthesis_output)
        
        # 5) Validate synthesis output structure
        assert synthesis_output.action == Action.list_callers
        assert synthesis_output.language == Language.en
        assert synthesis_output.answer  # Non-empty answer
        assert isinstance(synthesis_output.items, list)
        assert isinstance(synthesis_output.citations, list)
        assert 0.0 <= synthesis_output.confidence <= 1.0
        assert synthesis_output.evidence_count > 0
        
        # 6) Validate content quality
        log.info(f"Synthesis answer: {synthesis_output.answer}")
        log.info(f"Items count: {len(synthesis_output.items)}")
        log.info(f"Confidence: {synthesis_output.confidence:.2f}")
        
        # Should find main() as caller
        assert len(synthesis_output.items) >= 1, "Should find at least one caller"
        
        # Check first item structure (should be CallerCalleeItem)
        first_item = synthesis_output.items[0]
        assert "caller" in first_item, "Item should have caller field"
        assert "callee" in first_item, "Item should have callee field"
        assert "why" in first_item, "Item should have explanation"
        
        # Validate caller information
        caller = first_item["caller"]
        assert caller["id"] == "proj.app.main"
        assert caller["label"] == "Function"
        assert caller["name"] == "main"
        
        # Validate callee information
        callee = first_item["callee"]
        assert callee["id"] == "proj.utils.process_data"
        assert callee["label"] == "Function"
        
        # 7) Validate citations reference real evidence
        citation_ids = {citation.id for citation in synthesis_output.citations}
        evidence_ids = {result.id for result in validated_results}
        
        for citation_id in citation_ids:
            assert citation_id in evidence_ids, f"Citation {citation_id} not found in evidence"
        
        log.info("✅ test_synthesis_list_callers: SUCCESS")
        
    except Exception as e:
        log.error(f"❌ test_synthesis_list_callers: FAILED - {e}")
        raise

def test_03_synthesis_error_handling_insufficient_evidence(load_repo_into_memgraph):
        """
        Test synthesis error handling khi có insufficient evidence.
        
        Validates:
        - Graceful handling của empty/minimal evidence
        - Fallback output generation
        - Confidence adjustment
        """
        # Setup minimal repo
        repo_dir = load_repo_into_memgraph(
            {"empty.py": "# Empty file"},
            project_name="emptyproj"
        )

        intent = QueryIntent(
            action=Action.list_callers,
            mention="nonexistent.function", 
            language=Language.en
        )

        # Minimal/bad evidence
        validated_results = [
            PlanExecutionResult(
                step="NEIGHBORHOOD",
                label="Module",
                id="emptyproj.empty",
                name="empty.py",
                path="empty.py"
                # No snippet, minimal info
            )
        ]

        engine = create_synthesis_engine()
        synthesis_output = engine.synthesize(intent, validated_results)

        # Should still return valid output
        assert synthesis_output.action == Action.list_callers
        assert synthesis_output.language == Language.en
        assert synthesis_output.confidence < 0.5, "Confidence should be low with poor evidence"
        assert len(synthesis_output.items) == 0, "Should have no items với poor evidence"
        assert synthesis_output.notes, "Should have notes explaining limitations"
        
        log.info(f"✅ Low confidence output: {synthesis_output.confidence:.2f}")