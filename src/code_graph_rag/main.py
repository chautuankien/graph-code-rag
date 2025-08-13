from pathlib import Path
import uuid

from src.code_graph_rag.utils.logging_setup import setup_logging, pipeline_context, set_corr_id, get_logger
from src.code_graph_rag.pipeline.build_knowledge_graph import build_knowledge_graph_and_insert_db
from src.code_graph_rag.agent.graph_agent import graph, GraphState

Path("logs").mkdir(parents=True, exist_ok=True)
setup_logging(level="INFO", log_file="logs/app.log", force=True)

def build_knowledge_graph() -> None:
    repo_path = Path("sample_repo")
    export_path = "graph_export.cypher"
    with pipeline_context("build-kg") as ctx:
        log = get_logger(__name__)
        log.info("Start build graph")

        build_knowledge_graph_and_insert_db(
            repo_path=repo_path,
            export_path=export_path,
            bootstrap_schema=False,
            bootstrap_file="db_bootstrap.cypher",
        )

        log.info("Graph built successfully")

def run_agent(question: str) -> None:
    # Nếu có request id truyền từ caller, set_corr_id(req_id). Nếu không thì tạo.
    set_corr_id(str(uuid.uuid4())[:8])

    with pipeline_context("agent") as ctx:
        log = get_logger(__name__)
        log.info(f"Agent received question: {question}")

        state = GraphState(question=question)
        response = graph.invoke(state)
        response = GraphState.model_validate(response)
        log.info(f"Agent response: {response}")

if __name__ == "__main__":
    build_knowledge_graph()
    # run_agent("Who calls foo?")
