from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

from src.code_graph_rag.utils.file_utils import walk_codebase
from src.code_graph_rag.parser.ast_parser import ASTParser
from src.code_graph_rag.graph.exporter import export_to_cypher

def parser():
    repo_path = Path("tests/sample_repo")
    structure = walk_codebase(repo_path)

    for node_type, nodes in structure.items():
        print(f"\n== {node_type.upper()} ==")
        for node in nodes:
            print(node)

def test_ast():
    project_path = Path("sample_repo")
    parser = ASTParser(project_root=project_path)
    nodes, edges= parser.parse()

    # for node in nodes:
        # print("NODE:", node)

    for edge in edges:
        print("EDGE:", edge)


def test_agent():
    from src.code_graph_rag.agent.intent import llm_parse_intent, decide_route
    from src.code_graph_rag.agent.models import QueryIntent

    question = "Ai gọi foo?"
    intent = llm_parse_intent(question)
    route = decide_route(intent)

    print(f"Parsed Intent: {intent}")
    print(f"Decided Route: {route}")


if __name__ == "__main__":
    
    # parser()
    # test_ast()
    test_agent()
