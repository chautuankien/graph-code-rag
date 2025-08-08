from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

from src.code_graph_rag.utils.file_utils import walk_codebase
from src.code_graph_rag.parser.ast_parser import ASTParser
from src.code_graph_rag.graph.graph_builder import CodeGraphBuilder
from src.code_graph_rag.graph.exporter import export_to_cypher

def parser():
    repo_path = Path("tests/sample_repo")
    structure = walk_codebase(repo_path)

    for node_type, nodes in structure.items():
        print(f"\n== {node_type.upper()} ==")
        for node in nodes:
            print(node)

def test_ast():
    project_path = Path("tests/calls_edge_test")
    parser = ASTParser(project_root=project_path)
    nodes, edges= parser.parse()

    # for node in nodes:
        # print("NODE:", node)

    for edge in edges:
        print("EDGE:", edge)

def visualize_graph():
    repo_path = Path("tests/sample_repo")
    builder = CodeGraphBuilder(repo_path=repo_path)
    graph = builder.build()

    pos = nx.spring_layout(graph, seed=42)

    node_colors = []
    for _, data in graph.nodes(data=True):
        node_type = data.get("type", "")
        if node_type == "Function":
            node_colors.append("lightblue")
        elif node_type == "Class":
            node_colors.append("orange")
        elif node_type == "Method":
            node_colors.append("violet")
        elif node_type == "Module":
            node_colors.append("green")
        elif node_type == "ExternalPackage":
            node_colors.append("gray")
        else:
            node_colors.append("lightgray")
    
    edge_labels = nx.get_edge_attributes(graph, "type")

    nx.draw(graph, pos, with_labels=True, node_size=800, node_color=node_colors, font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)

    plt.title("Code Knowledge Graph")
    plt.tight_layout()
    plt.show()

def export():
    repo_path = Path("tests/sample_repo")
    builder = CodeGraphBuilder(repo_path)
    graph = builder.build()

    cypher_path = Path("graph_export.cypherl")
    export_to_cypher(graph, cypher_path)


if __name__ == "__main__":
    
    # parser()
    test_ast()
    # visualize_graph()
    # export()
