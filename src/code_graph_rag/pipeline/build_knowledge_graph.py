from pathlib import Path
import mgclient

from src.code_graph_rag.graph.graph_builder import CodeGraphBuilder
from src.code_graph_rag.graph.exporter import export_to_cypher

def build_knowledge_graph_and_insert_db(
        repo_path: str, export_path="graph_export.cypherl"):
    
    # 1. Build graph
    builder = CodeGraphBuilder(Path(repo_path))
    graph = builder.build()

    # 2. Export to CypherL
    export_to_cypher(graph, Path(export_path))

    # 3. Import into Memgraph
    conn = mgclient.connect(host="localhost", port=7687)
    cursor = conn.cursor()

    cursor.execute("MATCH (n) DETACH DELETE n")  # optional: clear DB

    with open(export_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("//"):  # Skip empty or comment
                continue

        try:
            cursor.execute(line)
        except Exception as e:
            print(f"❌ Failed to execute line:\n{line}\n→ {e}")

    print(f"✅ Loaded graph into Memgraph from {repo_path}")