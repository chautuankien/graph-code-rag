print("[DEBUG] exporter.py loaded")
from networkx import DiGraph
from pathlib import Path


def escape_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def export_to_cypher(graph: DiGraph, output_path: Path):
    """
    Export a NetworkX DiGraph to a .cypher file that can be loaded into Neo4j or Memgraph.
    """
    lines = []

    # Export nodes
    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("type", "Node")
        props = ", ".join(
            f"{k}: '{escape_string(str(v))}'"
            for k, v in attrs.items()
            if v is not None
        )
        lines.append(f"MERGE (:`{label}` {{{props}}});")

    # Export edges
    for src, dst, attrs in graph.edges(data=True):
        rel_type = attrs.get("type", "RELATES_TO")
        lines.append(
            f"MATCH (a {{id: '{escape_string(src)}'}}), (b {{id: '{escape_string(dst)}'}}) "
            f"MERGE (a)-[:`{rel_type}`]->(b);"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] Exported Cypher script to: {output_path}")
