"""Export parsed nodes and edges to Memgraph-compatible Cypher.

The exporter merges nodes by their natural keys and writes relationships with
minimal properties. Edges that cannot be resolved to internal nodes (for
example, CALLS to builtins or unresolved externals) are intentionally skipped.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable

from src.code_graph_rag.models.nodes import (
    ProjectNode, PackageNode, FolderNode, FileNode,
    ModuleNode, ClassNode, FunctionNode, MethodNode, ExternalPackageNode, BaseNode
)
from src.code_graph_rag.models.edges import (
    BaseEdge, ContainsEdge, DefinesEdge, InheritsEdge, OverridesEdge,
    CallsEdge, DependsOnExternalEdge, ImportsEdge, NodeType
)


def _escape(s: str) -> str:
    """Escape backslashes and single quotes for Cypher string literals.

    Args:
        s: Input string to be escaped.

    Returns:
        The escaped string safe to embed in Cypher.
    """
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _label_key_name_val(node: BaseNode) -> tuple[str, str, str]:
    """Map a node to its natural key triple.

    The natural key is the (label, key property name, key value) used to
    uniquely identify the node in the graph.

    Args:
        node: A Pydantic node model instance.

    Returns:
        A tuple of (label, key property name, key value).
    """
    if isinstance(node, ProjectNode):
        return "Project", "name", node.name
    if isinstance(node, PackageNode):
        return "Package", "qualified_name", node.qualified_name
    if isinstance(node, ModuleNode):
        return "Module", "qualified_name", node.qualified_name
    if isinstance(node, ClassNode):
        return "Class", "qualified_name", node.qualified_name
    if isinstance(node, MethodNode):
        return "Method", "qualified_name", node.qualified_name
    if isinstance(node, FunctionNode):
        return "Function", "qualified_name", node.qualified_name
    if isinstance(node, FolderNode):
        return "Folder", "path", node.path
    if isinstance(node, FileNode):
        return "File", "path", node.path
    if isinstance(node, ExternalPackageNode):
        return "ExternalPackage", "name", node.name
    raise TypeError(f"Unsupported node type: {type(node)}")

def _cy_props(props: dict[str, Any]) -> str:
    """Render a dict of properties as a Cypher map fragment.

    Args:
        props: Dictionary of properties to render.

    Returns:
        A string like "key1: 'val', key2: 3" suitable for Cypher maps.
    """
    parts: list[str] = []
    for k, v in props.items():
        if isinstance(v, bool):
            parts.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
        elif isinstance(v, list):
            arr = ", ".join(f"'{_escape(str(x))}'" for x in v)
            parts.append(f"{k}: [{arr}]")
        elif v is None:
            continue
        else:
            parts.append(f"{k}: '{_escape(str(v))}'")
    return ", ".join(parts)


def _node_props(node: BaseNode) -> dict[str, Any]:
    """Return all node properties excluding transient identifiers.

    Drops values that are None to keep emitted Cypher concise.

    Args:
        node: A Pydantic node model instance.

    Returns:
        A dictionary of properties to serialize.
    """
    data = node.model_dump()
    # Drop None values to keep Cypher concise.
    return {k: v for k, v in data.items() if v is not None}


def _edge_props(edge: BaseEdge) -> dict[str, Any]:
    """Return edge props excluding source/target/type and stringify enums.

    Converts CallsEdge enum fields to strings and drops None values.

    Args:
        edge: A Pydantic edge model instance.

    Returns:
        A dictionary of edge properties to serialize.
    """
    d = edge.model_dump()
    d.pop("source", None)
    d.pop("target", None)
    d.pop("type", None)
    # CallsEdge carries enum fields that must be stringified.
    if isinstance(edge, CallsEdge):
        if d.get("caller_type") is not None:
            d["caller_type"] = edge.caller_type.value  # type: ignore[attr-defined]
        if d.get("callee_type") is not None:
            d["callee_type"] = edge.callee_type.value  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if v is not None}

def _merge_edge_stmt(
    src: tuple[str, str, str],
    dst: tuple[str, str, str],
    rel: str,
    props: dict[str, Any],
) -> str:
    """Build a MERGE statement for a relationship between two nodes.

    Args:
        src: (label, key name, key value) triple for the source node.
        dst: (label, key name, key value) triple for the target node.
        rel: Relationship type name.
        props: Properties to set on the relationship.
    If empty, no properties are set.    
    Returns:
        A Cypher MATCH + MERGE statement string.
    """
    src_lbl, src_key, src_val = src
    dst_lbl, dst_key, dst_val = dst
    prop_str = _cy_props(props)
    prop_clause = f" {{ {prop_str} }}" if prop_str else ""
    return (
        f"MATCH (a:`{src_lbl}` {{ {src_key}: '{_escape(src_val)}' }}), "
        f"(b:`{dst_lbl}` {{ {dst_key}: '{_escape(dst_val)}' }}) "
        f"MERGE (a)-[:`{rel}`{prop_clause}]->(b);"
    )


def _merge_node_stmt(
    label: str, key_name: str, key_val: str, props: dict[str, Any]
) -> str:
    """Build a MERGE statement for a node using its natural key.

    Args:
        label: Node label.
        key_name: Key property name.
        key_val: Key property value.
        props: All node properties.

    Returns:
        A Cypher MERGE statement, with SET for non-key properties when present.
    """
    props_wo_key = {k: v for k, v in props.items() if k != key_name and v is not None}
    if props_wo_key:
        return (
            f"MERGE (n:`{label}` {{ {key_name}: '{_escape(key_val)}' }}) "
            f"SET n += {{ {_cy_props(props_wo_key)} }};"
        )
    return f"MERGE (:`{label}` {{ {key_name}: '{_escape(key_val)}' }});"


def export_to_cypher(
    nodes: Iterable[BaseNode],
    edges: Iterable[BaseEdge],
    output_path: Path,
) -> None:
    """Export nodes and edges to a Cypher file using natural-key MERGE.

    Args:
        nodes: Iterable of node models.
        edges: Iterable of edge models.
        output_path: Target .cypher file path.

    Algorithm:
      1. Emit MERGE for nodes using (label, key, value) and SET the rest.
      2. Build a registry: natural identifier -> (label, key, value).
      3. For each edge:
         - Skip if source/target do not resolve to internal nodes.
         - IMPORTS: keep only when target is an internal module.
         - CALLS: skip when target is None or builtin/external without nodes.
         - INHERITS: skip when base is external (no node).
         - Otherwise, MATCH source/target via registry and MERGE the edge.
    """
    lines: list[str] = []

    # 1) Nodes.
    #    - Build a registry to resolve edge identifiers.
    registry: dict[str, tuple[str, str, str]] = {}
    for node in nodes:
        label, key_name, key_val = _label_key_name_val(node)
        registry[key_val] = (label, key_name, key_val)
        lines.append(_merge_node_stmt(label, key_name, key_val, _node_props(node)))

    # 2) Edges.
    for edge in edges:
        rel_type = edge.type
        src_id = edge.source
        dst_id = edge.target

        # --- Filter cases that cannot be matched. ---
        if isinstance(edge, CallsEdge):
            # Target may be None/builtin/external → no node to attach.
            if dst_id is None or dst_id not in registry:
                continue

        if isinstance(edge, ImportsEdge):
            # Keep only when target is an internal module (has a node).
            if dst_id not in registry:
                continue

        if isinstance(edge, InheritsEdge):
            # External bases (raw) have no node.
            if dst_id not in registry:
                continue

        # Both source and target must be internal nodes.
        if src_id not in registry or dst_id not in registry:
            continue

        src_info = registry[src_id]
        dst_info = registry[dst_id]
        stmt = _merge_edge_stmt(src_info, dst_info, rel_type, _edge_props(edge))
        lines.append(stmt)

    # 3) Write to disk.
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")