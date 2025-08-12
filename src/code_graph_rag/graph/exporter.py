# src/code_graph_rag/graph/export_phase2.py
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
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _label_key_name_val(n: BaseNode) -> tuple[str, str, str]:
    """Map node -> (Label, key_prop_name, key_value) bằng khóa tự nhiên."""
    if isinstance(n, ProjectNode):
        return "Project", "name", n.name
    if isinstance(n, PackageNode):
        return "Package", "qualified_name", n.qualified_name
    if isinstance(n, ModuleNode):
        return "Module", "qualified_name", n.qualified_name
    if isinstance(n, ClassNode):
        return "Class", "qualified_name", n.qualified_name
    if isinstance(n, MethodNode):
        return "Method", "qualified_name", n.qualified_name
    if isinstance(n, FunctionNode):
        return "Function", "qualified_name", n.qualified_name
    if isinstance(n, FolderNode):
        return "Folder", "path", n.path
    if isinstance(n, FileNode):
        return "File", "path", n.path
    if isinstance(n, ExternalPackageNode):
        return "ExternalPackage", "name", n.name
    raise TypeError(f"Unsupported node type: {type(n)}")

def _node_props(n: BaseNode) -> dict[str, Any]:
    """Tất cả props (không thêm id)."""
    data = n.model_dump()
    # loại None để Cypher gọn hơn
    return {k: v for k, v in data.items() if v is not None}

def _cy_props(props: dict[str, Any]) -> str:
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

def _edge_props(e: BaseEdge) -> dict[str, Any]:
    """Edge props, bỏ source/target/type; convert enum -> str nếu cần."""
    d = e.model_dump()
    d.pop("source", None)
    d.pop("target", None)
    d.pop("type", None)
    # CallsEdge có enum
    if isinstance(e, CallsEdge):
        if d.get("caller_type") is not None:
            d["caller_type"] = e.caller_type.value  # type: ignore[attr-defined]
        if d.get("callee_type") is not None:
            d["callee_type"] = e.callee_type.value  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if v is not None}

def _merge_node_stmt(label: str, key_name: str, key_val: str, props: dict[str, Any]) -> str:
    """MERGE theo khóa tự nhiên; SET phần còn lại (không chạm key)."""
    props_wo_key = {k: v for k, v in props.items() if k != key_name and v is not None}
    if props_wo_key:
        return (
            f"MERGE (n:`{label}` {{ {key_name}: '{_escape(key_val)}' }}) "
            f"SET n += {{ {_cy_props(props_wo_key)} }};"
        )
    return f"MERGE (:`{label}` {{ {key_name}: '{_escape(key_val)}' }});"


def _merge_edge_stmt(
    src: tuple[str, str, str],
    dst: tuple[str, str, str],
    rel: str,
    props: dict[str, Any],
) -> str:
    src_lbl, src_key, src_val = src
    dst_lbl, dst_key, dst_val = dst
    prop_str = _cy_props(props)
    prop_clause = f" {{ {prop_str} }}" if prop_str else ""
    return (
        f"MATCH (a:`{src_lbl}` {{ {src_key}: '{_escape(src_val)}' }}), "
        f"(b:`{dst_lbl}` {{ {dst_key}: '{_escape(dst_val)}' }}) "
        f"MERGE (a)-[:`{rel}`{prop_clause}]->(b);"
    )


def export_to_cypher(
    nodes: Iterable[BaseNode],
    edges: Iterable[BaseEdge],
    output_path: Path,
) -> None:
    """Export Node/Edge (Phase-2) sang Cypher, MERGE theo khóa tự nhiên.

    Args:
        nodes: Danh sách node Pydantic.
        edges: Danh sách edge Pydantic.
        output_path: File .cypher để ghi.

    Algorithm:
      1) Ghi MERGE node theo (Label, key_prop, key_value) + SET props còn lại.
      2) Lập registry: identifier string (natural id của schema) -> (Label, key_prop, key_val)
      3) Với mỗi edge:
         - Bỏ qua nếu source/target không map ra node nội bộ.
         - IMPORTS: chỉ giữ khi target là internal module.
         - CALLS: bỏ khi target=None hoặc BUILTIN/EXTERNAL không có node.
         - INHERITS: bỏ nếu base là external (không có node).
         - Còn lại: MATCH src/dst theo registry, MERGE edge.
    """
    lines: list[str] = []

    # 1) Nodes
    #    - Tạo registry cho việc resolve identifier ở edges.
    registry: dict[str, tuple[str, str, str]] = {}
    for n in nodes:
        label, key_name, key_val = _label_key_name_val(n)
        registry[key_val] = (label, key_name, key_val)
        lines.append(_merge_node_stmt(label, key_name, key_val, _node_props(n)))

    # 2) Edges
    for e in edges:
        rel = e.type
        src_id = e.source
        dst_id = e.target

        # --- lọc các trường hợp không thể match ---
        if isinstance(e, CallsEdge):
            # target có thể None/builtin/external → không có node
            if dst_id is None or dst_id not in registry:
                continue

        if isinstance(e, ImportsEdge):
            # chỉ giữ khi target là internal module (có node)
            if dst_id not in registry:
                continue

        if isinstance(e, InheritsEdge):
            # external base (raw) không có node
            if dst_id not in registry:
                continue

        # nguồn/đích đều phải là node nội bộ
        if src_id not in registry or dst_id not in registry:
            continue

        src_info = registry[src_id]
        dst_info = registry[dst_id]
        stmt = _merge_edge_stmt(src_info, dst_info, rel, _edge_props(e))
        lines.append(stmt)

    # 3) dump
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")