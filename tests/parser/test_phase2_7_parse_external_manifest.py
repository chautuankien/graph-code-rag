import os
import textwrap
import pytest

from src.code_graph_rag.parser.ast_parser import ASTParser
from src.code_graph_rag.models.nodes import ExternalPackageNode
from src.code_graph_rag.models.edges import DependsOnExternalEdge

def _collect_external_nodes(nodes):
    """Return a dict: name -> node for ExternalPackageNode."""
    return {n.name: n for n in nodes if isinstance(n, ExternalPackageNode)}


def _project_dep_edges(edges, project_name: str | None = None):
    """
    Return all project-level DEPENDS_ON_EXTERNAL edges.
    If project_name is provided, filter by source == project_name.
    """
    out = [e for e in edges if isinstance(e, DependsOnExternalEdge)]
    if project_name is not None:
        out = [e for e in out if getattr(e, "source", None) == project_name]
    return out

def _edge_targets(edges):
    """Helper: return set of edge.target from a list of edges."""
    return {e.target for e in edges}

def test_pep621_dependencies(tmp_path):
    """
    PEP 621: [project].dependencies as a list of strings should produce
    ExternalPackageNodes and Project -> DEPENDS_ON_EXTERNAL edges.
    """
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(textwrap.dedent("""
        [project]
        name = "demo"
        version = "0.0.1"
        dependencies = [
            "requests>=2.31,<3",
            "pydantic>=2",
        ]
    """).strip(), encoding="utf-8")

    parser = ASTParser(project_root=tmp_path)
    nodes, edges = parser.parse()

    project_name = getattr(parser, "project_name", tmp_path.name)
    deps = _project_dep_edges(edges, project_name)
    targets = _edge_targets(deps)

    # External nodes present
    ext = _collect_external_nodes(nodes)
    assert "requests" in ext and "pydantic" in ext
    assert ext["requests"].version_spec == ">=2.31,<3"
    assert ext["pydantic"].version_spec == ">=2"

    # Project-level edges present
    assert {"requests", "pydantic"} <= targets

def test_pep621_optional_dependencies(tmp_path):
    """
    PEP 621: [project.optional-dependencies] should be merged and emitted
    as ExternalPackageNode + Project-level DEPENDS_ON_EXTERNAL edges.
    """
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(textwrap.dedent("""
        [project]
        name = "demo"
        version = "0.0.1"
        dependencies = ["numpy"]

        [project.optional-dependencies]
        image = ["Pillow==10.0.0"]
        dev = ["mypy>=1.7"]
    """).strip(), encoding="utf-8")

    parser = ASTParser(project_root=tmp_path)
    nodes, edges = parser.parse()

    project_name = getattr(parser, "project_name", tmp_path.name)
    deps = _project_dep_edges(edges, project_name)
    targets = _edge_targets(deps)

    ext = _collect_external_nodes(nodes)
    # Names are normalized per PEP 503 (lowercase, collapse punctuation to '-').
    assert "numpy" in ext
    assert "pillow" in ext
    assert "mypy" in ext
    assert ext["pillow"].version_spec == "==10.0.0"
    assert {"numpy", "pillow", "mypy"} <= targets

def test_poetry_dependencies_and_groups(tmp_path):
    """
    Poetry layout:
      [tool.poetry.dependencies] (dict, ignore 'python')
      [tool.poetry.group.<name>.dependencies] (dict)
    should be parsed and merged.
    """
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(textwrap.dedent("""
        [tool.poetry]
        name = "demo"
        version = "0.0.1"

        [tool.poetry.dependencies]
        python = "^3.11"
        Pillow = "^10.3"
        pyyaml = ">=6"

        [tool.poetry.group.dev.dependencies]
        pytest = "^8.2"
        mypy = { version = ">=1.7" }
    """).strip(), encoding="utf-8")

    parser = ASTParser(project_root=tmp_path)
    nodes, edges = parser.parse()

    project_name = getattr(parser, "project_name", tmp_path.name)
    deps = _project_dep_edges(edges, project_name)
    targets = _edge_targets(deps)

    ext = _collect_external_nodes(nodes)
    # 'python' should be ignored, others captured
    assert "pillow" in ext and "pyyaml" in ext and "pytest" in ext and "mypy" in ext
    assert ext["pillow"].version_spec == "^10.3"
    assert ext["pyyaml"].version_spec == ">=6"
    assert ext["pytest"].version_spec == "^8.2"
    assert ext["mypy"].version_spec == ">=1.7"
    assert {"pillow", "pyyaml", "pytest", "mypy"} <= targets

def test_requirements_only(tmp_path):
    """
    If only requirements.txt exists, it should be parsed and emitted.
    """
    (tmp_path / "requirements.txt").write_text(textwrap.dedent("""
        pandas>=2.2,<3
        bs4
        # comment line
    """).strip(), encoding="utf-8")

    parser = ASTParser(project_root=tmp_path)
    nodes, edges = parser.parse()

    project_name = getattr(parser, "project_name", tmp_path.name)
    deps = _project_dep_edges(edges, project_name)
    targets = _edge_targets(deps)

    ext = _collect_external_nodes(nodes)
    assert "pandas" in ext and "bs4" in ext
    assert ext["pandas"].version_spec == ">=2.2,<3"
    # No version for bs4
    assert ext["bs4"].version_spec in ("", None)
    assert {"pandas", "bs4"} <= targets