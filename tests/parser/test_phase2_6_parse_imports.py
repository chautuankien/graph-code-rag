import pytest
from src.code_graph_rag.models.edges import ImportsEdge
from src.code_graph_rag.parser.ast_parser import ASTParser

@pytest.fixture
def make_parser(tmp_path):
    """
    Returns a function that writes `code` to a .py file under tmp_path,
    runs ASTParser on it, and returns the parser instance.
    """
    def _make(code:str, module_qname: str = "proj.mod"):
        # Write code to a temporary .py file
        path = tmp_path / "mod.py"
        path.write_text(code, encoding="utf-8")
        # Instantiate ASTParser configured for this module
        p = ASTParser(project_root=tmp_path)
        p.module_symbols[module_qname] = module_qname
        p._parse_module(path, module_qname)
        return p
    return _make

@pytest.mark.parametrize("code, expected_imports",[
    # --- Basic import ---
    (
        "import os",
        [ImportsEdge(source="proj.mod", target="os", type="IMPORTS", import_name="os")]
    ),
    # --- Import with alias ---
    (
        "import numpy as np",
        [ImportsEdge(source="proj.mod", target="numpy", type="IMPORTS", import_name="numpy")]
    ),
    # --- Import from package ---
    (
        "from mypkg.utils import helper",
        [ImportsEdge(source="proj.mod", target="mypkg.utils", type="IMPORTS", import_name="mypkg.utils.helper")]
    ),
    # --- Wildcard import ---
    (
        "from mypkg.utils import *",
        [ImportsEdge(source="proj.mod", target="mypkg.utils", type="IMPORTS", import_name="mypkg.utils.*")]
    ),
    # --- Relative import ---
    (
        "from .submodule import Thing",
        [ImportsEdge(source="proj.mod", target="proj.mod.submodule", type="IMPORTS", import_name="proj.mod.submodule.Thing")]
    ),
])
def test_imports_edge(make_parser, code, expected_imports):
    parser = make_parser(code, module_qname="proj.mod")
    # Extract imports edges from the parser
    imports_edges = [e for e in parser.edges if isinstance(e, ImportsEdge)]
    # compare by tuple representation for clarity
    got = sorted(
        (e.source, e.target, e.import_name) for e in imports_edges
    )
    want = sorted(
        (e.source, e.target, e.import_name) for e in expected_imports
    )
    assert got == want, f"Expected {want}, but got {got} for code:\n{code}"