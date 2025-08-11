import pytest
from src.code_graph_rag.parser.ast_parser import ASTParser
from src.code_graph_rag.models.edges import DependsOnExternalEdge

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

@pytest.mark.parametrize("code, expect_pkgs", [
    ("import requests", {"requests"}),
    ("from PIL import Image", {"pillow"}),     # mapped
    ("import cv2 as cv", {"opencv-python"}),   # mapped
    ("from yaml import safe_load", {"pyyaml"}),# mapped
    ("from .local import util", set()),        # relative -> skip
    ("from ..pkg.sub import x", set()),          # relative -> skip
    # ("import proj.internal", set()),           # internal -> skip (seed module_symbols)
])
def test_module_depends_edges(make_parser, code, expect_pkgs):
    p = make_parser(code, module_qname="proj.mod")
    depends = {e.target for e in p.edges if isinstance(e, DependsOnExternalEdge)
               and e.source == "proj.mod"}
    assert depends == expect_pkgs, f"Expected {expect_pkgs}, but got {depends} for code:\n{code}"
