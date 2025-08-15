import pytest

from src.code_graph_rag.models.edges import ImportsEdge
from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

@pytest.mark.parametrize(
    "code, expected_imports, extra_files",
    [
        # --- Basic import (external stdlib) ---
        # NOTE: Parser emits an IMPORTS edge to target="os". Exporter will drop it
        #       (not an internal node), so we only assert on the parser output here.
        (
            "import os",
            [ImportsEdge(source="proj.mod", target="os", type="IMPORTS", import_name="os")],
            {},
        ),
        # --- Import with alias (external package) ---
        # Parser normalizes alias to the base 'numpy' in import_name/target.
        # Exporter drops IMPORTS for external targets — still assert on parser edges.
        (
            "import numpy as np",
            [ImportsEdge(source="proj.mod", target="numpy", type="IMPORTS", import_name="numpy")],
            {},
        ),
        # --- Import from package (absolute, inside repo) ---
        # The sample adds mypkg/utils.py so the repo contains that module.
        # Parser keeps target="mypkg.utils" (legacy behavior kept for test parity).
        (
            "from mypkg.utils import helper",
            [
                ImportsEdge(
                    source="proj.mod",
                    target="mypkg.utils",
                    type="IMPORTS",
                    import_name="mypkg.utils.helper",
                )
            ],
            {
                # Define the imported module and symbol inside the repo
                "mypkg/utils.py": "def helper():\n    return 1\n",
                "__init__.py": "",
                "mypkg/__init__.py": "",
            },
        ),
        # --- Wildcard import ---
        (
            "from mypkg.utils import *",
            [
                ImportsEdge(
                    source="proj.mod",
                    target="mypkg.utils",
                    type="IMPORTS",
                    import_name="mypkg.utils.*",
                )
            ],
            {
                "mypkg/utils.py": "def helper():\n    return 1\n",
                "__init__.py": "",
                "mypkg/__init__.py": "",
            },
        ),
        # --- Relative import ---
        # Legacy parser behavior resolves '.submodule' relative to 'proj.mod'
        # into 'proj.mod.submodule' (kept to match previous unit tests).
        (
            "from .submodule import Thing",
            [
                ImportsEdge(
                    source="proj.mod",
                    target="proj.mod.submodule",
                    type="IMPORTS",
                    import_name="proj.mod.submodule.Thing",
                )
            ],
            {
                "submodule.py": "class Thing:\n    pass\n",
                "__init__.py": "",  # mark repo root as a package if parser needs it
            },
        ),

        (
            "",
            [ImportsEdge(source="proj.pkg.a", target="proj.pkg.b", type="IMPORTS", import_name="pkg.b")],
            {
                "__init__.py": "",
                "pkg/b.py": (
                    "def g():\n"
                    "    return 1\n"
                ),
                "pkg/a.py": (
                    "import pkg.b\n"
                    "def f():\n"
                    "    return pkg.b.g()\n"
                ),
 
            }
        )
    ],
)
def test_imports_edge(make_parser, code, expected_imports, extra_files):
    """
    Integration-flavored test for the import parser:
      - Builds a tiny repo under a directory named 'proj'
      - Loads the graph into Memgraph using the real build pipeline
      - Runs ASTParser on 'proj.mod' and asserts the emitted ImportsEdge list

    NOTE:
      We intentionally assert on the parser edges (not DB edges) because the
      exporter keeps only IMPORTS to internal modules. External imports such as
      'os'/'numpy' are dropped at export time by design.
    """
    parser = make_parser(code, extra_files=extra_files)

    imports_edges = [e for e in parser.edges if isinstance(e, ImportsEdge)]
    log.debug("test_imports_edge: imports_edges=%s", imports_edges)

    got = sorted((e.source, e.target, e.import_name) for e in imports_edges)
    want = sorted((e.source, e.target, e.import_name) for e in expected_imports)
    assert got == want, f"Expected {want}, but got {got} for code:\n{code}"
