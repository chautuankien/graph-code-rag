import pytest
from pathlib import Path

from src.code_graph_rag.parser.ast_parser import ASTParser
from src.code_graph_rag.models.nodes import *
from src.code_graph_rag.models.edges import *


@pytest.fixture
def make_fs(tmp_path):
    """
    Create a reproducible project root named 'proj' under the pytest tmp_path
    and return a helper to materialize a file/folder structure, run ASTParser,
    and return the parser.

    The structure is described as a list of relative POSIX paths from 'proj/'.
    - Directories end with '/'
    - Regular files are created empty
    - Python files create modules
    - A directory containing '__init__.py' is treated as a Package
    """
    proj_root = tmp_path / "proj"
    proj_root.mkdir()

    def _make(structure: list[str]):
        # Materialize structure
        for rel in structure:
            rel_path = Path(rel)
            if rel.endswith("/"):
                (proj_root / rel_path).mkdir(parents=True, exist_ok=True)
            else:
                file_path = proj_root / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("", encoding="utf-8")

        # Run parser only for filesystem walking phase
        p = ASTParser(project_root=str(proj_root))
        # Prefer calling the walker to avoid coupling with later phases
        p._walk_files_and_dirs()
        return p

    return _make

def contains_edges_as_tuples(p: ASTParser):
    return sorted(
        (e.source, e.type, e.target) for e in p.edges if isinstance(e, ContainsEdge)
    )

def nodes_summary(p: ASTParser):
    """Summarize nodes by type and identifier (name/path/qualified_name)."""
    proj = [n.name for n in p.nodes if isinstance(n, ProjectNode)]
    folders = [n.path for n in p.nodes if isinstance(n, FolderNode)]
    packages = [n.qualified_name for n in p.nodes if isinstance(n, PackageNode)]
    modules = [n.qualified_name for n in p.nodes if isinstance(n, ModuleNode)]
    files = [n.path for n in p.nodes if isinstance(n, FileNode)]
    return {
        "ProjectNode": sorted(proj),
        "FolderNode": sorted(set(folders)),
        "PackageNode": sorted(set(packages)),
        "ModuleNode": sorted(set(modules)),
        "FileNode": sorted(set(files)),
    }

@pytest.mark.parametrize(
    "structure, expected_edges, expected_nodes",
    [
        # 0) Empty project
        (
            [],
            [],
            {
                "ProjectNode": ["proj"],
                "FolderNode": [],
                "PackageNode": [],
                "ModuleNode": [],
                "FileNode": [],
            }
        ),

        # 1) Simple project with a non-Python file
        (
            [
                "docs/",
                "docs/readme.txt",
            ],
            [
                ("proj", "CONTAINS_FOLDER", "docs"),
                ("docs", "CONTAINS_FILE", "docs/readme.txt"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": ["docs"],
                "PackageNode": [],
                "ModuleNode": [],
                "FileNode": ["docs/readme.txt"],
            },
        ),
        # 2) Root-level module and non-code file
        (
            [
                "app.py",
                "README.md",
            ],
            [
                ("proj", "CONTAINS_MODULE", "proj.app"),
                ("proj", "CONTAINS_FILE", "README.md"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": [],
                "PackageNode": [],
                "ModuleNode": ["proj.app"],
                "FileNode": ["README.md"],
            },
        ),
        # 3) Root-level package with module (includes __init__.py as a module)
        (
            [
                "pkg/__init__.py",
                "pkg/m.py",
            ],
            [
                ("proj", "CONTAINS_PACKAGE", "proj.pkg"),
                ("proj.pkg", "CONTAINS_MODULE", "proj.pkg.__init__"),
                ("proj.pkg", "CONTAINS_MODULE", "proj.pkg.m"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": [],
                "PackageNode": ["proj.pkg"],
                "ModuleNode": ["proj.pkg.__init__", "proj.pkg.m"],
                "FileNode": [],
            },
        ),
        # 4) Nested package with subpackage and module (includes both __init__.py modules)
        (
            [
                "pkg/__init__.py",
                "pkg/sub/__init__.py",
                "pkg/sub/n.py",
            ],
            [
                ("proj", "CONTAINS_PACKAGE", "proj.pkg"),
                ("proj.pkg", "CONTAINS_MODULE", "proj.pkg.__init__"),
                ("proj.pkg", "CONTAINS_PACKAGE", "proj.pkg.sub"),
                ("proj.pkg.sub", "CONTAINS_MODULE", "proj.pkg.sub.__init__"),
                ("proj.pkg.sub", "CONTAINS_MODULE", "proj.pkg.sub.n"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": [],
                "PackageNode": ["proj.pkg", "proj.pkg.sub"],
                "ModuleNode": ["proj.pkg.__init__", "proj.pkg.sub.__init__", "proj.pkg.sub.n"],
                "FileNode": [],
            },
        ),
        # 5) Folder containing a module (folder is not a package)
        (
            [
                "lib/",
                "lib/util.py",
            ],
            [
                ("proj", "CONTAINS_FOLDER", "lib"),
                ("lib", "CONTAINS_MODULE", "proj.lib.util"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": ["lib"],
                "PackageNode": [],
                "ModuleNode": ["proj.lib.util"],
                "FileNode": [],
            },
        ),
        # 6) Package containing a regular folder with a non-python file (includes __init__.py module)
        (
            [
                "pkg/__init__.py",
                "pkg/docs/",
                "pkg/docs/guide.txt",
            ],
            [
                ("proj", "CONTAINS_PACKAGE", "proj.pkg"),
                ("proj.pkg", "CONTAINS_MODULE", "proj.pkg.__init__"),
                ("proj.pkg", "CONTAINS_FOLDER", "pkg/docs"),
                ("pkg/docs", "CONTAINS_FILE", "pkg/docs/guide.txt"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": ["pkg/docs"],
                "PackageNode": ["proj.pkg"],
                "ModuleNode": ["proj.pkg.__init__"],
                "FileNode": ["pkg/docs/guide.txt"],
            },
        ),
        # 7) Mixed root: module, file, folder->module, package->module (package includes __init__.py module)
        (
            [
                "root_mod.py",
                "data.txt",
                "a/",
                "a/x.py",
                "pkg/__init__.py",
                "pkg/m.py",
            ],
            [
                ("proj", "CONTAINS_MODULE", "proj.root_mod"),
                ("proj", "CONTAINS_FILE", "data.txt"),
                ("proj", "CONTAINS_FOLDER", "a"),
                ("a", "CONTAINS_MODULE", "proj.a.x"),
                ("proj", "CONTAINS_PACKAGE", "proj.pkg"),
                ("proj.pkg", "CONTAINS_MODULE", "proj.pkg.__init__"),
                ("proj.pkg", "CONTAINS_MODULE", "proj.pkg.m"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": ["a"],
                "PackageNode": ["proj.pkg"],
                "ModuleNode": ["proj.a.x", "proj.pkg.__init__", "proj.pkg.m", "proj.root_mod"],
                "FileNode": ["data.txt"],
            },
        ),
        # 8) Package directly contains a non-python file (package -> file) and includes __init__.py module
        (
            [
                "pkg/__init__.py",
                "pkg/README.md",
            ],
            [
                ("proj", "CONTAINS_PACKAGE", "proj.pkg"),
                ("proj.pkg", "CONTAINS_MODULE", "proj.pkg.__init__"),
                ("proj.pkg", "CONTAINS_FILE", "pkg/README.md"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": [],
                "PackageNode": ["proj.pkg"],
                "ModuleNode": ["proj.pkg.__init__"],
                "FileNode": ["pkg/README.md"],
            },
        ),
        # 9) Nested folders: folder -> folder -> module
        (
            [
                "a/",
                "a/b/",
                "a/b/c.py",
            ],
            [
                ("proj", "CONTAINS_FOLDER", "a"),
                ("a", "CONTAINS_FOLDER", "a/b"),
                ("a/b", "CONTAINS_MODULE", "proj.a.b.c"),
            ],
            {
                "ProjectNode": ["proj"],
                "FolderNode": ["a", "a/b"],
                "PackageNode": [],
                "ModuleNode": ["proj.a.b.c"],
                "FileNode": [],
            },
        ),
    ],
)
def test_phase2_1_parse_filesystem(
    make_fs,
    structure: list[str],
    expected_edges: list[tuple[str, str, str]],
    expected_nodes: dict[str, list[str]],
):
    """
    Test the filesystem parsing phase of the ASTParser.
    It should correctly identify folders, packages, modules, and files.
    """
    parser = make_fs(structure)

    # Verify CONTAINS_* edges
    got_edges = contains_edges_as_tuples(parser)
    want_edges = sorted(expected_edges)
    assert got_edges == want_edges, f"Edges mismatch.\nGot:   {got_edges}\nWant:  {want_edges}"

    # Verify nodes by type/identifier
    got_nodes = nodes_summary(parser)
    assert got_nodes == expected_nodes, f"Nodes mismatch.\nGot:   {got_nodes}\nWant:  {expected_nodes}"