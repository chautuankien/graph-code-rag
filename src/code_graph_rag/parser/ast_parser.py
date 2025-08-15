import os
import ast
import re
from pathlib import Path
import builtins
import tomllib
import glob
from collections import defaultdict

from src.code_graph_rag.models.nodes import *
from src.code_graph_rag.models.edges import *

class ASTParser():
    def __init__(self, project_root: str):
        """
        Initialize a new ASTParser for a given codebase root.

        Parameters
        ----------
        project_root : str
            Path to the repository or project directory to analyze. 

        Attributes initialized
        ----------------------
        project_root : pathlib.Path
            Absolute path to the project root (anchor for walking and relative paths).
        project_name : str
            Name of the project, taken from the root directory's basename. Used as a
            stable prefix for qualified names (qnames) to ensure global uniqueness.
        nodes : list[BaseNode]
            Collector for all node objects created by the parser (Project, Folder,
            Package, Module, Class, Function, Method, ExternalPackage).
        edges : list[BaseEdge]
            Collector for all relationship objects (CONTAINS_*, DEFINES(_METHOD),
            CALLS, IMPORTS, INHERITS, OVERRIDES, DEPENDS_ON_EXTERNAL).
        folder_qname_map : dict[str, str]
            Maps a folder path (relative to project root, e.g., "src/models")
            to its package qualified_name if it is a Python package (has __init__.py).
            This ensures correct parent-child relationships when building CONTAINS_* edges:
            - Packages use qualified_name identifiers
            - Folders use path identifiers
        module_symbols : dict[str, str]
            Registry of internal modules discovered during filesystem walk.
            Key and value are the module's qualified name for quick membership checks.
        import_map : dict[str, str]
            Per-parser alias resolution map for imports (alias → resolved target).
            Populated in _handle_imports and used for later call resolution phases.
        func_symbols : dict[str, str]
            Global function symbol table: function_name → function_qualified_name.
            Filled when encountering FunctionDef outside class scopes (also nested functions).
        method_symbols : dict[str, str]
            Global method symbol table: method_name → method_qualified_name.
            Filled when encountering FunctionDef inside class scopes.
        class_symbols : dict[str, str]
            Class symbol table: class_name → class_qualified_name.
        builtin_funcs : set[str]
            Snapshot of Python builtins (dir(builtins)) for quick detection of built-in calls.
        class_method_symbols : dict[str, dict[str, str]]
            Per-class method tables: class_qname → { method_name: method_qname }.
            Enables later detection of OVERRIDES across inheritance hierarchies.
        class_bases : dict[str, list[str]]
            Per-class base list: class_qname → [base_target, ...]. Targets may be internal
            qnames or raw dotted names if unresolved/external.
        external_packages : dict[str, ExternalPackageNode]
            Cache of ExternalPackageNode objects keyed by normalized (PEP 503) package name.
            Ensures idempotent creation and later version_spec upgrades.
        mod_to_external_used : dict[str, set[str]]
            Tracks module-level external usage: module_qname → set(normalized_pkg_names).
            Prevents duplicate module → DEPENDS_ON_EXTERNAL edges.
        """
        self.project_root = Path(project_root).resolve()
        self.project_name = self.project_root.name  # Name of the project root directory

        self.nodes = [] # Accumulator for all node instances the parser discovers/emits
        self.edges = [] # Accumulator for all edge instances the parser discovers/emits

        # Maps folder paths to their qualified names when they are Python packages
        # Key: folder path (str) - relative path from project root (e.g., "src/models")  
        # Value: qualified name (str) - dot-separated identifier (e.g., "myproject.src.models")
        # This mapping is essential for establishing correct parent-child relationships
        # because packages use qualified_name as identifier while folders use path
        self.folder_qname_map = {}

        self.module_symbols = {}     # Internal module registry: mod_qname → mod_qname
        self.import_map = {}         # Import alias resolution across modules: alias → resolved target
        self.func_symbols = {}       # Top-level/nested function table: name → qualified_name
        self.method_symbols = {}     # Method table: name → qualified_name (populated within class scope)
        self.class_symbols = {}      # Class table: name → qualified_name
        self.builtin_funcs = set(dir(builtins))

        self.class_method_symbols = {}  # {class_qname: {method_name: method_qualified_name}}
        self.class_bases = {}          # {class_qname: [base_class_target]}

        # External packages cache: normalized_name -> ExternalPackageNode
        self.external_packages: dict[str, ExternalPackageNode] = {}
        self.mod_to_external_used: dict[str, set[str]] = defaultdict(set)
    
    def parse(self) -> tuple[list, list]:
        """Run the full parse and return lists of node and edge objects.

        Overview
        --------
        parse() coordinates the main phases to construct the code knowledge graph:
          1) Filesystem traversal (Phase 2.2)
             - Create structural nodes (Project, Folder, Package, Module, File)
             - Establish CONTAINS_* edges
             - Seed symbol tables for internal modules
             - Parse each Python module to harvest classes, functions, methods, imports, calls
          2) Declared dependency discovery (Phase 2.7)
             - Read pyproject.toml and/or requirements*.txt
             - Upsert ExternalPackageNode for each dependency
             - Emit Project → DEPENDS_ON_EXTERNAL edges (deduped)
          3) Post-processing overrides (Phase 2.5.2)
             - Based on previously collected class_bases and per-class method tables
             - Emit OVERRIDES edges for methods shadowing base-class methods

        Returns
        -------
        (nodes, edges) : tuple[list, list]
            The complete list of node and edge objects created during parsing.
        """

        self._walk_files_and_dirs()
        self._parse_dependencies_from_manifest()
        self._handle_overrides()

        return self.nodes, self.edges
    
    def _walk_files_and_dirs(self):
        """
        Traverse the project filesystem to construct the structural part of the graph.

        Why this method exists
        ----------------------
        This method is responsible for:
        - Discovering structural elements: Project, Folder, Package (dir with __init__.py),
          Module (.py files), and non-Python File.
        - Emitting CONTAINS_* edges to reflect the containment hierarchy.
        - Seeding symbol tables (e.g., module_symbols) used by later phases (imports, calls).
        - Ensuring consistent identifier strategy across nodes:
          - ProjectNode: identified by `name` (the basename of project root path).
          - PackageNode: identified by `qualified_name` (qname).
          - FolderNode: identified by `path` (relative path).
          - ModuleNode: identified by `qualified_name` (qname).
          - FileNode: identified by `path` (relative path).

        Identifier strategy & containment rules
        ---------------------------------------
        - Qualified names are prefixed with `project_name` for uniqueness across repositories:
            project_name.[subdirs].[module_name]
        - Parent container selection uses both physical path and package mapping:
          - If a directory is a Python package (has __init__.py), we create a PackageNode and
            register folder_qname_map[path] = package_qname so that children attach to qname.
          - Otherwise, we create a FolderNode and children attach to its path.
        - Edge emission:
          - Project/Folder/Package --CONTAINS_PACKAGE--> Package (target=qualified_name)
          - Project/Folder/Package --CONTAINS_FOLDER--> Folder (target=path)
          - Project/Folder/Package --CONTAINS_MODULE--> Module (target=qualified_name)
          - Project/Folder/Package --CONTAINS_FILE--> File (target=path)

        Example
        -------
        For a project "myproj" with:
          src/
            __init__.py
            util.py
            data/
              __init__.py
              loader.py
            assets/
              logo.png

        This method creates:
          - ProjectNode("myproj")
          - PackageNode("myproj.src"), ModuleNode("myproj.src.util")
          - PackageNode("myproj.src.data"), ModuleNode("myproj.src.data.loader")
          - FolderNode("src/assets"), FileNode("src/assets/logo.png")
        And edges:
          - myproj --CONTAINS_PACKAGE--> myproj.src
          - myproj.src --CONTAINS_MODULE--> myproj.src.__init__
          - myproj.src --CONTAINS_MODULE--> myproj.src.util
          - myproj.src --CONTAINS_PACKAGE--> myproj.src.data
          - myproj.src.data --CONTAINS_MODULE--> myproj.src.data.__init__
          - myproj.src.data --CONTAINS_MODULE--> myproj.src.data.loader
          - myproj.src --CONTAINS_FOLDER--> src/assets
          - src/assets --CONTAINS_FILE--> src/assets/logo.png
        """
        # Create the root project node using the project directory name
        project_node = ProjectNode(name=self.project_root.name)
        self.nodes.append(project_node)

        # 1️⃣ Pre-scan để seed toàn bộ module_symbols
        for dirpath, _, filenames in os.walk(self.project_root):
            current_path = Path(dirpath)
            for file in filenames:
                if file.endswith(".py"):
                    rel_file_path = (current_path / file).relative_to(self.project_root)
                    mod_qname = ".".join([self.project_root.name] + list(rel_file_path.with_suffix('').parts))
                    self.module_symbols[mod_qname] = mod_qname

        # 2️⃣ Parse structure + modules
        # Recursively traverse all directories and files in the project
        for dirpath, dirnames, filenames in os.walk(self.project_root):
            current_path = Path(dirpath)
            rel_path = current_path.relative_to(self.project_root)

            # Detect if current directory is a Python package (contains __init__.py)
            is_package = "__init__.py" in filenames
            
            # Check if we're processing the project root directory
            current_is_root = rel_path == Path(".") or not rel_path.parts

            # Generate path string and qualified name for current directory
            current_path_str = str(rel_path)
            # Qualified name for packages/modules: prefix with project name, then join relative parts.
            current_qname = ".".join([self.project_root.name] + list(rel_path.parts))

            def get_parent_source():
                """
                Determine the parent container's identifier and type for CONTAINS_* edges.

                Returns
                -------
                (parent_identifier, parent_type) : tuple[str, str]
                    parent_identifier:
                      - ProjectNode.name for direct children of the root
                      - PackageNode.qualified_name if the parent directory is a package
                      - FolderNode.path if the parent directory is a regular folder
                    parent_type:
                      - "Project" | "Package" | "Folder" (informational; not used in edge schema)
                """
                parent_path = rel_path.parent # Relative path of the parent directory
                
                # If there is no parent (this is an immediate child of the project root),
                # the parent container is the ProjectNode (identified by project_name).
                if not parent_path.parts:
                    return self.project_root.name, "Project"
                else:
                    parent_path_str = str(parent_path)
                    # If the parent directory was registered as a package, use its qname.
                    if parent_path_str in self.folder_qname_map:
                        return self.folder_qname_map[parent_path_str], "Package"
                    # Otherwise, it is a regular folder; use its path as the identifier.
                    else:
                        return parent_path_str, "Folder"

            # For non-root directories, create either a PackageNode or FolderNode
            # and wire it to its parent with the appropriate CONTAINS_* edge.
            if not current_is_root:
                # Get parent information for establishing containment relationships
                parent_source, parent_type = get_parent_source()    # Obtain parent identifier for edge source
                
                if is_package:
                    # Create a Package node for directories containing __init__.py
                    pkg_node = PackageNode(
                        qualified_name=current_qname,  # Unique qualified name
                        name=current_path.name,        # Simple directory name
                        path=current_path_str          # Relative path from project root
                    )
                    self.nodes.append(pkg_node)
                    
                    # Remember that this directory is a package; future children should attach by qname
                    self.folder_qname_map[current_path_str] = current_qname
                    
                    # Create CONTAINS_PACKAGE relationship from parent to this package
                    self.edges.append(ContainsEdge(
                        source=parent_source,     # Parent identifier (varies by parent type)
                        target=current_qname,     # Package uses qualified_name as identifier
                        type="CONTAINS_PACKAGE"
                    ))
                else:
                    # Create a FolderNode (regular directory without __init__.py)
                    folder_node = FolderNode(
                        path=current_path_str,    # Relative path serves as identifier
                        name=current_path.name    # Simple directory name
                    )
                    self.nodes.append(folder_node)
                    
                    # Create CONTAINS_FOLDER relationship from parent to this folder
                    self.edges.append(ContainsEdge(
                        source=parent_source,     # Parent identifier (varies by parent type)
                        target=current_path_str,  # Folder uses path as identifier
                        type="CONTAINS_FOLDER"
                    ))

            # Handle files located in the current directory (both Python and non-Python)
            for file in filenames:
                file_path = current_path / file
                rel_file_path = file_path.relative_to(self.project_root)
                ext = file_path.suffix

                # Determine which container "owns" this file for the CONTAINS_* edge:
                # - If we are at the root, the container is the ProjectNode (identified by project_name).
                # - If current dir is a package, use the package qname.
                # - Otherwise, use the folder's path.
                if current_is_root:
                    container_source = self.project_root.name
                else:
                    if is_package:
                        # File is in a package directory - use package qualified_name
                        container_source = current_qname
                    else:
                        # File is in a regular folder - use folder path
                        container_source = current_path_str

                if ext == ".py":
                    # For Python source files, create a ModuleNode.
                    # Module qname = project_name + relative path without extension, with path separators replaced by dots.
                    mod_qname = ".".join([self.project_root.name] + list(rel_file_path.with_suffix('').parts))

                    module_node = ModuleNode(
                        qualified_name=mod_qname,      # Unique qualified name
                        name=file,                     # Filename with extension
                        path=str(rel_file_path)        # Relative path from project root
                    )
                    self.nodes.append(module_node)
                    
                    # Create CONTAINS_MODULE relationship from container to module
                    self.edges.append(ContainsEdge(
                        source=container_source,  # Container identifier (varies by container type)
                        target=mod_qname,         # Module uses qualified_name as identifier
                        type="CONTAINS_MODULE"
                    ))

                    # Parse the module to extract semantic elements (classes, functions, imports, calls, etc.).
                    self._parse_module(file_path, mod_qname)
                else:
                    # For non-Python files, create a FileNode with its relative path as identifier.
                    file_node = FileNode(
                        path=str(rel_file_path),  # Relative path serves as identifier
                        name=file,                # Filename with extension
                        extension=ext             # File extension for categorization
                    )
                    self.nodes.append(file_node)
                    
                    # Create CONTAINS_FILE relationship from container to file
                    self.edges.append(ContainsEdge(
                        source=container_source,      # Container identifier (varies by container type)
                        target=str(rel_file_path),     # File uses path as identifier
                        type="CONTAINS_FILE"
                    ))

    def _parse_dependencies_from_manifest(self):
        """
        Discover declared external dependencies from project manifest files
        (prefer `pyproject.toml`, fall back to `requirements*.txt`) and
        materialize them into the graph as:
        1) ExternalPackageNode(name=<normalized>, version_spec=<constraint>)
        2) Project-level edges: ProjectNode --DEPENDS_ON_EXTERNAL--> ExternalPackageNode

        Why this method exists
        ----------------------
        - “ImportsEdge” found in source files tell us which *modules* are used,
        but they don’t tell us the *declared* dependency list nor version constraints.
        - The manifest is the source of truth for declared packages and versions.
        - We emit **Project → DEPENDS_ON_EXTERNAL** edges for the whole project to capture
        declared intent, independent of which modules import what.

        What counts as a dependency
        ---------------------------
        - `pyproject.toml`:
            * PEP 621: `[project].dependencies` (list of strings)
            * Optional deps: `[project.optional-dependencies]` (dict of lists)
            * Poetry: `[tool.poetry.dependencies]` (dict; ignore `"python"`)
            * Poetry groups (optional): `[tool.poetry.group.*.dependencies]` (dict)
        If `pyproject.toml` exists, it is preferred and we do not look at `requirements*.txt`
        unless you want to merge (this implementation *does* merge if both exist).
        - `requirements*.txt` (fallback or merge):
            * Each non-empty, non-comment line is parsed via `_split_req_line()`.

        Idempotency & deduplication
        ---------------------------
        - `_upsert_external_package()` guarantees one node per normalized package name and
        upgrades `version_spec` when we learn more (e.g., from manifest).
        - We deduplicate **Project-level** DEPENDS_ON_EXTERNAL edges by checking existing edges
        where `source == self.project_name`.

        Side effects
        ------------
        - Appends new ExternalPackageNode objects to `self.nodes` (only on first sight).
        - Appends new Project-level DependsOnExternalEdge objects to `self.edges`
        (only if not already present).
        - Safe to call multiple times.

        Examples
        --------
        pyproject.toml (PEP 621):
            [project]
            dependencies = [
                "requests>=2.31,<3",
                "pydantic>=2",
            ]

        requirements.txt:
            numpy
            pandas>=2.1,<3  # pinned range

        Both inputs will produce ExternalPackageNode("requests", ">=2.31,<3"), etc.,
        and ProjectNode("proj") --DEPENDS_ON_EXTERNAL--> ExternalPackageNode("requests").
        """
        # Collect all dependencies we discover into a temporary dict:
        #   normalized_package_name -> version_spec (string, possibly empty)
        # If the same package is found multiple times, last one wins (which is fine;
        # `_upsert_external_package` merges/keeps the most informative spec).
        discovered: dict[str, str] = {}

        # Utility to add a (name, spec) into the `discovered` dict consistently:
        def _record_dep(raw_name: str, spec: str) -> None:
            if not raw_name:
                return
            # We do *not* normalize here; let _upsert_external_package() do normalization
            # so that all normalization rules remain in one place.
            # However, for dict keying we *do* want stable keys; use the same normalization
            # as the node creation to avoid duplicate inserts in this local dict.
            norm = self._norm_pkg_name(raw_name)
            # Prefer non-empty spec if we already recorded an empty one earlier.
            if norm in discovered:
                if not discovered[norm] and spec:
                    discovered[norm] = spec
            else:
                discovered[norm] = spec or ""

        # ----------------------------------------------------------
        # 1) Parse pyproject.toml if present (preferred information)
        # ----------------------------------------------------------
    
        pyproject = Path(self.project_root) / "pyproject.toml"
        if pyproject.exists():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except Exception:
                data = {}

            # PEP 621: [project].dependencies (list of strings)
            for item in (data.get("project", {}) or {}).get("dependencies", []) or []:
                result = self._split_req_line(item)
                if result:
                    name, spec = result
                    _record_dep(name, spec)
            
            # PEP 621 optional dependencies: [project.optional-dependencies]
            # Structure: { "extra_name": ["pkg1", "pkg2>=1", ...], ... }
            opt_deps = (data.get("project", {}) or {}).get("optional-dependencies", {}) or {}
            for _extra, lines in opt_deps.items():
                for item in lines or []:
                    result = self._split_req_line(item)
                    if result:
                        name, spec = result
                        _record_dep(name, spec)
            
            # Poetry: [tool.poetry.dependencies]  (dict: name -> spec or table)
            poetry_deps = (data.get("tool", {}) or {}).get("poetry", {}).get("dependencies", {}) or {}
            for name, spec in poetry_deps.items():
                if str(name).lower() == "python":
                    continue  # Not a distribution; it's the interpreter constraint.
                # Poetry can express version as a string or as a table with `version` field.
                if isinstance(spec, str):
                    _record_dep(name, spec)
                elif isinstance(spec, dict):
                    _record_dep(name, spec.get("version", ""))
                
            # Poetry groups (optional): [tool.poetry.group.<grp>.dependencies]
            poetry_groups = (data.get("tool", {}) or {}).get("poetry", {}).get("group", {}) or {}
            for _grp, section in poetry_groups.items():
                group_deps = (section or {}).get("dependencies", {}) or {}
                for name, spec in group_deps.items():
                    if isinstance(spec, str):
                        _record_dep(name, spec)
                    elif isinstance(spec, dict):
                        _record_dep(name, spec.get("version", ""))
            
        # ----------------------------------------------------------------
        # 2) requirements*.txt fallback/merge (only if files actually exist)
        # ----------------------------------------------------------------
        # If we haven't discovered any dependencies yet, fall back to requirements files.
        if not discovered:
            req_glob = str(Path(self.project_root) / "requirements*.txt")
            for req_path in glob.glob(req_glob):
                try:
                    with open(req_path, "r", encoding="utf-8") as f:
                        for raw_line in f:
                            result = self._split_req_line(raw_line)
                            if not result:
                                continue
                            name, spec = result
                            _record_dep(name, spec)
                except OSError:
                    # Non-fatal: skip unreadable files
                    continue

        # ----------------------------------------------------------------------
        # 3) Upsert nodes and emit Project-level DEPENDS_ON_EXTERNAL edges (dedup)
        # ----------------------------------------------------------------------
        # Build a set of already-emitted project-level dependencies to avoid duplicates,
        # while leaving module-level edges untouched.
        already_emitted = {
            e.target
            for e in self.edges
            if getattr(e, "type", None) == "DEPENDS_ON_EXTERNAL"
            and getattr(e, "source", None) == self.project_name
        }

        for norm_pkg, spec in discovered.items():
            # Ensure there is a node for this package (create or update version_spec).
            node = self._upsert_external_package(norm_pkg, spec)

            # Emit Project → DEPENDS_ON_EXTERNAL only once per package.
            if node.name not in already_emitted:
                self.edges.append(
                    DependsOnExternalEdge(
                        source=self.project_name,  # Project node identifier
                        target=node.name,          # External package node name (normalized)
                        type="DEPENDS_ON_EXTERNAL"
                    )
                )
                already_emitted.add(node.name)

    def _parse_module(self, module_path: Path, mod_qname: str):
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()

        try: 
            tree = ast.parse(source)
        except SyntaxError:
            return
        
        # Handle imports first to ensure all symbols are registered
        self._handle_imports(tree, mod_qname)

        # Context stack: track current ClassDef to distinguish function vs method
        context_stack = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._handle_class(node, mod_qname, context_stack)
            elif isinstance(node, ast.FunctionDef):
                self._handle_function(node, mod_qname, context_stack)
    
    def _handle_imports(self, tree: ast.AST, mod_qname: str):
        """
        Parse top-level import statements in a module and emit IMPORTS edges.

        Parameters:
            tree (ast.AST): Parsed AST of the current module.
            mod_qname (str): Qualified name of the current module (e.g., "proj.pkg.mod").

        Behavior:
        - Handles two forms:
          A) `import x [as y]`:
             - Records alias → target in self.import_map.
             - Emits ImportsEdge(source=mod_qname, target=x, import_name=x).
          B) `from pkg.mod import name [as alias]` (including relative imports with leading dots):
             - Resolves the base module for relative imports using node.level.
             - Builds a normalized import_name (e.g., "pkg.mod.name" or just "pkg.mod" for "*").
             - Records alias → target in self.import_map.
             - Emits ImportsEdge(source=mod_qname, target=import_name, import_name=import_name).

        Notes:
        - Internal modules already discovered are stored in self.module_symbols.
          If an import points to an internal module, target_qname equals that qualified name.
          Otherwise, the raw dotted string is kept (treated as external or unresolved).
        - node.level indicates the number of leading dots in relative imports.
          Example: from ..utils import helper (level=2).
        """
        for node in tree.body:  # Iterate over top-level statements in the module
            # -------- Case A: `import foo [as bar]` --------
            if isinstance(node, ast.Import):  # Handle absolute imports like `import os` or `import pkg.mod`
                for alias in node.names:  # Multiple names can be imported in one statement
                    # raw module name
                    full_name = alias.name            # e.g. "os" or "mypkg.sub"
                    alias_name = alias.asname or alias.name  # Use alias if provided; otherwise the original name
                    # resolve target: is it internal?

                    # Try to resolve the imported name as an internal module.
                    # Internal modules are stored in self.module_symbols with a project-prefixed qname
                    # (e.g., "proj.pkg.b"), but the AST may give us an absolute package path
                    # relative to the project root (e.g., "pkg.b").
                    # 1) First, prepend project_name and check (matches most internal imports).
                    # 2) If not found, check the raw full_name directly (covers edge cases where
                    #    the module_symbols already contains it without prefix).
                    # 3) Otherwise, treat as external/unresolved and keep the raw full_name.
                    project_prefixed = f"{self.project_name}.{full_name}"
                    if project_prefixed in self.module_symbols:
                        target_qname = project_prefixed
                    elif full_name in self.module_symbols:
                        target_qname = full_name
                    else:
                        target_qname = full_name

                    # record alias → target for later resolution
                    self.import_map[alias_name] = target_qname  # Map imported symbol/alias to its resolved target
                    # emit edge
                    self.edges.append(ImportsEdge(
                        source=mod_qname,            # The module performing the import
                        target=target_qname,         # The module/symbol being imported (internal or raw)
                        type="IMPORTS",              # Relationship label
                        import_name=full_name        # Textual module path as written in code
                    ))

                    # Record external use for this module
                    self._record_external_use(mod_qname, full_name, is_relative=False)

            # ----- Case B: `from pkg.mod import name [as alias]` -----
            elif isinstance(node, ast.ImportFrom):  # Handle `from ... import ...` forms (supports relative imports)
                # Relative if node.level > 0
                is_rel = node.level and node.level > 0
                
                module_part = node.module or ""  # Base specified after `from`; empty for `from . import X`
                # compute base qname for relative imports
                if node.level > 0:  # node.level counts leading dots in `from ....` (number of package levels to go up)
                    # base_parts = mod_qname.split('.')[:-node.level]  # Strip `level` trailing parts from current module qname
                    base_parts = mod_qname.split('.')
                    module_part = '.'.join(base_parts + ([module_part] if module_part else []))  # Append explicit module if present
                for alias in node.names:  # Each imported name (could be multiple, or a wildcard "*")
                    name_part = alias.name          # e.g. "helper" or "*"
                    alias_name = alias.asname or name_part  # Use alias if present; otherwise the imported name
                    
                    # full import string + target resolution (preserve base module as target for wildcard)
                    if name_part == "*":
                        import_name = f"{module_part}.*"   # desired textual form for wildcard
                        target_base = module_part          # keep target as the base module
                    else:
                        import_name = f"{module_part}.{name_part}"
                        target_base = module_part

                    # Normalize for internal match
                    project_prefixed = f"{self.project_name}.{target_base}"
                    if project_prefixed in self.module_symbols:
                        target_qname = project_prefixed 
                    elif target_base in self.module_symbols:
                        target_qname = target_base 
                    else:
                        target_qname = target_base

                    # update import_map
                    self.import_map[alias_name] = target_qname  # Map alias/name → resolved target
                    # emit edge
                    self.edges.append(ImportsEdge(
                        source=mod_qname,           # The module performing the import
                        target=target_qname,        # The resolved module/symbol name
                        type="IMPORTS",             # Relationship label
                        import_name=import_name     # Normalized textual import path
                    ))

                    # Record external use for this module
                    self._record_external_use(mod_qname, module_part, is_relative=is_rel)

    def _handle_class(self, node: ast.ClassDef, mod_qname: str, context_stack: list):
        """
        Register a class definition, emit structural/semantic edges, and enqueue its internals.

        Why this method exists
        ----------------------
        - Classes are first-class semantic entities in the graph. We need to:
          1) Create a ClassNode with metadata (decorators, docstring, etc).
          2) Connect it to the defining module via DEFINES.
          3) Capture inheritance targets to emit INHERITS edges and store bases for
             the later OVERRIDES pass.
          4) Discover and register methods (FunctionDef inside the class body).

        Parameters
        ----------
        node : ast.ClassDef
            The AST node representing a class definition.
        mod_qname : str
            The qualified name of the module that contains this class.
        context_stack : list
            A simple stack used to track parse context (e.g., inside class/function).
            It is updated here to ensure functions found in the class body are treated as methods.

        Behavior
        --------
        - Compute the class qualified name and record it in class_symbols (name → qname).
        - Create a ClassNode; append to self.nodes and a DEFINES edge from the module.
        - Emit INHERITS edges and persist class_bases[class_qname] = [base_targets].
        - Initialize per-class method table self.class_method_symbols[class_qname] = {}.
        - Iterate child FunctionDef to register methods via _handle_function().
        """
        qualified_name = f"{mod_qname}.{node.name}" # e.g., "myproj.src.MyClass"
        decorators = [ast.unparse(d) for d in node.decorator_list]  # Extract decorators as strings
        docstring = ast.get_docstring(node)             # Extract docstring if available    

        self.class_symbols[node.name] = qualified_name  # Register class in the symbol table

        class_node = ClassNode(
            name=node.name,
            qualified_name=qualified_name,
            decorators=decorators,
            start_line=node.lineno,
            end_line=node.end_lineno if hasattr(node, "end_lineno") else node.lineno,
            docstring=docstring,
            parent=mod_qname
        )

        self.nodes.append(class_node)
        self.edges.append(DefinesEdge(
            source=mod_qname,
            target=qualified_name,
            type="DEFINES"
        ))

        # Capture inheritance relationships and store bases for OVERRIDES phase
        self._handle_inherits(node, qualified_name)

        # Traverse class body to process methods
        context_stack.append("class")   # Enter class context so functions become methods
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                self._handle_function(child, qualified_name, context_stack)
        context_stack.pop()             # Exit class context
    
    def _handle_function(self, node: ast.FunctionDef, mod_or_class_qname: str, context_stack: list):
        """
        Register a function or method, emit DEF edges, collect calls, and recurse into nested defs.

        Why this method exists
        ----------------------
        - Functions and methods are primary semantic nodes. We need to:
          1) Decide whether the definition is a top-level/nested function or a class method.
          2) Create the corresponding node (FunctionNode or MethodNode) with metadata.
          3) Connect it to its parent via DEFINES or DEFINES_METHOD.
          4) Populate symbol tables used for call resolution and overrides.
          5) Walk the function body to emit CALLS edges and register nested functions.

        Parameters
        ----------
        node : ast.FunctionDef
            AST node for the function/method definition.
        mod_or_class_qname : str
            Qualified name of the parent container (module or class).
        context_stack : list
            Parser context stack. If the top is "class", the current def is a method.

        Behavior
        --------
        - Compute qualified_name and capture metadata: decorators, docstring, parameters, return type.
        - Instantiate FunctionNode or MethodNode depending on context.
        - Emit DEFINES or DEFINES_METHOD edge from parent to this node.
        - Update func_symbols or method_symbols and per-class method table for OVERRIDES detection.
        - Walk the body to:
          - Find ast.Call nodes → _handle_call().
          - Discover nested FunctionDef → recurse to register them.
        """
        qualified_name = f"{mod_or_class_qname}.{node.name}"    # e.g., "myproj.src.MyClass.my_method" or "myproj.src.my_function"
        decorators = [ast.unparse(d) for d in node.decorator_list]  # Extract decorators as strings
        docstring = ast.get_docstring(node) # Extract docstring if available
        params = [arg.arg for arg in node.args.args]    # Extract parameter names
        return_type = ast.unparse(node.returns) if node.returns else None   # Extract return type if specified

        is_method = context_stack and context_stack[-1] == "class"  # Inside class? Then it's a Method
        func_node = MethodNode if is_method else FunctionNode   # Node type discriminator

        func_obj = func_node(
            name=node.name,
            qualified_name=qualified_name,
            decorators=decorators,
            start_line=node.lineno,
            end_line=node.end_lineno if hasattr(node, "end_lineno") else node.lineno,
            docstring=docstring,
            is_anonymous=False,
            parent=mod_or_class_qname,
            signature=f"{node.name}({', '.join(params)})",
            parameters=params,
            return_type=return_type
        )
        self.nodes.append(func_obj)
        self.edges.append(DefinesEdge(
            source=mod_or_class_qname,
            target=qualified_name,
            type="DEFINES_METHOD" if is_method else "DEFINES"
        ))

        # Update symbol tables to enable call/override resolution
        if is_method:
            self.method_symbols[node.name] = qualified_name # Global map: method name → method qname
            # Record this method under its declaring class for OVERRIDES analysis
            if mod_or_class_qname in self.class_method_symbols:
                self.class_method_symbols[mod_or_class_qname][node.name] = qualified_name
        else:
            self.func_symbols[node.name] = qualified_name   # Global map: function name → function qname
        
        # Walk the function body to collect CALLS edges
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._handle_call(child, qualified_name, is_method) 

        # Recurse into nested function definitions (def inside def)
        context_stack.append("function")    # Enter function context for correct nesting semantics
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                self._handle_function(child, qualified_name, context_stack)
        context_stack.pop()                 # Exit function context
    
    def _handle_call(self, node: ast.Call, current_class_or_func_qname: str, is_method: bool):
        """
        Analyze a call expression and emit a CALLS edge with resolved callee type.

        Why this method exists
        ----------------------
        - Call sites form the backbone of dynamic relationships in the graph.
        - We attempt lightweight resolution to classify calls as:
          Function, Method, Constructor, Built-in, or External, and store
          the raw callee text for future refinement.

        Parameters
        ----------
        node : ast.Call
            The call expression to analyze.
        current_class_or_func_qname : str
            Qualified name of the current caller (function or method).
        is_method : bool
            True if the caller is a MethodNode, false if a FunctionNode.

        Behavior
        --------
        - Case 1: ast.Name
          - If name matches known function/method/class → resolve to qname and type.
          - If name is a Python builtin → mark as BUILTIN (target=None).
        - Case 2: ast.Attribute
          - If receiver is `self` → treat as method of the current class (best-effort).
          - Else if attribute name matches a known class → treat as constructor.
          - Else → mark as EXTERNAL and keep the raw dotted expression.
        - Emit a CallsEdge with caller_type/callee_type and callee_raw preserved.

        """
        callee_raw = ast.unparse(node.func) # Raw text of the callee (e.g., "requests.get")
        # Simple identifier call, e.g., foo()
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.func_symbols:                   # Known function in the project
                callee_qname = self.func_symbols[name]
                callee_type = NodeType.FUNCTION             
            elif name in self.method_symbols:               # Known function in the project
                callee_qname = self.method_symbols[name]
                callee_type = NodeType.METHOD
            elif name in self.class_symbols:                # Calling a class name → constructor call
                callee_qname = self.class_symbols[name]
                callee_type = NodeType.CONSTRUCTOR
            elif name in self.builtin_funcs:                # Built-in function (e.g., "len", "print")
                callee_qname = None 
                callee_type = NodeType.BUILTIN
            else:                                           # Unresolved/unknown
                callee_qname, callee_type = None, None
        # Qualified attribute call, e.g., obj.method() or pkg.fn()
        elif isinstance(node.func, ast.Attribute):
            # Special-case: self.method() inside a class → treat as method of current class
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                method_name = node.func.attr
                callee_qname = f"{current_class_or_func_qname}.{method_name}"
                callee_type = NodeType.METHOD
            else:
                # Fallbacks for attribute calls
                attr_name = node.func.attr
                if attr_name in self.class_symbols:
                    callee_qname = self.class_symbols[attr_name]
                    callee_type = NodeType.CONSTRUCTOR
                else:
                    # Treat as external call (e.g., requests.get)
                    callee_qname = ast.unparse(node.func)   # vd: "requests.get"
                    callee_type = NodeType.EXTERNAL
        else:
            callee_qname, callee_type = None, None
        
        # Emit the CALLS relationship with classified types
        self.edges.append(CallsEdge(
            source=current_class_or_func_qname,
            target=callee_qname,
            type="CALLS",
            caller_type=NodeType.METHOD if is_method else NodeType.FUNCTION,
            callee_type=callee_type,
            callee_raw=callee_raw
        ))
    
    def _handle_inherits(self, node: ast.ClassDef, class_qname: str):
        """
        Emit INHERITS edges for each base and persist bases for later OVERRIDES detection.

        Why this method exists
        ----------------------
        - Inheritance drives polymorphism. Capturing base classes enables:
          1) Upstream hierarchy queries (who inherits from X?).
          2) Downstream OVERRIDES detection after all methods are known.

        Parameters
        ----------
        node : ast.ClassDef
            The class AST containing base expressions in node.bases.
        class_qname : str
            Qualified name of the current class.

        Behavior
        --------
        - For each base:
          - If ast.Name and resolvable via self.class_symbols → link to internal class qname.
          - Else use ast.unparse(base) to keep the dotted raw representation (external/unresolved).
        - Append an InheritsEdge for each base.
        - Store the list of base targets in self.class_bases[class_qname].
        - Initialize an empty per-class method table (if not already present).
        """
        bases = []                                                  # Accumulator of resolved/normalized base targets
        for base in node.bases:
            if isinstance(base, ast.Name): 
                base_name = base.id
                if base_name in self.class_symbols:
                    target_qname = self.class_symbols[base_name]
                else:
                    target_qname = base_name
            elif isinstance(base, ast.Attribute):
                target_qname = ast.unparse(base)
            else:
                target_qname = ast.unparse(base)
            self.edges.append(InheritsEdge(
                source=class_qname,
                target=target_qname,
                type="INHERITS"
            ))
            bases.append(target_qname)
        self.class_bases[class_qname] = bases  # Lưu lại base class

        # Tạo bảng method cho class này
        self.class_method_symbols[class_qname] = {}

    def _handle_overrides(self):
        """
        Compute and emit OVERRIDES edges after all classes/methods have been registered.

        Why this method exists
        ----------------------
        - Method overriding is determined only when the full class hierarchy and
          per-class method tables are known. This post-processing phase walks the
          recorded class_bases and class_method_symbols to connect overriding methods
          to their base class counterparts.

        Behavior
        --------
        - For each class:
          - For each of its methods:
            - For each of its bases:
              - If the base is internal (i.e., present in class_method_symbols):
                - If the base defines a method with the same name:
                  → Emit OVERRIDES(child_method → base_method).
              - External bases (raw dotted strings) are skipped.
        """
        # Iterate each class and its bases
        for class_qname, bases in self.class_bases.items():
            # Iterate each method in the class
            method_map = self.class_method_symbols.get(class_qname, {})
            for method_name, method_qname in method_map.items():
                # Iterate each base class of this class
                for base in bases:
                    # If the base is internal, check its methods
                    base_method_map = self.class_method_symbols.get(base, {})
                    base_method_qname = base_method_map.get(method_name)
                    if base_method_qname:
                        # Có method trùng tên ở base class, tạo OVERRIDES edge
                        self.edges.append(OverridesEdge(
                            source=method_qname,
                            target=base_method_qname,
                            type="OVERRIDES"
                        ))
                    # Nếu base là external (tên raw string), skip
    
    def _record_external_use(self, mod_qname: str, import_name: str, is_relative: bool = False) -> None:
        """
        Decide whether `import_name` (e.g., "numpy", "PIL.Image") used by `mod_qname`
        is an *external* dependency and, if yes, emit a module-level
        DEPENDS_ON_EXTERNAL edge to the corresponding ExternalPackageNode.

        Parameters
        ----------
        mod_qname : str
            Qualified name of the current module (e.g., "proj.pkg.mod").
        import_name : str
            The (already resolved) import token. For ImportFrom, we pass the base
            module part (e.g., "requests.exceptions", "PIL") rather than the leaf symbol.
        is_relative : bool, keyword-only
            True if the originating ImportFrom was relative (node.level > 0).
            Relative imports are *always internal*, so they must be skipped here.

        Behavior
        --------
        1) If `is_relative` is True → skip (internal by definition).
        2) If top-level token or the entire `import_name` is recognized as *internal*
        (present in `self.module_symbols`) → skip.
        3) Otherwise map known module→package (PIL→Pillow, cv2→opencv-python, ...),
        normalize the package name (PEP 503), upsert ExternalPackageNode (spec = ""),
        and emit a single module-level DEPENDS_ON_EXTERNAL edge (deduplicated per module).

        Notes
        -----
        - This method is idempotent. Repeated calls for the same (module, package)
        will not create duplicate edges thanks to `self.mod_to_external_used`.
        """
        # 1) Relative imports are always internal
        if is_relative:
            return

        if not import_name:
            return

        # 2) Extract top-level token: "a.b.c" -> "a"
        top = import_name.split(".", 1)[0]

        # 3) Internal module discovered in Phase 2.2? -> skip
        if top in self.module_symbols or import_name in self.module_symbols:
            return

        # 4) Map to distribution name and normalize
        mapper = getattr(self, "WELL_KNOWN_IMPORT_TO_PKG", None)
        if mapper is None:
            # Fallback to a module-level constant if you keep it there.
            try:
                mapper = WELL_KNOWN_IMPORT_TO_PKG
            except NameError:
                mapper = {}
        raw_pkg = mapper.get(top, top)
        norm_pkg = self._norm_pkg_name(raw_pkg)

        # 5) Dedup per module
        used = self.mod_to_external_used.setdefault(mod_qname, set())
        if norm_pkg in used:
            return

        # 6) Ensure node exists (spec may be filled later by manifest)
        self._upsert_external_package(norm_pkg, "")

        # 7) Emit Module -> DEPENDS_ON_EXTERNAL
        self.edges.append(
            DependsOnExternalEdge(
                source=mod_qname,
                target=norm_pkg,
                type="DEPENDS_ON_EXTERNAL",
            )
        )
        used.add(norm_pkg)

    @staticmethod
    def _split_req_line(line: str) -> tuple[str, str] | None:
        """
        Parse a single requirement spec line into (package_name, version_spec).

        This function is intentionally robust for common requirement formats found in
        pyproject/requirements files. It supports:
          - Simple pins:        "requests==2.31.0"
          - Ranges:             "pandas>=2.1,<3"
          - No version:         "numpy"
          - Extras:             "uvicorn[standard]>=0.27"
          - Env markers:        "Pillow>=10 ; python_version >= '3.10'"
          - Inline comments:    "PyYAML>=6.0  # for config"
          - Editable/VCS lines: "-e git+https://...#egg=package"  (returns ("package",""))

        Returns:
            (name, spec) if a package name can be extracted, otherwise None for
            lines that are empty, comment-only, or unsupported.

        Notes:
            - The returned name is *raw* (not PEP 503 normalized). Normalization
              should be applied by the caller (e.g., inside _upsert_external_package).
            - The spec is returned exactly as it appears after the operator
              (e.g., ">=2.1,<3"). If no version is present, spec = "".
        """
        if not line:
            return None
        
        s = line.strip()
        if not s or s.startswith("#"):
            return None
        
        # Drop inline comments: "pkg>=1  # comment" -> "pkg>=1"
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if not s:
            return None
        
        # Drop environment markers: "pkg>=1 ; python_version >= '3.10'"
        if ";" in s:
            s = s.split(";", 1)[0].strip()
        if not s:
            return None
        
        # Handle editable/VCS lines (best-effort):
        #   -e git+...#egg=package
        #   git+...#egg=package
        editable_prefixes = ("-e ", "--editable ")
        if s.startswith(editable_prefixes) or s.startswith("git+"):
            m = re.search(r"[#&]egg=([A-Za-z0-9_.\-]+)", s)
            if m:
                return m.group(1), ""
            # If we cannot extract a name, skip this line silently
            return None
        
        # Strip extras: "uvicorn[standard]" -> "uvicorn"
        # We only drop the extras segment; the version part (if any) follows after.
        # We find the first version operator and split before it; extras are removed from the name part.
        # Supported operators (ordered by length so longer ones match first).
        operators = ("==", ">=", "<=", ">", "<", "~=", "!=")
        op_pos = len(s)
        op_used = None
        for op in operators:
            pos = s.find(op)
            if pos != -1 and pos < op_pos:
                op_pos, op_used = pos, op
        
        if op_used is None:
            # No version operator found: entire string is the package token (possibly with extras)
            name_part = s
            spec_part = ""

        else:
            name_part = s[:op_pos].rstrip()
            spec_part = s[op_pos:].strip()  # keep the operator and everything after
        
        # Remove extras from the name part: "pkg[foo,bar]" -> "pkg"
        if "[" in name_part:
            name_part = name_part.split("[", 1)[0].strip()
        
        # Empty name → skip
        if not name_part:
            return None

        return name_part, spec_part

    @staticmethod
    def _norm_pkg_name(name: str) -> str:
        """
        Normalize a distribution name per PEP 503:
          - lowercase
          - collapse runs of -, _, . into single '-'
        This guarantees that different spellings of the same distribution map to one key.
        """
        name = name.strip().lower()
        return re.sub(r"[-_.]+", "-", name)

    def _upsert_external_package(self, pkg_name: str, version_spec: str | None = "") -> ExternalPackageNode:
        """
        Create or update an ExternalPackageNode in the parser's state.

        This method ensures:
          1) We only ever keep one node per *normalized* package name.
          2) If a node already exists with empty version_spec and we now have a non-empty
             version_spec (e.g., discovered from pyproject.toml), we update the node in-place.
          3) The node is appended to `self.nodes` only on first creation (no duplicates).

        Args:
            pkg_name:     A raw distribution name as it appears in import/manifest (e.g., "Pillow", "cv2").
            version_spec: The version constraint string (e.g., ">=2.31,<3"). If unknown, pass "".

        Returns:
            The ExternalPackageNode instance representing this distribution.
        """

        # Normalize the name so that "PyYAML", "pyyaml", "yaml" (after mapping) merge consistently.
        norm = self._norm_pkg_name(pkg_name)
        node = self.external_packages.get(norm)

        if node is None:
            # First time we see this package → create the node and cache it.
            node = ExternalPackageNode(name=norm, version_spec=version_spec or "")
            self.external_packages[norm] = node
            self.nodes.append(node)
            return node

        # Node already exists: upgrade version_spec if previously unknown/empty.
        if not getattr(node, "version_spec", "") and version_spec:
            node.version_spec = version_spec

        return node

WELL_KNOWN_IMPORT_TO_PKG = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "dateutil": "python-dateutil",
    "bs4": "beautifulsoup4",
    "skimage": "scikit-image",
    "Crypto": "pycryptodome",
}

if __name__ == "__main__":
    parser = ASTParser("tests/sample_repo/")
    parser._parse_module("tests/sample_repo/tests/test_sample.py")
