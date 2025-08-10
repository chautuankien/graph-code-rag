import os
import ast
import re
from pathlib import Path
import builtins
import tomllib
import glob

from src.code_graph_rag.models.nodes import *
from src.code_graph_rag.models.edges import *

class ASTParser():
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.project_name = self.project_root.name  # Name of the project root directory

        self.nodes = []
        self.edges = []
        # Maps folder paths to their qualified names when they are Python packages
        # Key: folder path (str) - relative path from project root (e.g., "src/models")  
        # Value: qualified name (str) - dot-separated identifier (e.g., "myproject.src.models")
        # This mapping is essential for establishing correct parent-child relationships
        # because packages use qualified_name as identifier while folders use path
        self.folder_qname_map = {}

        self.module_symbols = {}
        self.import_map = {}
        self.func_symbols = {}   # name → qualified_name
        self.method_symbols = {} # name → qualified_name
        self.class_symbols = {}
        self.builtin_funcs = set(dir(builtins))

        self.class_method_symbols = {}  # {class_qname: {method_name: method_qualified_name}}
        self.class_bases = {}          # {class_qname: [base_class_target]}

        # External packages cache: normalized_name -> ExternalPackageNode
        self.external_packages: dict[str, ExternalPackageNode] = {}
    
    def parse(self) -> tuple[list, list]:
        """Run the full parse and return lists of node and edge objects."""
        self._walk_files_and_dirs()
        self._parse_dependencies_from_manifest()
        self._handle_overrides()
        return self.nodes, self.edges
    
    def _walk_files_and_dirs(self):
        """
        Recursively walk the project directory to build a comprehensive graph structure.
        
        This method performs a complete traversal of the project filesystem and:
        1. Detects structural elements: folders, packages (directories with __init__.py), 
            Python modules (.py files), and regular files
        2. Creates appropriate node objects for each discovered element
        3. Establishes CONTAINS_* relationships following the containment hierarchy
        4. Maintains consistent identifier mapping (name/qualified_name/path) across node types
        
        The traversal follows these containment rules:
        - Project (identified by name) can contain: Packages, Folders, Modules, Files
        - Package (identified by qualified_name) can contain: Packages, Folders, Modules, Files  
        - Folder (identified by path) can contain: Packages, Folders, Modules, Files
        
        Qualified names follow the pattern: project_name.folder1.folder2.module_name
        """
        # Create the root project node using the project directory name
        project_node = ProjectNode(name=self.project_root.name)
        self.nodes.append(project_node)

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
            # Include project name in qualified name to ensure uniqueness across projects
            current_qname = ".".join([self.project_root.name] + list(rel_path.parts))

            def get_parent_source():
                """
                Determine the parent container and its type for establishing CONTAINS_* relationships.
                
                Returns:
                    tuple: (parent_identifier, parent_type) where:
                    - parent_identifier: The unique identifier for the parent node
                    - parent_type: The node type ("Project", "Package", or "Folder")
                """
                parent_path = rel_path.parent
                
                # If no parent parts, this is a direct child of the project root
                if not parent_path.parts:
                    return self.project_root.name, "Project"
                else:
                    parent_path_str = str(parent_path)
                    # Check if parent directory was registered as a package
                    if parent_path_str in self.folder_qname_map:
                        # Parent is a package - use its qualified name as identifier
                        return self.folder_qname_map[parent_path_str], "Package"
                    else:
                        # Parent is a regular folder - use its path as identifier
                        return parent_path_str, "Folder"

            # Process non-root directories (skip the project root itself)
            if not current_is_root:
                # Get parent information for establishing containment relationships
                parent_source, parent_type = get_parent_source()
                
                if is_package:
                    # Create a Package node for directories containing __init__.py
                    pkg_node = PackageNode(
                        qualified_name=current_qname,  # Unique qualified name
                        name=current_path.name,        # Simple directory name
                        path=current_path_str          # Relative path from project root
                    )
                    self.nodes.append(pkg_node)
                    
                    # Register this package in the mapping for future parent lookups
                    self.folder_qname_map[current_path_str] = current_qname
                    
                    # Create CONTAINS_PACKAGE relationship from parent to this package
                    self.edges.append(ContainsEdge(
                        source=parent_source,     # Parent identifier (varies by parent type)
                        target=current_qname,     # Package uses qualified_name as identifier
                        type="CONTAINS_PACKAGE"
                    ))
                else:
                    # Create a Folder node for regular directories
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

            # Process all files in the current directory
            for file in filenames:
                file_path = current_path / file
                rel_file_path = file_path.relative_to(self.project_root)
                ext = file_path.suffix

                # Determine the container that will "own" this file
                if current_is_root:
                    # File is directly in project root
                    container_source = self.project_root.name
                else:
                    if is_package:
                        # File is in a package directory - use package qualified_name
                        container_source = current_qname
                    else:
                        # File is in a regular folder - use folder path
                        container_source = current_path_str

                if ext == ".py":
                    # Create Module node for Python source files
                    # Include project name in qualified name for global uniqueness
                    mod_qname = ".".join([self.project_root.name] + list(rel_file_path.with_suffix('').parts))
                    
                    self.module_symbols[mod_qname] = mod_qname

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

                    self._parse_module(file_path, mod_qname)
                else:
                    # Create File node for non-Python files
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
                    if full_name in self.module_symbols:
                        target_qname = full_name     # Internal module already known by its qualified name
                    else:
                        target_qname = full_name     # External/unknown for now — keep raw dotted path
                    # record alias → target for later resolution
                    self.import_map[alias_name] = target_qname  # Map imported symbol/alias to its resolved target
                    # emit edge
                    self.edges.append(ImportsEdge(
                        source=mod_qname,            # The module performing the import
                        target=target_qname,         # The module/symbol being imported (internal or raw)
                        type="IMPORTS",              # Relationship label
                        import_name=full_name        # Textual module path as written in code
                    ))

            # ----- Case B: `from pkg.mod import name [as alias]` -----
            elif isinstance(node, ast.ImportFrom):  # Handle `from ... import ...` forms (supports relative imports)
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

                    # resolve target qname
                    if import_name in self.module_symbols:
                        target_qname = target_base   # Internal module/symbol already known
                    else:
                        target_qname = target_base   # External/unknown — keep normalized dotted path
                    # update import_map
                    self.import_map[alias_name] = target_qname  # Map alias/name → resolved target
                    # emit edge
                    self.edges.append(ImportsEdge(
                        source=mod_qname,           # The module performing the import
                        target=target_qname,        # The resolved module/symbol name
                        type="IMPORTS",             # Relationship label
                        import_name=import_name     # Normalized textual import path
                    ))

    def _handle_class(self, node: ast.ClassDef, mod_qname: str, context_stack: list):
        qualified_name = f"{mod_qname}.{node.name}"
        decorators = [ast.unparse(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)

        self.class_symbols[node.name] = qualified_name

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

        # Detect inherit edge
        self._handle_inherits(node, qualified_name)

        # Detect method in a class
        context_stack.append("class")
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                self._handle_function(child, qualified_name, context_stack)
        context_stack.pop()
    
    def _handle_function(self, node: ast.FunctionDef, mod_or_class_qname: str, context_stack: list):
        qualified_name = f"{mod_or_class_qname}.{node.name}"
        decorators = [ast.unparse(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)
        params = [arg.arg for arg in node.args.args]
        return_type = ast.unparse(node.returns) if node.returns else None

        is_method = context_stack and context_stack[-1] == "class"
        func_node = MethodNode if is_method else FunctionNode

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

        if is_method:
            self.method_symbols[node.name] = qualified_name
            # Ghi lại method cho class này
            if mod_or_class_qname in self.class_method_symbols:
                self.class_method_symbols[mod_or_class_qname][node.name] = qualified_name
        else:
            self.func_symbols[node.name] = qualified_name
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._handle_call(child, qualified_name, is_method) 

        # Detect sub-function in a nested function
        context_stack.append("function")
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                self._handle_function(child, qualified_name, context_stack)
        context_stack.pop()
    
    def _handle_call(self, node: ast.Call, current_class_or_func_qname: str, is_method: bool):
        callee_raw = ast.unparse(node.func)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.func_symbols:
                callee_qname = self.func_symbols[name]
                callee_type = NodeType.FUNCTION
            elif name in self.method_symbols:
                callee_qname = self.method_symbols[name]
                callee_type = NodeType.METHOD
            elif name in self.class_symbols:
                callee_qname = self.class_symbols[name]
                callee_type = NodeType.CONSTRUCTOR
            elif name in self.builtin_funcs:
                callee_qname = None 
                callee_type = NodeType.BUILTIN
            else:
                callee_qname, callee_type = None, None
        # Case 2: gọi tới method qua self (self.method())
        elif isinstance(node.func, ast.Attribute):
            # Check self.method()
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                method_name = node.func.attr
                callee_qname = f"{current_class_or_func_qname}.{method_name}"
                callee_type = NodeType.METHOD
            else:
                # Không resolve được (không phải self.method), check external
                attr_name = node.func.attr
                # Nếu không phải class local
                if attr_name in self.class_symbols:
                    callee_qname = self.class_symbols[attr_name]
                    callee_type = NodeType.CONSTRUCTOR
                else:
                    # External call
                    callee_qname = ast.unparse(node.func)   # vd: "requests.get"
                    callee_type = NodeType.EXTERNAL
        else:
            callee_qname, callee_type = None, None
        
                    # Tạo CallsEdge
        self.edges.append(CallsEdge(
            source=current_class_or_func_qname,
            target=callee_qname,
            type="CALLS",
            caller_type=NodeType.METHOD if is_method else NodeType.FUNCTION,
            callee_type=callee_type,
            callee_raw=callee_raw
        ))
    
    def _handle_inherits(self, node: ast.ClassDef, class_qname: str):
        # Detect inherit edge
        bases = []
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
        Sau khi đã parse xong toàn bộ class/method, phát hiện các OVERRIDES edge.
        """
        # Duyệt từng class
        for class_qname, bases in self.class_bases.items():
            # Duyệt từng method trong class đó
            method_map = self.class_method_symbols.get(class_qname, {})
            for method_name, method_qname in method_map.items():
                # Duyệt từng base class của class này
                for base in bases:
                    # Nếu base là class internal, kiểm tra method trong đó
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


if __name__ == "__main__":
    parser = ASTParser("tests/sample_repo/")
    parser._parse_module("tests/sample_repo/tests/test_sample.py")
