import os
import ast
from pathlib import Path
from typing import Any
import builtins

from src.code_graph_rag.models.nodes import *
from src.code_graph_rag.models.edges import *

class ASTParser():
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.nodes = []
        self.edges = []
        # Maps folder paths to their qualified names when they are Python packages
        # Key: folder path (str) - relative path from project root (e.g., "src/models")  
        # Value: qualified name (str) - dot-separated identifier (e.g., "myproject.src.models")
        # This mapping is essential for establishing correct parent-child relationships
        # because packages use qualified_name as identifier while folders use path
        self.folder_qname_map = {}

        self.func_symbols = {}   # name → qualified_name
        self.method_symbols = {} # name → qualified_name
        self.class_symbols = {}
        self.builtin_funcs = set(dir(builtins))
    
    def parse(self) -> tuple[list, list]:
        """Run the full parse and return lists of node and edge objects."""
        self._walk_files_and_dirs()
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

    def _parse_module(self, module_path: Path, mod_qname: str):
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()

        try: 
            tree = ast.parse(source)
        except SyntaxError:
            return
        
        # Context stack: track current ClassDef to distinguish function vs method
        context_stack = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._handle_class(node, mod_qname, context_stack)
            elif isinstance(node, ast.FunctionDef):
                self._handle_function(node, mod_qname, context_stack)
    
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
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "self":
                    method_name = node.func.attr
                    # Trong context class, tìm method
                    callee_qname = f"{current_class_or_func_qname}.{method_name}"
                    callee_type = NodeType.METHOD
                else:   # case function -> class.method (process_data -> Calculator.add)
                    attr_name = node.func.attr  # tên phía sau dấu chấm
                    if attr_name in self.class_symbols:
                        # NẾU tên class phía sau thuộc local class (rất hiếm khi dùng import kiểu này), vẫn gán là constructor
                        callee_qname = self.class_symbols[attr_name]
                        callee_type = NodeType.CONSTRUCTOR
                    else:
                        # Không chắc chắn là constructor, gán là None
                        callee_qname, callee_type = None, None

            else:
                # Chưa xử lý (phase sau)
                callee_qname, callee_type = None, None
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
        

if __name__ == "__main__":
    parser = ASTParser("tests/sample_repo/")
    parser._parse_module("tests/sample_repo/tests/test_sample.py")
