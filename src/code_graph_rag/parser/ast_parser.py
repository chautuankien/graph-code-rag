import ast
from pathlib import Path
from typing import Any

class ASTParser():
    def __init__(self, module_path: Path, module_id: str):
        self.module_path = module_path
        self.module_id = module_id
        self.nodes = []
        self.edges = []
        self.source = module_path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source, filename=str(module_path))
    
    def parse(self) -> dict[str, list[dict[str, Any]]]:
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.FunctionDef):
                self._handle_function(node, parent_class=None)
            elif isinstance(node, ast.ClassDef):
                self._handle_class(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._handle_import(node)
    
        return {
            "nodes": self.nodes,
            "edges": self.edges,
        }
    
    def _handle_function(self, node: ast.FunctionDef, parent_class: str):
        name = node.name
        node_type = "Method" if parent_class else "Function"
        func_id = f"{node_type.lower()}: {self.module_id}.{name}"

        self.nodes.append({
            "id": func_id,
            "type": node_type,
            "name": name,
            "defined_in": self.module_id,
            "lineno": node.lineno,
            "docstring": ast.get_docstring(node),
        })

        # Edge: module/class DEFINES function/method
        if parent_class:
            self.edges.append({
                "from": f"class:{self.module_id}.{parent_class}",
                "to": func_id,
                "type": "DEFINES_METHOD"
            })
        else:
            self.edges.append({
                "from": f"module:{self.module_id}",
                "to": func_id,
                "type": "DEFINES"
            })
        
        # CALLS inside this function
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if call_name:
                    self.edges.append({
                        "from": func_id,
                        "to": f"function_or_method:{call_name}",
                        "type": "CALLS"
                    })
    
    def _handle_class(self, node: ast.ClassDef):
        class_name = node.name
        class_id = f"class:{self.module_id}.{class_name}"
        self.nodes.append({
            "id": class_id,
            "type": "Class",
            "name": class_name,
            "defined_in": self.module_id,
            "lineno": node.lineno,
            "docstring": ast.get_docstring(node),
        })

        self.edges.append({
            "from": f"module:{self.module_id}",
            "to": class_id,
            "type": "DEFINES"
        })

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                self._handle_function(child, parent_class=class_name)

    def _handle_import(self, node: ast.Import | ast.ImportFrom):
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.nodes.append({
                    "id": f"external:{alias.name}",
                    "type": "ExternalPackage",
                    "name": alias.name
                })
                self.edges.append({
                    "from": f"module:{self.module_id}",
                    "to": f"external:{alias.name}",
                    "type": "DEPENDS_ON_EXTERNAL"
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or "unknown"
            self.nodes.append({
                "id": f"external:{module}",
                "type": "ExternalPackage",
                "name": module
            })
            self.edges.append({
                "from": f"module:{self.module_id}",
                "to": f"external:{module}",
                "type": "DEPENDS_ON_EXTERNAL"
            })

    def _get_call_name(self, node: ast.Call) -> str | None:
        # Trích tên hàm gọi ra dưới dạng chuỗi
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None