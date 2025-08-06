from pathlib import Path
import networkx as nx
from src.code_graph_rag.utils.file_utils import walk_codebase
from src.code_graph_rag.parser.ast_parser import ASTParser

class CodeGraphBuilder:
    """
    Class that builds a knowledge graph from a Python codebase using AST parsing.
    """
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.graph = nx.DiGraph()
        self.structure = walk_codebase(repo_path)
    
    def build(self) -> nx.DiGraph:
        """
        Build a directed graph (DiGraph) that contains the structure and logic of the codebase.

        Returns:
            nx.DiGraph: The constructed knowledge graph
        """
        # Add structural nodes: folders, files, modules, packages
        for node_type in ["folders", "files", "packages", "modules"]:
            for node in self.structure[node_type]:
                self._add_node(node)
        
        # Add CONTAINS_* edges between folders/files/modules
        self._add_structure_edges()
        
        # Parse all modules and add their AST nodes/edges
        for module in self.structure["modules"]:
            module_path = Path(module["path"])
            parser = ASTParser(module_path, module_id=module["id"].split(":")[1])
            result = parser.parse()

            for node in result["nodes"]:
                self._add_node(node)

            for edge in result["edges"]:
                self._add_edge(edge)

            # Optional: link file node to module node
            self._add_edge({
                "from": f"file:{module_path.relative_to(self.repo_path)}",
                "to": module["id"],
                "type": "IS_MODULE"
            })

        return self.graph

    def _add_node(self, node: dict):
        """
        Add a node to the graph with its metadata.
        """
        self.graph.add_node(node["id"], **node)
    
    def _add_edge(self, edge: dict):
        """
        Add an edge to the graph with its type as label.
        """
        self.graph.add_edge(edge["from"], edge["to"], type=edge["type"])
    
    def _add_structure_edges(self):
        """
        Add structural edges like CONTAINS_FOLDER, CONTAINS_FILE, CONTAINS_MODULE based on filesystem layout.
        """
        for node in self.structure["files"]:
            parent = Path(node["path"]).parent.relative_to(self.repo_path)
            self._add_edge({
                "from": f"folder:{parent}" if parent != Path(".") else "project: root",
                "to": node["id"],
                "type": "CONTAINS_FILE"
            })
        
        for node in self.structure["modules"]:
            parent = Path(node["path"]).parent.relative_to(self.repo_path)
            self._add_edge({
                "from": f"folder:{parent}" if parent != Path('.') else "project:root",
                "to": node["id"],
                "type": "CONTAINS_MODULE"
            })
        
        for node in self.structure["folders"]:
            parent = Path(node["path"]).parent.relative_to(self.repo_path)
            if parent != Path('.'):
                self._add_edge({
                    "from": f"folder:{parent}",
                    "to": node["id"],
                    "type": "CONTAINS_FOLDER"
                })
            else:
                self._add_edge({
                    "from": "project:root",
                    "to": node["id"],
                    "type": "CONTAINS_FOLDER"
                })
            
        for node in self.structure["packages"]:
            self._add_edge({
                "from": "project:root",
                "to": node["id"],
                "type": "CONTAINS_PACKAGE"
            })