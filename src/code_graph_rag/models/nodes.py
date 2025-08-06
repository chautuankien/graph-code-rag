from .base import BaseNode

class ProjectNode(BaseNode):
    """Represents the root of the codebase/project."""
    name: str

class PackageNode(BaseNode):
    """A Python package (directory with __init__.py)."""
    qualified_name: str
    path: str

class FolderNode(BaseNode):
    """A folder in the file system, may or may not be a Python package."""
    path: str

class FileNode(BaseNode):
    """Any non-Python file in the project."""
    path: str
    extension: str

class ModuleNode(BaseNode):
    """A Python source file (.py)."""
    qualified_name: str
    path: str

class ClassNode(BaseNode):
    """Represents a class definition."""
    qualified_name: str
    decorators: list[str]
    start_line: int
    end_line: int
    docstring: str | None = None
    parent: str | None = None

class FunctionNode(BaseNode):
    """Represents a standalone or nested function."""
    qualified_name: str
    decorators: list[str]
    start_line: int
    end_line: int
    docstring: str | None = None
    is_anonymous: bool
    parent: str | None = None
    signature: str | None = None
    parameters: list[str]
    return_type: str | None = None

class MethodNode(FunctionNode):
    """Represents a method within a class."""
    parent: str  # class name

class ExternalPackageNode(BaseNode):
    """Represents a third-party dependency parsed from requirements or pyproject.toml."""
    version_spec: str
