from .base import BaseEdge
from typing import Literal

class ContainsEdge(BaseEdge):
    """Represents a structural containment relationship (project → folder, folder → file, etc)."""
    type: Literal[
        "CONTAINS_PACKAGE", "CONTAINS_FOLDER",
        "CONTAINS_FILE", "CONTAINS_MODULE"
    ]

class DefinesEdge(BaseEdge):
    """Represents that a module or class defines another entity (class, method, etc)."""
    type: Literal["DEFINES", "DEFINES_METHOD"]

class InheritsEdge(BaseEdge):
    """Represents class inheritance (A inherits from B)."""
    type: Literal["INHERITS"]

class OverridesEdge(BaseEdge):
    """Represents that a method overrides a method from superclass."""
    type: Literal["OVERRIDES"]

class CallsEdge(BaseEdge):
    """Represents a function/method calling another function/method."""
    type: Literal["CALLS"]
    lineno: int | None = None
    caller: str | None = None
    callee: str | None = None

class DependsOnExternalEdge(BaseEdge):
    """Represents dependency on an external package (from pyproject/requirements)."""
    type: Literal["DEPENDS_ON_EXTERNAL"]

class ImportsEdge(BaseEdge):
    """Represents an import statement linking two modules."""
    type: Literal["IMPORTS"]
    import_name: str