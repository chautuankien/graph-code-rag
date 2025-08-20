from .base import BaseEdge
from typing import Literal
from enum import Enum

### CONTAINS EDGE ###
class ContainsEdge(BaseEdge):
    """Represents a structural containment relationship (project → folder, folder → file, etc)."""
    type: Literal[
        "CONTAINS_PACKAGE", "CONTAINS_FOLDER",
        "CONTAINS_FILE", "CONTAINS_MODULE"
    ]

### DEFINED EDGE ###
class DefinesEdge(BaseEdge):
    """Represents that a module or class defines another entity (class, method, etc)."""
    type: Literal["DEFINES", "DEFINES_METHOD"]

### INHERIT EDGE ###
class InheritsEdge(BaseEdge):
    """Represents class inheritance (A inherits from B)."""
    type: Literal["INHERITS"]

### OVERRIDE EDGE ###
class OverridesEdge(BaseEdge):
    """Represents that a method overrides a method from superclass."""
    type: Literal["OVERRIDES"]

### CALLS EDGE ###
class NodeType(str, Enum):
    FUNCTION = "Function"
    METHOD = "Method"
    CONSTRUCTOR = "Constructor"
    BUILTIN = "BuiltIn"
    EXTERNAL = "External"

class CallsEdge(BaseEdge):
    """Represents a function/method calling another function/method."""
    target: str | None
    type: Literal["CALLS"]
    caller_type: NodeType | None
    callee_type: NodeType | None
    callee_raw: str

### DEPENDSON EDGE ###
class DependsOnExternalEdge(BaseEdge):
    """Represents dependency on an external package (from pyproject/requirements)."""
    type: Literal["DEPENDS_ON_EXTERNAL"]

### IMPORT EDGE ###
class ImportsEdge(BaseEdge):
    """Represents an import statement linking two modules."""
    type: Literal["IMPORTS"]
    import_name: str