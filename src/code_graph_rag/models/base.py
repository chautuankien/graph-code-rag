from pydantic import BaseModel

class BaseNode(BaseModel):
    name: str

class BaseEdge(BaseModel):
    source: str  # qualified_name
    target: str  # qualified_name
    type: str