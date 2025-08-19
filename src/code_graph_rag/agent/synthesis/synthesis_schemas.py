from typing import Dict, Any
from src.code_graph_rag.agent.models import Action

# Base entity reference schema
ENTITY_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "id": {"type": "string"},
        "name": {"type": "string"}
    },
    "required": ["label", "id", "name"],
    "additionalProperties": False
}

# Citation schema
CITATION_SCHEMA = {
    "type": "object", 
    "properties": {
        "label": {"type": "string"},
        "id": {"type": "string"},
        "snippet_lines": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2
        }
    },
    "required": ["label", "id"],
    "additionalProperties": False
}

# Base synthesis output schema
BASE_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "language": {"type": "string", "enum": ["vi", "en"]},
        "answer": {"type": "string"},
        "items": {"type": "array"},
        "citations": {
            "type": "array",
            "items": CITATION_SCHEMA
        },
        "notes": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_count": {"type": "integer", "minimum": 0}
    },
    "required": ["action", "language", "answer", "items", "citations"],
    "additionalProperties": False
}

# Action-specific item schemas
CALLER_CALLEE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "caller": ENTITY_REF_SCHEMA,
        "callee": ENTITY_REF_SCHEMA, 
        "why": {"type": "string"},
        "call_count": {"type": "integer", "minimum": 1}
    },
    "required": ["caller", "callee", "why"],
    "additionalProperties": False
}

RELATIONSHIP_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "source": ENTITY_REF_SCHEMA,
        "relationship": {"type": "string"},
        "target": ENTITY_REF_SCHEMA,
        "why": {"type": "string"},
        "details": {"type": "object"}
    },
    "required": ["source", "relationship", "target", "why"],
    "additionalProperties": False
}

EXPLAIN_FUNCTION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "purpose": {"type": "string"},
        "inputs": {
            "type": "array",
            "items": {"type": "string"}
        },
        "outputs": {"type": "string"},
        "side_effects": {
            "type": "array", 
            "items": {"type": "string"}
        },
        "exceptions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "complexity_hint": {"type": "string"},
        "snippet_excerpt": {"type": "string"},
        "dependencies": {
            "type": "array",
            "items": ENTITY_REF_SCHEMA
        }
    },
    "required": ["summary", "purpose", "inputs"],
    "additionalProperties": False
}

FLOW_STEP_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "step_id": {"type": "string"},
        "entity": ENTITY_REF_SCHEMA,
        "operation": {"type": "string"},
        "why": {"type": "string"},
        "order": {"type": "integer", "minimum": 0}
    },
    "required": ["step_id", "entity", "operation", "why", "order"],
    "additionalProperties": False
}

FLOW_PATH_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "path_id": {"type": "string"},
        "steps": {
            "type": "array",
            "items": FLOW_STEP_ITEM_SCHEMA
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "description": {"type": "string"}
    },
    "required": ["path_id", "steps", "confidence", "description"],
    "additionalProperties": False
}

# Mapping từ Action → complete schema
def get_synthesis_schema(action: Action) -> Dict[str, Any]:
    """Get complete JSON schema for a specific action"""
    
    # Copy base schema
    schema = dict(BASE_SYNTHESIS_SCHEMA)
    
    # Set action constraint
    schema["properties"]["action"] = {"const": action.value}
    
    # Set items schema based on action
    if action in (Action.list_callers, Action.list_callees):
        schema["properties"]["items"] = {
            "type": "array",
            "items": CALLER_CALLEE_ITEM_SCHEMA,
            "maxItems": 200  # safety limit
        }
    
    elif action in (Action.imports, Action.inherits_tree, Action.overrides, Action.depends_external):
        schema["properties"]["items"] = {
            "type": "array", 
            "items": RELATIONSHIP_ITEM_SCHEMA,
            "maxItems": 200
        }
    
    elif action == Action.explain_function:
        schema["properties"]["items"] = {
            "type": "array",
            "items": EXPLAIN_FUNCTION_ITEM_SCHEMA,
            "maxItems": 5  # usually just 1 function explanation
        }
    
    elif action in (Action.trace_flow, Action.impact_analysis):
        schema["properties"]["items"] = {
            "type": "array",
            "items": FLOW_PATH_ITEM_SCHEMA,
            "maxItems": 10  # multiple paths possible
        }
    
    else:
        # Fallback for unknown actions
        schema["properties"]["items"] = {
            "type": "array",
            "items": {"type": "object"},  # flexible
            "maxItems": 100
        }
    
    return schema

# Validation helpers
def validate_synthesis_output(data: dict, action: Action) -> tuple[bool, str | None]:
    """Validate synthesis output against schema"""
    import jsonschema
    
    try:
        schema = get_synthesis_schema(action)
        jsonschema.validate(data, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)

# Schema descriptions for LLM prompts
ACTION_DESCRIPTIONS = {
    Action.list_callers: {
        "action_value": "list_callers",
        "purpose": "List functions/methods that call the target function",
        "items_description": "Array of caller-callee relationships with explanations",
        "example_item": {
            "caller": {"label": "Function", "id": "proj.app.main", "name": "main"},
            "callee": {"label": "Function", "id": "proj.utils.helper", "name": "helper"}, 
            "why": "main() calls helper() to process data",
            "call_count": 1
        }
    },
    
    Action.list_callees: {
        "action_value": "list_callees",
        "purpose": "List functions/methods called by the target function",
        "items_description": "Array of caller-callee relationships with explanations",
        "example_item": {
            "caller": {"label": "Function", "id": "proj.app.main", "name": "main"},
            "callee": {"label": "Function", "id": "proj.utils.helper", "name": "helper"},
            "why": "main() calls helper() to process data"
        }
    },
    
    Action.imports: {
        "action_value": "imports",
        "purpose": "List import relationships for the target module",
        "items_description": "Array of import relationships",
        "example_item": {
            "source": {"label": "Module", "id": "proj.app", "name": "app"},
            "relationship": "IMPORTS",
            "target": {"label": "Module", "id": "proj.utils", "name": "utils"},
            "why": "app module imports utils for helper functions"
        }
    },

    Action.inherits_tree: {
        "action_value": "inherits_tree",
        "purpose": "Show class inheritance hierarchy and relationships",
        "items_description": "Array of inheritance relationships showing class hierarchy",
        "example_item": {
            "source": {"label": "Class", "id": "proj.models.User", "name": "User"},
            "relationship": "INHERITS",
            "target": {"label": "Class", "id": "proj.models.BaseModel", "name": "BaseModel"},
            "why": "User class inherits from BaseModel to get common functionality",
            "details": {"inheritance_level": 1, "is_direct": True}
        }
    },

    Action.overrides: {
        "action_value": "overrides",
        "purpose": "Find method override relationships in inheritance hierarchy",
        "items_description": "Array of method override relationships",
        "example_item": {
            "source": {"label": "Method", "id": "proj.models.User.save", "name": "save"},
            "relationship": "OVERRIDES", 
            "target": {"label": "Method", "id": "proj.models.BaseModel.save", "name": "save"},
            "why": "User.save() overrides BaseModel.save() to add validation",
            "details": {"override_type": "method", "adds_functionality": True}
        }
    },

    Action.depends_external: {
        "action_value": "depends_external",
        "purpose": "Show external package dependencies and usage",
        "items_description": "Array of external dependency relationships",
        "example_item": {
            "source": {"label": "Module", "id": "proj.utils.http", "name": "http"},
            "relationship": "DEPENDS_ON_EXTERNAL",
            "target": {"label": "ExternalPackage", "id": "requests", "name": "requests"},
            "why": "http module uses requests library for HTTP operations",
            "details": {"usage_type": "import", "version_spec": ">=2.31,<3"}
        }
    },
    
    Action.explain_function: {
        "action_value": "explain_function",
        "purpose": "Explain what a function does, its inputs/outputs, side effects",
        "items_description": "Single function explanation with detailed analysis",
        "example_item": {
            "summary": "Processes user input and returns formatted result",
            "purpose": "Main entry point for data processing pipeline",
            "inputs": ["user_data: dict", "options: Optional[dict]"],
            "outputs": "Processed data as formatted string",
            "side_effects": ["Logs processing steps", "Updates global state"],
            "exceptions": ["ValueError: if user_data is invalid"],
            "complexity_hint": "O(n) where n is size of user_data"
        }
    },
    
    Action.trace_flow: {
        "action_value": "trace_flow",
        "purpose": "Trace execution flow from source to destination",
        "items_description": "Array of possible execution paths",
        "example_item": {
            "path_id": "path_1",
            "steps": [
                {"step_id": "1", "entity": {"label": "Function", "id": "proj.app.main", "name": "main"}, "operation": "calls", "why": "Entry point", "order": 0},
                {"step_id": "2", "entity": {"label": "Function", "id": "proj.utils.helper", "name": "helper"}, "operation": "processes", "why": "Data processing", "order": 1}
            ],
            "confidence": 0.9,
            "description": "Main execution path through data processing"
        }
    },

    Action.impact_analysis: {
        "action_value": "impact_analysis",
        "purpose": "Analyze potential impact of changes to the target component",
        "items_description": "Array of impact paths showing affected components",
        "example_item": {
            "path_id": "impact_1",
            "steps": [
                {"step_id": "1", "entity": {"label": "Function", "id": "proj.utils.helper", "name": "helper"}, "operation": "modified", "why": "Direct change target", "order": 0},
                {"step_id": "2", "entity": {"label": "Function", "id": "proj.app.main", "name": "main"}, "operation": "affected", "why": "Calls the modified function", "order": 1},
                {"step_id": "3", "entity": {"label": "Function", "id": "proj.api.handler", "name": "handler"}, "operation": "affected", "why": "Indirectly affected through main()", "order": 2}
            ],
            "confidence": 0.8,
            "description": "Impact chain from helper modification to API handlers",
            "risk_level": "medium"
        }
    }
}

def get_action_description(action: Action) -> dict:
    """Get description and example for an action"""
    return ACTION_DESCRIPTIONS.get(action, {
        "purpose": f"Handle {action.value} query",
        "items_description": "Array of relevant items", 
        "example_item": {"placeholder": "action-specific structure"}
    })