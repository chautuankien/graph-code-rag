import json
from langchain.prompts import ChatPromptTemplate

from src.code_graph_rag.agent.models import (
    QueryIntent, Action,
    PlanExecutionResult, SynthesisOutput
)
from src.code_graph_rag.agent.llm import run_llm_json

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

        
def synthesize(intent: QueryIntent, results: list[PlanExecutionResult], provider:str) -> SynthesisOutput:
    """Main synthesis method - keep it simple"""
    try:
        # 1. Prepare evidence context (simple string concatenation)
        evidence_text = _prepare_evidence(results)
        
        # 2. Get action-specific prompt
        prompt = _get_prompt(intent.action)
        
        # 3. Call LLM
        response_json = _call_llm(prompt, evidence_text, intent, provider)
        
        # 4. Return structured output
        return SynthesisOutput(
            action=intent.action.value,
            answer=response_json.get("answer", "Analysis completed"),
            items=response_json.get("items", []),
            evidence_count=len(results),
            confidence=response_json.get("confidence", 0.8)
        )
        
    except Exception as e:
        log.error(f"Synthesis failed: {e}")
        # Simple fallback
        return SynthesisOutput(
            action=intent.action.value,
            answer=f"Analysis failed: {str(e)}",
            items=[],
            evidence_count=len(results),
            confidence=0.1
        )

def _prepare_evidence(results: list[PlanExecutionResult]) -> str:
    if not results:
        return "No evidence available."
    
    evidence_parts = []
    for i, result in enumerate(results[:10]):  # Limit to first 10 items
        evidence_parts.append(f"--- EVIDENCE {i+1} ---")
        evidence_parts.append(f"Source: {result.step}")
        evidence_parts.append(f"Label: {result.label or 'Unknown'}")
        evidence_parts.append(f"ID: {result.id or 'Unknown'}")
        
        if result.name:
            evidence_parts.append(f"Name: {result.name}")
        if result.docstring:
            evidence_parts.append(f"Docstring: {result.docstring}")
        if result.snippet:
            evidence_parts.append(f"Code:\n{result.snippet}")
        
        evidence_parts.append("")  # Empty line separator
    
    return "\n".join(evidence_parts)

def _get_prompt(action: Action) -> str:
    """Simple action-specific prompts"""
    base_instructions = """
You are analyzing code relationships. Return ONLY valid JSON with this structure:
{{
    "action": "action_name",
    "answer": "Human-readable summary of findings",
    "items": [...], 
    "confidence": 0.8,
    "evidence_count": 1
}}

Rules:
- Base your analysis ONLY on the provided evidence
- Don't hallucinate or make assumptions
- Keep explanations concise and technical
"""
    
    action_prompts = {
            Action.list_callers: base_instructions + """
For callers analysis, each item should be:
{{
    "caller": "function/method name that calls the target",
    "callee": "target function being called", 
    "location": "file location if available",
    "why": "brief explanation of the call relationship"
}}
""",
            Action.list_callees: base_instructions + """
For callees analysis, each item should be:
{{
    "caller": "source function",
    "callee": "function/method being called",
    "location": "file location if available", 
    "why": "brief explanation of what this call does"
}}
""",
            Action.explain_function: base_instructions + """
For function explanation, use this structure:
{{
    "answer": "What this function does and its purpose",
    "items": [{{
        "summary": "One-line description",
        "purpose": "Why this function exists",
        "inputs": ["parameter descriptions"],
        "outputs": "return value description",
        "logic": "key steps or algorithm"
    }}],
    "confidence": 0.9
}}
""",
            Action.imports: base_instructions + """
For imports analysis, each item should be:
{{
    "source": "module doing the import",
    "target": "imported module/symbol", 
    "import_type": "module|symbol|package",
    "why": "purpose of this import"
}}
""",
            Action.inherits_tree: base_instructions + """
For inheritance analysis, each item should be:
{{
    "child": "class that inherits",
    "parent": "base class",
    "inheritance_type": "direct|indirect",
    "why": "reason for inheritance"
}}
""",
            Action.overrides: base_instructions + """
For override analysis, each item should be:
{{
    "child_method": "overriding method",
    "parent_method": "method being overridden",
    "class": "class where override happens",
    "why": "reason for override"
}}
""",
            Action.depends_external: base_instructions + """
For external dependencies analysis, each item should be:
{{
    "source": "module using external package",
    "external_package": "external package name",
    "usage_type": "import|call|inherit",
    "why": "purpose of dependency"
}}
""",
            Action.trace_flow: base_instructions + """
For flow tracing analysis, each item should be:
{{
    "path_id": "path_1",
    "source": "starting function",
    "target": "destination function",
    "steps": ["step1", "step2", "step3"],
    "confidence": 0.9,
    "description": "description of this path"
}}
""",
            Action.impact_analysis: base_instructions + """
For impact analysis, each item should be:
{{
    "affected_component": "component that would be impacted",
    "impact_type": "direct|indirect", 
    "risk_level": "low|medium|high",
    "why": "reason for impact"
}}
"""
    }
    
    return action_prompts.get(action, base_instructions)

def _call_llm(prompt: str, evidence_text: str, intent: QueryIntent, provider: str) -> dict:
    """Call LLM and return parsed JSON"""
    # Create proper prompt template
    prompt_template = ChatPromptTemplate.from_messages([
            ("user", f"""
{prompt}

EVIDENCE:
{{evidence_text}}

TARGET: {{target}}

Generate the JSON now:
""")
        ])
        
    # Prepare payload
    payload = {
            "evidence_text": evidence_text,
            "target": intent.mention or "unknown"
        }
        
        
    try:
        # Use correct run_llm_json API from llm.py
        response = run_llm_json(
            prompt=prompt_template,
            payload=payload,
            schema=SynthesisOutput,
            provider=provider,
            temperature=0.0,
            max_tokens=1500,
            max_retries=2
        )
        
        # Convert Pydantic model to dict
        return response.model_dump()
            
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return {
            "answer": f"LLM call failed: {str(e)}",
            "items": [],
            "confidence": 0.1
        }