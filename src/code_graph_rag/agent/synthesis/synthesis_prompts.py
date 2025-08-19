from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any
import json

from src.code_graph_rag.agent.models import Action, Language, QueryIntent
from src.code_graph_rag.agent.synthesis.synthesis_schemas import get_action_description, get_synthesis_schema

def build_synthesis_prompt(action: Action, language: Language) -> ChatPromptTemplate:
    """
    Build action-specific prompt template for synthesis.
    
    Each action requires different analysis approach:
    - list_callers/callees: relationship analysis
    - explain_function: code understanding 
    - trace_flow: path analysis
    
    Args:
        action: The type of analysis to perform
        language: Response language (vi/en)
        
    Returns:
        ChatPromptTemplate configured for the action
    """
    
    # Get action-specific instructions and schema
    action_desc = get_action_description(action)
    schema = get_synthesis_schema(action)
    
    # Base instructions cho tất cả actions
    base_instructions = _get_base_instructions(language)
    
    # Action-specific instructions
    action_instructions = _get_action_instructions(action, language)
    
    # Schema documentation cho LLM
    schema_doc = _format_schema_for_llm(schema, action_desc)
    
    # Build system message
    system_template = f"""{base_instructions}

{action_instructions}

{schema_doc}

CRITICAL RULES:
1. Output ONLY valid JSON - no explanations, no markdown, no code fences
2. Ground every claim in the provided evidence - cite only provided IDs
3. Use language: {language.value}
4. If evidence is insufficient, reduce confidence and add notes
5. Never include internal file paths or sensitive information
"""

    # User message template với placeholders
    user_template = """QUERY CONTEXT:
Action: {action}
Target: {mention}
Language: {language}

EVIDENCE:
{evidence_context}

ANALYSIS REQUEST:
{analysis_request}

Generate the synthesis JSON now:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def _get_base_instructions(language: Language) -> str:
    """
    Core instructions cho synthesis task, language-aware.
    
    These instructions apply to all actions và ensure:
    - JSON-only output (no prose)
    - Evidence grounding (no hallucination)
    - Proper citation format
    """
    
    if language == Language.vi:
        return """Bạn là một AI assistant chuyên phân tích mã nguồn.
Nhiệm vụ: Phân tích evidence được cung cấp và tạo ra kết quả có cấu trúc JSON.

NGUYÊN TẮC CỐT LÕI:
- Chỉ sử dụng thông tin từ evidence được cung cấp
- Không bịa đặt thông tin không có trong evidence
- Trả về JSON hợp lệ theo đúng schema yêu cầu
- Giải thích rõ ràng lý do cho mỗi kết luận"""
    
    else:  # English
        return """You are an AI assistant specialized in source code analysis.
Task: Analyze provided evidence and generate structured JSON results.

CORE PRINCIPLES:
- Use ONLY information from provided evidence
- Never hallucinate information not in evidence  
- Return valid JSON matching the required schema
- Explain reasoning for each conclusion clearly"""

def _get_action_instructions(action: Action, language: Language) -> str:
    """
    Action-specific analysis instructions.
    
    Each action requires different analysis approach:
    - Callers/callees: Focus on function relationships và call patterns
    - Explain: Focus on function purpose, inputs/outputs, side effects
    - Flow: Focus on execution paths và data flow
    """
    
    instructions = {
        Language.vi: {
            Action.list_callers: """
PHÂN TÍCH CALLERS:
- Xác định các function/method nào gọi target function
- Giải thích tại sao mỗi caller gọi target (mục đích gì)
- Đếm số lần gọi nếu có thông tin
- Ưu tiên theo tần suất hoặc tầm quan trọng""",

            Action.list_callees: """
PHÂN TÍCH CALLEES:  
- Xác định các function/method nào được target function gọi
- Giải thích tại sao target gọi mỗi callee (mục đích gì)
- Phân tích thứ tự gọi nếu có thể xác định
- Chú ý dependencies và side effects""",

            Action.explain_function: """
PHÂN TÍCH FUNCTION:
- Tóm tắt mục đích chính của function
- Liệt kê inputs (parameters) và ý nghĩa
- Xác định outputs (return values) 
- Phân tích side effects (file I/O, state changes, etc.)
- Liệt kê exceptions có thể xảy ra
- Đánh giá độ phức tạp thuật toán nếu có thể""",

            Action.imports: """
PHÂN TÍCH IMPORTS:
- Xác định module nào import module/symbol nào
- Giải thích lý do import (sử dụng để làm gì)
- Phân biệt internal vs external imports
- Chú ý circular dependencies nếu có""",

            Action.inherits_tree: """
PHÂN TÍCH INHERITANCE:
- Xác định class hierarchy và inheritance relationships
- Phân tích direct và indirect inheritance
- Giải thích lý do inheritance (tại sao inherit từ base class)
- Chú ý multiple inheritance và diamond problem
- Đánh giá tính hợp lý của inheritance structure""",

            Action.overrides: """
PHÂN TÍCH OVERRIDES:
- Xác định method nào override method nào trong hierarchy
- Giải thích lý do override (thêm functionality, thay đổi behavior)
- Phân tích có gọi super() hay không
- Chú ý breaking changes trong override
- Đánh giá tính consistent của override behavior""",

            Action.depends_external: """
PHÂN TÍCH EXTERNAL DEPENDENCIES:
- Xác định external packages được sử dụng
- Phân tích cách sử dụng (import style, usage patterns)
- Liệt kê version constraints và compatibility
- Chú ý potential security risks
- Đề xuất alternatives nếu có""",

            Action.trace_flow: """
PHÂN TÍCH FLOW:
- Tìm đường đi từ source đến destination
- Mô tả từng bước trong execution path
- Đánh giá confidence cho mỗi path
- Chú ý branches và alternative paths""",

            Action.impact_analysis: """
PHÂN TÍCH IMPACT:
- Xác định components nào bị ảnh hưởng khi thay đổi target
- Phân tích direct và indirect impacts
- Đánh giá risk level cho mỗi impact
- Đề xuất testing strategy"""
        },
        
        Language.en: {
            Action.list_callers: """
CALLER ANALYSIS:
- Identify which functions/methods call the target function
- Explain why each caller invokes the target (purpose)
- Count call frequency if information available
- Prioritize by frequency or importance""",

            Action.list_callees: """
CALLEE ANALYSIS:
- Identify which functions/methods are called by the target
- Explain why target calls each callee (purpose)
- Analyze call order if determinable
- Note dependencies and side effects""",

            Action.explain_function: """
FUNCTION ANALYSIS:
- Summarize the main purpose of the function
- List inputs (parameters) and their meaning
- Identify outputs (return values)
- Analyze side effects (file I/O, state changes, etc.)
- List possible exceptions
- Assess algorithmic complexity if possible""",

            Action.imports: """
IMPORT ANALYSIS:
- Identify which module imports which module/symbol
- Explain import rationale (used for what purpose)
- Distinguish internal vs external imports
- Note circular dependencies if present""",

            Action.inherits_tree: """
INHERITANCE ANALYSIS:
- Identify class hierarchy and inheritance relationships
- Analyze direct and indirect inheritance
- Explain inheritance rationale (why inherit from base class)
- Note multiple inheritance and diamond problems
- Assess inheritance structure reasonableness""",

            Action.overrides: """
OVERRIDE ANALYSIS:
- Identify which methods override which in hierarchy
- Explain override rationale (add functionality, change behavior)
- Analyze whether super() is called
- Note breaking changes in overrides
- Assess override behavior consistency""",

            Action.depends_external: """
EXTERNAL DEPENDENCY ANALYSIS:
- Identify external packages being used
- Analyze usage patterns (import style, usage patterns)
- List version constraints and compatibility
- Note potential security risks
- Suggest alternatives if available""",

            Action.trace_flow: """
FLOW ANALYSIS:
- Find execution paths from source to destination
- Describe each step in the execution path
- Assess confidence for each path
- Note branches and alternative paths""",

            Action.impact_analysis: """
IMPACT ANALYSIS:
- Identify components affected by target changes
- Analyze direct and indirect impacts
- Assess risk level for each impact
- Suggest testing strategy"""
        }
    }
    
    return instructions[language].get(action, f"Analyze {action.value} relationships in the code.")

def _format_schema_for_llm(schema: Dict[str, Any], action_desc: Dict[str, Any]) -> str:
    """
    Format JSON schema thành human-readable documentation cho LLM.
    
    LLM cần hiểu:
    - Required fields và data types
    - Expected structure cho items array
    - Example của valid output
    
    Args:
        schema: JSON schema dict
        action_desc: Action description với examples
        
    Returns:
        Formatted schema documentation
    """
    
    # Extract key schema information
    required_fields = schema.get("required", [])
    items_schema = schema.get("properties", {}).get("items", {})

    # Get example và escape braces
    example_item = action_desc.get('example_item', {})
    example_json = json.dumps(example_item, indent=2, ensure_ascii=False)
    
    # Escape ALL curly braces in JSON example
    escaped_example = example_json.replace("{", "{{").replace("}", "}}")

    action_value = action_desc.get('action_value', 'list_callers')
    
    # Build readable format
    schema_doc = f"""
JSON SCHEMA REQUIREMENTS:

Required fields: {', '.join(required_fields)}

Expected structure:
{{{{
  "action": "{action_desc.get('action_value', '')}",
  "language": "vi" or "en",
  "answer": "Brief summary of findings",
  "items": [
    // {action_desc.get('items_description', 'Array of analysis results')}
  ],
  "citations": [
    // References to evidence used: {{{{"label": "Function", "id": "qualified_name"}}}}
  ],
  "confidence": 0.0-1.0,  // Overall confidence in results
  "evidence_count": number  // Count of evidence items analyzed
}}}}

EXAMPLE ITEM:
{escaped_example}
"""
    
    return schema_doc

def get_analysis_request(action: Action, intent: QueryIntent) -> str:
    """
    Generate specific analysis request based on action và intent.
    
    Converts user intent thành specific instructions cho LLM.
    Includes target mentions, constraints, và expected focus areas.
    """
    
    mention = intent.mention or "unknown"
    mention_dst = intent.mention_dst
    
    if action in (Action.list_callers, Action.list_callees):
        direction = "callers of" if action == Action.list_callers else "functions called by"
        return f"Analyze {direction} `{mention}`. Include call frequency and purpose for each relationship."
    
    elif action == Action.explain_function:
        return f"Provide comprehensive analysis of function `{mention}` including purpose, inputs/outputs, side effects, and complexity."
    
    elif action == Action.trace_flow:
        if mention_dst:
            return f"Trace execution flow from `{mention}` to `{mention_dst}`. Show all possible paths and decision points."
        else:
            return f"Analyze execution flow starting from `{mention}`. Show downstream call chains."
    
    elif action == Action.impact_analysis:
        return f"Analyze what would be impacted if `{mention}` were modified. Include direct and indirect effects."
    
    elif action == Action.imports:
        return f"Analyze import relationships for `{mention}`. Show what it imports and what imports it."
    
    else:
        return f"Perform {action.value} analysis on `{mention}`."

# Template cache để avoid rebuilding
_TEMPLATE_CACHE: Dict[tuple[Action, Language], ChatPromptTemplate] = {}

def get_synthesis_prompt(action: Action, language: Language) -> ChatPromptTemplate:
    """
    Get cached prompt template cho action/language combination.
    
    Caching improves performance và ensures consistency.
    Templates are immutable nên safe to cache.
    """
    
    cache_key = (action, language)
    if cache_key not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[cache_key] = build_synthesis_prompt(action, language)
    
    return _TEMPLATE_CACHE[cache_key]