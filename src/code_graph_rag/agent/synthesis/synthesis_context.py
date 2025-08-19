from typing import List, Dict, Any, Tuple
import re
from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.code_graph_rag.agent.models import PlanExecutionResult, QueryIntent, Action
from src.code_graph_rag.utils.logging_setup import get_logger

log = get_logger(__name__)

class EvidenceChunk(BaseModel):
    """
    Single piece of evidence với metadata cho context preparation.
    
    Represents one piece of evidence (function, class, etc.) với:
    - Content (snippet, docstring)  
    - Relevance scoring
    - Token cost estimation
    """
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For PlanExecutionResult
    )

    result: PlanExecutionResult          # Original result
    content: str                         # Prepared content for LLM
    relevance_score: float = Field(      # 0.0-1.0, higher = more relevant
        ge=0.0, le=1.0, 
        description="Relevance score based on query intent"
    )
    token_estimate: int = Field(         # Rough token count
        ge=0,
        description="Estimated token cost for this chunk"
    )
    priority: int = Field(               # 1=highest, used for sorting
        ge=1, le=10,
        description="Priority level (1=highest priority)"
    )
    
    # Optional computed fields
    @computed_field
    @property
    def entity_id(self) -> str:
        """Quick access to entity ID"""
        return self.result.id or "unknown"
    
    @computed_field  
    @property
    def entity_type(self) -> str:
        """Quick access to entity type"""
        return self.result.label or "Unknown"
    
    @computed_field
    @property
    def has_code_snippet(self) -> bool:
        """Check if this chunk includes code snippet"""
        return bool(self.result.snippet)
    
    @computed_field
    @property
    def quality_score(self) -> float:
        """Combined quality score based on available metadata"""
        score = 0.0
        if self.result.snippet:
            score += 0.4  # Has code
        if self.result.docstring:
            score += 0.3  # Has documentation  
        if self.result.signature:
            score += 0.2  # Has signature
        if self.result.path and self.result.start_line:
            score += 0.1  # Has location info
        return min(score, 1.0)
        
    def __str__(self) -> str:
        """Human-readable representation"""
        return f"EvidenceChunk({self.entity_type}:{self.entity_id}, relevance={self.relevance_score:.2f}, tokens={self.token_estimate})"
    
    def __repr__(self) -> str:
        return self.__str__()

class ContextPreparationConfig(BaseModel):
    """Configuration for evidence context preparation"""
    
    max_tokens: int = Field(3000, ge=500, le=8000)
    max_snippet_lines: int = Field(20, ge=5, le=100) 
    max_items: int = Field(50, ge=1, le=200)
    quality_threshold: float = Field(0.1, ge=0.0, le=1.0)
    relevance_threshold: float = Field(0.0, ge=0.0, le=1.0)
    
    # Token estimation settings
    chars_per_token: int = Field(4, ge=2, le=6)  # Rough estimation
    context_overhead_tokens: int = Field(200, ge=50, le=500)  # Headers, formatting
    
class ContextPreparationResult(BaseModel):
    """Result of context preparation with metadata"""
    
    context: str = Field(description="Formatted evidence context for LLM")
    chunks_selected: list[EvidenceChunk] = Field(description="Selected evidence chunks")
    chunks_total: int = Field(description="Total chunks available")
    tokens_used: int = Field(description="Estimated tokens used")
    tokens_available: int = Field(description="Token budget")
    quality_stats: dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def selection_ratio(self) -> float:
        """Ratio of selected vs total chunks"""
        return len(self.chunks_selected) / max(self.chunks_total, 1)
    
    @computed_field
    @property
    def token_utilization(self) -> float:
        """Ratio of tokens used vs available"""
        return self.tokens_used / max(self.tokens_available, 1)

class ContextPreparationError(Exception):
    """Raised when context preparation fails"""
    pass

def prepare_evidence_context(
    validated_results: List[PlanExecutionResult],
    intent: QueryIntent,
    config: ContextPreparationConfig | None = None
) -> ContextPreparationResult:
    """
    Convert validated results thành LLM context string, optimized for token budget.
    
    Now returns rich ContextPreparationResult với full metadata.
    
    Args:
        validated_results: Cleaned evidence from validation phase
        intent: User query intent (for relevance scoring)
        config: Configuration object (uses defaults if None)
        
    Returns:
        ContextPreparationResult with context and metadata
        
    Raises:
        ContextPreparationError: If no valid evidence can be prepared
    """
    
    # Use default config if none provided
    if config is None:
        config = ContextPreparationConfig()
    
    if not validated_results:
        log.warning("prepare_evidence_context: No validated results provided")
        return ContextPreparationResult(
            context="No evidence available for analysis.",
            chunks_selected=[],
            chunks_total=0,
            tokens_used=0,
            tokens_available=config.max_tokens
        )
    
    log.debug(f"prepare_evidence_context: Processing {len(validated_results)} results")
    
    # Step 1: Convert results to evidence chunks với relevance scoring
    chunks = []
    for result in validated_results:
        try:
            chunk = _create_evidence_chunk(result, intent, config)
            if chunk and chunk.relevance_score >= config.relevance_threshold:
                chunks.append(chunk)
        except Exception as e:
            log.warning(f"Failed to create chunk for {result.id}: {e}")
            continue
    
    if not chunks:
        raise ContextPreparationError("No valid evidence chunks could be created")
    
    # Step 2: Sort by relevance và priority  
    chunks.sort(key=lambda c: (-c.priority, -c.relevance_score, c.token_estimate))
    
    # Step 3: Select chunks within token budget
    selected_chunks = _select_chunks_by_budget(chunks, config)
    
    log.debug(f"prepare_evidence_context: Selected {len(selected_chunks)}/{len(chunks)} chunks")
    
    # Step 4: Format thành structured context
    context = _format_evidence_context(selected_chunks, intent)
    
    # Step 5: Calculate metadata
    total_tokens = sum(c.token_estimate for c in selected_chunks) + config.context_overhead_tokens
    quality_stats = _calculate_quality_stats(selected_chunks)
    
    return ContextPreparationResult(
        context=context,
        chunks_selected=selected_chunks,
        chunks_total=len(chunks),
        tokens_used=total_tokens,
        tokens_available=config.max_tokens,
        quality_stats=quality_stats
    )

def _create_evidence_chunk(
    result: PlanExecutionResult, 
    intent: QueryIntent,
    config: ContextPreparationConfig
) -> EvidenceChunk | None:
    """Convert single PlanExecutionResult thành EvidenceChunk using Pydantic validation"""
    
    if not (result.label and result.id):
        return None
    
    # Compute relevance score based on intent và result
    relevance_score = _compute_relevance_score(result, intent)
    
    # Prepare content string (same logic as before)
    content = _prepare_content_string(result, config.max_snippet_lines)
    
    # Estimate token count
    token_estimate = len(content) // config.chars_per_token
    
    # Determine priority
    priority = _compute_priority(result, intent)
    
    try:
        # Use Pydantic validation
        return EvidenceChunk(
            result=result,
            content=content,
            relevance_score=relevance_score,
            token_estimate=token_estimate,
            priority=priority
        )
    except Exception as e:
        log.warning(f"Failed to create EvidenceChunk: {e}")
        return None

def _compute_relevance_score(result: PlanExecutionResult, intent: QueryIntent) -> float:
    """
    Compute relevance score (0.0-1.0) cho evidence item.
    
    Factors:
    - Match với target mention
    - Evidence type relevance to action
    - Quality indicators (has snippet, docstring)
    """
    
    score = 0.0
    
    # Base score from ID match
    mention = intent.mention or ""
    if mention and result.id:
        if mention == result.id:
            score += 0.5  # Exact match
        elif mention in result.id or result.id.endswith(f".{mention}"):
            score += 0.3  # Partial match
        elif result.name and mention == result.name:
            score += 0.4  # Name match
    
    # Boost for evidence types relevant to action
    action_relevance = {
        Action.list_callers: {"Function": 0.3, "Method": 0.3},
        Action.list_callees: {"Function": 0.3, "Method": 0.3},
        Action.explain_function: {"Function": 0.4, "Method": 0.4},
        Action.imports: {"Module": 0.4, "ExternalPackage": 0.2},
    }
    
    relevance_bonus = action_relevance.get(intent.action, {}).get(result.label, 0.1)
    score += relevance_bonus
    
    # Quality bonuses
    if result.snippet:
        score += 0.2  # Has code snippet
    if result.docstring:
        score += 0.1  # Has documentation
    if result.signature:
        score += 0.1  # Has function signature
    
    return min(score, 1.0)  # Cap at 1.0

def _compute_priority(result: PlanExecutionResult, intent: QueryIntent) -> int:
    """
    Compute priority level (1=highest) cho evidence ordering.
    
    Priority determines order when relevance scores are equal.
    Evidence types critical for analysis get higher priority.
    """
    
    # Priority 1: Direct target analysis (META step results)
    if result.step == "META":
        return 1
    
    # Priority 2: Core relationships (CALLERS, CALLEES)
    if result.step in ("CALLERS_TOP", "CALLEES_TOP"):
        return 2
    
    # Priority 3: Code discovery (ENTRY_FUNCS_BY_KEYWORD)
    if result.step == "ENTRY_FUNCS_BY_KEYWORD":
        return 3
        
    # Priority 4: Structural relationships (IMPORTS, INHERITS)
    if result.step in ("IMPORTS", "INHERITS_DIRECT", "OVERRIDDEN_BY"):
        return 4
    
    # Priority 5: Context/neighborhood 
    if result.step in ("NEIGHBORHOOD", "METHODS_OF_CLASS"):
        return 5
    
    # Priority 6: Everything else
    return 6

def _select_chunks_by_budget(
    chunks: List[EvidenceChunk],
    config: ContextPreparationConfig
) -> List[EvidenceChunk]:
    """Select chunks using Pydantic config object"""
    
    selected = []
    total_tokens = config.context_overhead_tokens  # Start with overhead
    
    for chunk in chunks:
        # Check limits
        if len(selected) >= config.max_items:
            break
            
        # Check token budget (always include first chunk if possible)
        if total_tokens + chunk.token_estimate > config.max_tokens and selected:
            break
            
        selected.append(chunk)
        total_tokens += chunk.token_estimate
        
        log.debug(f"Selected chunk: {chunk.entity_id} ({chunk.token_estimate} tokens)")
    
    log.debug(f"_select_chunks_by_budget: {total_tokens}/{config.max_tokens} tokens used")
    
    return selected

def _format_evidence_context(
    chunks: List[EvidenceChunk],
    intent: QueryIntent
) -> str:
    """
    Format selected chunks thành structured context cho LLM.
    
    Creates clear, parseable context với:
    - Header với query information
    - Numbered evidence items
    - Clear separation between items
    """
    
    if not chunks:
        return "No evidence available."
    
    context_parts = []
    
    # Header với query context
    context_parts.append("=== EVIDENCE FOR ANALYSIS ===")
    context_parts.append(f"Query target: {intent.mention or 'N/A'}")
    context_parts.append(f"Analysis type: {intent.action.value}")
    context_parts.append(f"Evidence items: {len(chunks)}")
    context_parts.append("")
    
    # Format each evidence item
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"--- EVIDENCE {i} ---")
        context_parts.append(chunk.content)
        context_parts.append("")  # Blank line separator
    
    context_parts.append("=== END EVIDENCE ===")
    
    return "\n".join(context_parts)

def _calculate_quality_stats(chunks: List[EvidenceChunk]) -> Dict[str, float]:
    """Calculate quality statistics for selected chunks"""
    
    if not chunks:
        return {}
    
    stats = {
        "avg_relevance": sum(c.relevance_score for c in chunks) / len(chunks),
        "avg_quality": sum(c.quality_score for c in chunks) / len(chunks),
        "has_code_ratio": sum(1 for c in chunks if c.has_code_snippet) / len(chunks),
        "avg_tokens_per_chunk": sum(c.token_estimate for c in chunks) / len(chunks),
        "priority_distribution": {
            f"priority_{p}": sum(1 for c in chunks if c.priority == p) / len(chunks)
            for p in range(1, 7)
        }
    }
    
    return stats

def _prepare_content_string(result: PlanExecutionResult, max_snippet_lines: int) -> str:
    """
    Prepare formatted content string for a single evidence item.
    
    Creates structured, LLM-readable representation với:
    - Basic entity information (ENTITY, NAME)
    - Metadata (DOCSTRING, SIGNATURE, LOCATION)
    - Code snippet (CODE block, trimmed if needed)
    - Source information (SOURCE step)
    
    Args:
        result: Single evidence result from plan execution
        max_snippet_lines: Maximum lines to include in code snippet
        
    Returns:
        Formatted content string ready for LLM context
    """
    
    content_parts = []
    
    # 1. Basic entity information (always present)
    content_parts.append(f"ENTITY: {result.label} `{result.id}`")
    
    # 2. Add display name if different from ID
    if result.name and result.name != result.id:
        # For qualified names like "proj.app.main", name might be just "main"
        content_parts.append(f"NAME: {result.name}")
    
    # 3. Add docstring if available (important for understanding)
    if result.docstring:
        trimmed_doc = _trim_text(result.docstring, max_length=200)
        content_parts.append(f"DOCSTRING: {trimmed_doc}")
    
    # 4. Add function/method signature if available
    if result.signature:
        content_parts.append(f"SIGNATURE: {result.signature}")
    
    # 5. Add location information (file path + line numbers)
    if result.path:
        location = result.path
        if result.start_line and result.end_line:
            location += f":{result.start_line}-{result.end_line}"
        content_parts.append(f"LOCATION: {location}")
    
    # 6. Add code snippet if available (most valuable part)
    if result.snippet:
        trimmed_snippet = _trim_code_snippet(result.snippet, max_snippet_lines)
        content_parts.append(f"CODE:\n```python\n{trimmed_snippet}\n```")
    
    # 7. Add step context (which adapter generated this evidence)
    content_parts.append(f"SOURCE: {result.step}")
    
    # 8. Add extra fields if present (adapter-specific data)
    if hasattr(result, 'extra') and result.extra:
        extra_info = _format_extra_fields(result.extra)
        if extra_info:
            content_parts.append(f"EXTRA: {extra_info}")
    
    return "\n".join(content_parts)

def _trim_code_snippet(snippet: str, max_lines: int) -> str:
    """
    Trim code snippet to max_lines, preserving important parts.
    
    Strategy:
    - Keep function/class signature (first few lines)
    - Keep some body content  
    - Add ellipsis if truncated
    - Preserve indentation and structure
    
    Args:
        snippet: Raw code snippet
        max_lines: Maximum lines to keep
        
    Returns:
        Trimmed snippet with ellipsis if needed
    """
    
    if not snippet or not snippet.strip():
        return ""
    
    lines = snippet.splitlines()
    
    if len(lines) <= max_lines:
        return snippet
    
    # Special handling for functions/classes/methods
    first_line = lines[0].strip() if lines else ""
    is_function_or_class = any(first_line.startswith(keyword) for keyword in [
        'def ', 'async def ', 'class ', '@'  # Include decorators
    ])
    
    if is_function_or_class:
        # For functions/classes: keep signature + some body + ellipsis
        result_lines = []
        
        # 1. Keep decorators and signature (first few lines until body starts)
        signature_end = 1
        for i, line in enumerate(lines):
            if i == 0 or line.strip().startswith('@') or line.strip().endswith(':'):
                signature_end = i + 1
            else:
                break
        
        # Include signature lines
        result_lines.extend(lines[:signature_end])
        
        # 2. Add some body content if space allows
        remaining_lines = max_lines - signature_end - 1  # -1 for ellipsis
        if remaining_lines > 0 and signature_end < len(lines):
            body_lines = lines[signature_end:signature_end + remaining_lines]
            result_lines.extend(body_lines)
        
        # 3. Add ellipsis with proper indentation
        if signature_end + remaining_lines < len(lines):
            # Try to match indentation of last body line
            last_line = result_lines[-1] if result_lines else ""
            indent = len(last_line) - len(last_line.lstrip())
            ellipsis = " " * indent + "# ... (truncated)"
            result_lines.append(ellipsis)
        
        return "\n".join(result_lines)
    
    else:
        # General case: keep first N-1 lines + ellipsis
        result_lines = lines[:max_lines - 1]
        result_lines.append("# ... (truncated)")
        return "\n".join(result_lines)

def _trim_text(text: str, max_length: int) -> str:
    """
    Trim text to max_length characters, adding ellipsis if needed.
    
    Tries to break at word boundaries để avoid cutting words.
    
    Args:
        text: Text to trim
        max_length: Maximum characters to keep
        
    Returns:
        Trimmed text with ellipsis if needed
    """
    
    if not text or len(text) <= max_length:
        return text
    
    # Reserve space for ellipsis
    target_length = max_length - 3
    
    # Try to break at word boundary
    truncated = text[:target_length]
    last_space = truncated.rfind(' ')
    last_newline = truncated.rfind('\n')
    
    # Use the latest boundary (space or newline)
    break_point = max(last_space, last_newline)
    
    # Only break at boundary if it's not too early (keep at least 70% of target)
    if break_point > target_length * 0.7:
        truncated = truncated[:break_point]
    
    return truncated + "..."

def _format_extra_fields(extra: Dict[str, Any]) -> str:
    """
    Format extra fields from PlanExecutionResult.extra into readable string.
    
    Handles common extra field types:
    - call_count, weight, hops (numeric metrics)
    - relationship types, edge types
    - version specs, import details
    
    Args:
        extra: Dictionary of extra fields
        
    Returns:
        Formatted string or empty string if no useful extra data
    """
    
    if not extra:
        return ""
    
    formatted_parts = []
    
    # Handle common extra field types
    for key, value in extra.items():
        if value is None:
            continue
            
        # Format based on key type
        if key in ('call_count', 'weight', 'hops', 'frequency'):
            formatted_parts.append(f"{key}={value}")
        elif key in ('relationship_type', 'edge_type', 'import_type'):
            formatted_parts.append(f"{key}={value}")
        elif key == 'version_spec' and value:
            formatted_parts.append(f"version={value}")
        elif key == 'confidence' and isinstance(value, (int, float)):
            formatted_parts.append(f"confidence={value:.2f}")
        elif isinstance(value, (str, int, float, bool)):
            # Simple types: include as key=value
            formatted_parts.append(f"{key}={value}")
        elif isinstance(value, list) and len(value) <= 3:
            # Short lists: include as comma-separated
            formatted_parts.append(f"{key}=[{', '.join(map(str, value))}]")
        else:
            # Complex types: just indicate presence
            formatted_parts.append(f"{key}=<{type(value).__name__}>")
    
    return ", ".join(formatted_parts) if formatted_parts else ""

# Utility functions for token estimation and validation
def estimate_context_tokens(context: str) -> int:
    """Rough token estimation for context string."""
    # Simple heuristic: 4 characters ≈ 1 token for English/Vietnamese
    return len(context) // 4

def validate_evidence_context(context: str, max_tokens: int) -> Tuple[bool, str]:
    """Validate prepared context meets requirements."""
    
    if not context.strip():
        return False, "Empty context"
    
    estimated_tokens = estimate_context_tokens(context)
    if estimated_tokens > max_tokens * 1.1:  # 10% tolerance
        return False, f"Context too large: {estimated_tokens} > {max_tokens} tokens"
    
    if "=== EVIDENCE FOR ANALYSIS ===" not in context:
        return False, "Missing evidence header"
    
    return True, "Valid"