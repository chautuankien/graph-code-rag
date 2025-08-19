# src/code_graph_rag/agent/synthesis.py

from typing import Dict, Any, Optional
import json
import time
from pydantic import ValidationError

from langchain.prompts import ChatPromptTemplate

from src.code_graph_rag.agent.models import (
    QueryIntent, PlanExecutionResult, SynthesisOutput, SynthesisError, Action, Language
)
from src.code_graph_rag.agent.synthesis.synthesis_prompts import get_synthesis_prompt, get_analysis_request
from src.code_graph_rag.agent.synthesis.synthesis_schemas import get_synthesis_schema, validate_synthesis_output
from src.code_graph_rag.agent.synthesis.synthesis_context import (
    prepare_evidence_context, ContextPreparationConfig, ContextPreparationResult
)
from src.code_graph_rag.agent.llm import run_llm_json, LLMJsonError
from src.code_graph_rag.utils.logging_setup import get_logger

log = get_logger(__name__)

class SynthesisEngine:
    """
    Core synthesis engine that orchestrates evidence → structured JSON conversion.
    
    Responsibilities:
    1. Coordinate all synthesis components (prompts, context, schemas, LLM)
    2. Handle errors và retry logic gracefully
    3. Ensure deterministic output (same input → same output)
    4. Provide rich debugging information
    5. Abstract complexity from GraphAgent
    """
    
    def __init__(
        self,
        provider: str = "nvidia",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        seed: int = 42,
        context_config: Optional[ContextPreparationConfig] = None
    ):
        """
        Initialize synthesis engine với LLM và context settings.
        
        Args:
            provider: LLM provider ("nvidia" or "openai")
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Max tokens for LLM response
            seed: Random seed for reproducibility
            context_config: Evidence context preparation config
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        
        # Use default context config if none provided
        self.context_config = context_config or ContextPreparationConfig()
        
        # Track engine stats for monitoring
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "repair_attempts": 0,
            "avg_processing_time_ms": 0.0
        }
    
    def synthesize(
        self,
        intent: QueryIntent,
        validated_results: list[PlanExecutionResult]
    ) -> SynthesisOutput:
        """
        Main synthesis method: convert evidence → structured JSON.
        
        This is the primary interface used by GraphAgent. Handles full pipeline:
        1. Evidence context preparation
        2. Prompt template selection
        3. LLM invocation với structured output
        4. JSON parsing với repair attempts
        5. Content sanitization
        
        Args:
            intent: User query intent (action, language, mentions)
            validated_results: Clean evidence from validation phase
            
        Returns:
            SynthesisOutput: Structured result với items và citations
            
        Raises:
            SynthesisError: If synthesis fails after all retries
        """
        start_time = time.time()
        self.stats["total_calls"] += 1
        
        try:
            log.info(f"synthesize: Starting synthesis for action={intent.action.value}, evidence_count={len(validated_results)}")
            
            # Step 1: Prepare evidence context (token-optimized)
            context_result = self._prepare_context(validated_results, intent)
            log.debug(f"synthesize: Context prepared - {context_result.tokens_used}/{context_result.tokens_available} tokens")
            log.debug("synthesize.context:\n %s", context_result.context[:100] + "...")  # Truncate for logging

            # Step 2: Get schema và prompt template
            schema = get_synthesis_schema(intent.action)
            prompt = get_synthesis_prompt(intent.action, intent.language)
            log.debug(f"synthesize: Using schema={schema}")
            
            # Step 3: Build prompt payload
            payload = self._build_prompt_payload(intent, context_result)
            log.debug(f"synthesize: Built payload with {len(payload)} variables")
            log.debug(f"synthesize: Payload content: {json.dumps(payload, indent=2)}")
            
            # Step 4: Call LLM với structured output
            raw_synthesis = self._call_llm_with_retry(prompt, payload, schema, intent)
            log.debug(f"synthesize: LLM response received - {raw_synthesis.answer[:100]}... (truncated)")
            
            # Step 5: Post-process và sanitize
            final_output = self._post_process_synthesis(raw_synthesis, context_result, intent)
            log.debug(f"synthesize: Post-processed output - confidence={final_output.confidence:.2f}, "
                        f"items={len(final_output.items)}, citations={len(final_output.citations)}")
            
            # Step 6: Update stats
            processing_time = int((time.time() - start_time) * 1000)
            self.stats["successful_calls"] += 1
            self._update_avg_processing_time(processing_time)
            
            # Add processing metadata
            final_output.processing_time_ms = processing_time
            final_output.evidence_count = len(context_result.chunks_selected)
            
            log.info(f"synthesize: Success - confidence={final_output.confidence:.2f}, items={len(final_output.items)}")
            return final_output
            
        except Exception as e:
            self.stats["failed_calls"] += 1
            log.error(f"synthesize: Failed after {time.time() - start_time:.2f}s - {e}")
            
            # Return degraded but valid output
            return self._create_fallback_output(intent, validated_results, str(e))
    
    def _prepare_context(
        self,
        validated_results: list[PlanExecutionResult],
        intent: QueryIntent
    ) -> ContextPreparationResult:
        """
        Prepare evidence context optimized for LLM consumption.
        
        Delegates to synthesis_context module but adds engine-specific logic:
        - Adjust token budget based on LLM max_tokens
        - Log context quality metrics
        - Handle context preparation failures gracefully
        """
        try:
            # Reserve tokens for LLM response
            available_tokens = int(self.max_tokens * 0.7)  # 70% for context, 30% for response
            
            # Update context config với available budget
            context_config = ContextPreparationConfig(
                max_tokens=min(available_tokens, self.context_config.max_tokens),
                max_snippet_lines=self.context_config.max_snippet_lines,
                max_items=self.context_config.max_items,
                quality_threshold=self.context_config.quality_threshold,
                relevance_threshold=self.context_config.relevance_threshold
            )
            
            result = prepare_evidence_context(validated_results, intent, context_config)
            
            # Log context quality
            log.debug(f"_prepare_context: selection_ratio={result.selection_ratio:.2%}, "
                     f"token_utilization={result.token_utilization:.2%}")
            
            return result
            
        except Exception as e:
            log.warning(f"_prepare_context: Failed to prepare optimal context - {e}")
            
            # Fallback: minimal context
            return ContextPreparationResult(
                context=f"Limited evidence available: {len(validated_results)} items",
                chunks_selected=[],
                chunks_total=len(validated_results),
                tokens_used=50,
                tokens_available=available_tokens
            )
    
    def _build_prompt_payload(
        self,
        intent: QueryIntent,
        context_result: ContextPreparationResult
    ) -> Dict[str, Any]:
        """
        Build payload for prompt template formatting.
        
        Creates all variables needed by synthesis prompt template:
        - Query context (action, target, language)
        - Evidence context (formatted string)
        - Analysis request (action-specific instructions)
        """
        return {
            "action": intent.action.value,
            "mention": intent.mention or "N/A",
            "language": intent.language.value,
            "evidence_context": context_result.context,
            "analysis_request": get_analysis_request(intent.action, intent)
        }
    
    def _call_llm_with_retry(
        self,
        prompt: ChatPromptTemplate,
        payload: Dict[str, Any],
        schema: Dict[str, Any],
        intent: QueryIntent,
        max_retries: int = 2
    ) -> SynthesisOutput:
        """
        Call LLM với structured output và retry logic.
        
        Uses run_llm_json from llm.py module với synthesis-specific error handling.
        Tracks repair attempts untuk monitoring.
        """
        try:
            # Use run_llm_json với synthesis schema
            from src.code_graph_rag.agent.models import SynthesisOutput as SynthesisOutputModel
            
            synthesis_output = run_llm_json(
                prompt=prompt,
                payload=payload,
                schema=SynthesisOutputModel,  # Use Pydantic model directly
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                max_retries=max_retries
            )
            
            return synthesis_output
            
        except LLMJsonError as e:
            # Track repair attempts
            if "repair" in str(e).lower():
                self.stats["repair_attempts"] += 1
            
            log.warning(f"_call_llm_with_retry: LLM JSON error - {e}")
            raise SynthesisError(
                error_type="llm_error",
                message=f"LLM failed to generate valid JSON: {e}",
                raw_response=getattr(e, 'last_response', None),
                repair_attempted=True
            )
        
        except Exception as e:
            log.error(f"_call_llm_with_retry: Unexpected error - {e}")
            raise SynthesisError(
                error_type="unknown_error",
                message=f"Synthesis failed: {e}",
                raw_response=None,
                repair_attempted=False
            )
    
    def _post_process_synthesis(
        self,
        synthesis_output: SynthesisOutput,
        context_result: ContextPreparationResult,
        intent: QueryIntent
    ) -> SynthesisOutput:
        """
        Post-process và sanitize synthesis output.
        
        Ensures output quality và security:
        1. Validate citations exist in evidence
        2. Sanitize content (remove internal paths, trim long snippets)
        3. Adjust confidence based on evidence quality
        4. Add missing metadata
        """
        
        # Step 1: Validate và clean citations
        valid_citations = self._validate_citations(synthesis_output.citations, context_result)
        synthesis_output.citations = valid_citations
        
        # Step 2: Sanitize content
        synthesis_output = self._sanitize_content(synthesis_output, intent)
        
        # Step 3: Adjust confidence based on evidence quality
        evidence_quality = context_result.quality_stats.get("avg_relevance", 0.5)
        selection_ratio = context_result.selection_ratio
        
        # Lower confidence if evidence is poor or incomplete
        confidence_penalty = 0.0
        if evidence_quality < 0.3:
            confidence_penalty += 0.2
        if selection_ratio < 0.5:
            confidence_penalty += 0.1
        
        synthesis_output.confidence = max(0.1, synthesis_output.confidence - confidence_penalty)
        
        # Step 4: Add metadata if missing
        if not synthesis_output.evidence_count:
            synthesis_output.evidence_count = len(context_result.chunks_selected)
        
        log.debug(f"_post_process_synthesis: confidence={synthesis_output.confidence:.2f}, "
                 f"citations={len(synthesis_output.citations)}")
        
        return synthesis_output
    
    def _validate_citations(
        self,
        citations: list,
        context_result: ContextPreparationResult
    ) -> list:
        """
        Validate that all citations reference actual evidence.
        
        Removes citations that don't correspond to evidence chunks,
        ensuring no hallucinated references.
        """
        valid_entity_ids = {chunk.entity_id for chunk in context_result.chunks_selected}
        valid_citations = []
        
        for citation in citations:
            if hasattr(citation, 'id') and citation.id in valid_entity_ids:
                valid_citations.append(citation)
            else:
                log.warning(f"_validate_citations: Removing invalid citation {getattr(citation, 'id', 'unknown')}")
        
        return valid_citations
    
    def _sanitize_content(self, synthesis_output: SynthesisOutput, intent: QueryIntent) -> SynthesisOutput:
        """
        Sanitize synthesis content for security và quality.
        
        Removes/masks:
        - Internal file paths (keep relative paths only)
        - Long code excerpts (trim to reasonable length)
        - Sensitive information patterns
        """
        # Sanitize answer text
        synthesis_output.answer = self._sanitize_text(synthesis_output.answer)
        
        # Sanitize items (action-specific)
        sanitized_items = []
        for item in synthesis_output.items:
            if isinstance(item, dict):
                sanitized_item = self._sanitize_item(item, intent.action)
                sanitized_items.append(sanitized_item)
            else:
                sanitized_items.append(item)
        
        synthesis_output.items = sanitized_items
        
        # Sanitize notes
        if synthesis_output.notes:
            synthesis_output.notes = self._sanitize_text(synthesis_output.notes)
        
        return synthesis_output
    
    def _sanitize_text(self, text: str) -> str:
        """Remove sensitive patterns from text."""
        if not text:
            return text
        
        # Remove absolute paths, keep relative
        import re
        text = re.sub(r'/[a-zA-Z0-9_./]+/', '<path>', text)
        text = re.sub(r'C:\\[a-zA-Z0-9_.\\]+', '<path>', text)
        
        return text
    
    def _sanitize_item(self, item: Dict[str, Any], action: Action) -> Dict[str, Any]:
        """Sanitize individual item based on action type."""
        sanitized = dict(item)
        
        # Common sanitization
        for key, value in sanitized.items():
            if isinstance(value, str):
                sanitized[key] = self._sanitize_text(value)
            elif isinstance(value, dict) and 'id' in value:
                # Sanitize nested entity references
                value['id'] = self._sanitize_text(value['id'])
        
        # Action-specific sanitization
        if action == Action.explain_function:
            # Trim long snippet excerpts
            if 'snippet_excerpt' in sanitized and len(sanitized['snippet_excerpt']) > 500:
                sanitized['snippet_excerpt'] = sanitized['snippet_excerpt'][:497] + "..."
        
        return sanitized
    
    def _create_fallback_output(
        self,
        intent: QueryIntent,
        validated_results: list[PlanExecutionResult],
        error_message: str
    ) -> SynthesisOutput:
        """
        Create minimal valid output when synthesis fails.
        
        Ensures GraphAgent always receives valid SynthesisOutput,
        even when LLM or processing fails.
        """
        log.warning(f"_create_fallback_output: Creating fallback for {intent.action.value}")
        
        fallback_answer = {
            Language.vi: f"Không thể phân tích hoàn chỉnh. Tìm thấy {len(validated_results)} evidence items.",
            Language.en: f"Unable to complete analysis. Found {len(validated_results)} evidence items."
        }
        
        return SynthesisOutput(
            action=intent.action,
            language=intent.language,
            answer=fallback_answer[intent.language],
            items=[],  # Empty items array
            citations=[],
            confidence=0.1,  # Very low confidence
            evidence_count=len(validated_results),
            notes=f"Synthesis failed: {error_message[:100]}..."  # Truncated error
        )
    
    def _update_avg_processing_time(self, processing_time_ms: int) -> None:
        """Update running average of processing time."""
        current_avg = self.stats["avg_processing_time_ms"]
        total_calls = self.stats["successful_calls"]
        
        if total_calls == 1:
            self.stats["avg_processing_time_ms"] = processing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats["avg_processing_time_ms"] = (
                alpha * processing_time_ms + (1 - alpha) * current_avg
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        return dict(self.stats)
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "repair_attempts": 0,
            "avg_processing_time_ms": 0.0
        }

# Convenience factory function
def create_synthesis_engine(
    provider: str = "nvidia",
    max_evidence_tokens: int = 3000,
    temperature: float = 0.0
) -> SynthesisEngine:
    """
    Factory function to create configured synthesis engine.
    
    Provides sensible defaults for common use cases.
    """
    context_config = ContextPreparationConfig(
        max_tokens=max_evidence_tokens,
        max_snippet_lines=20,
        max_items=50
    )
    
    return SynthesisEngine(
        provider=provider,
        temperature=temperature,
        max_tokens=2000,
        seed=42,
        context_config=context_config
    )