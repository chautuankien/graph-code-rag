"""Validation and retry helpers for plan execution results.

This module validates adapter output rows, deduplicates them, and performs
bounded retries for missing required steps to keep answers useful and
traceable.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Callable, Iterable
from src.code_graph_rag.agent.models import (
    PlanExecutionResult, PlanStep, QueryIntent,
    ResolvedEntity, ValidationReport
)
from src.code_graph_rag.agent.plan_runner import PlanExecutionError

# Allowed node labels aligned with the graph schema invariants.
VALID_LABELS = {"Project","Folder","Package","Module","File","Class","Function","Method","ExternalPackage"}


def _row_ok(r: PlanExecutionResult) -> tuple[bool, str | None]:
    if not (r.step and r.label and r.id):
        return False, "missing-core-fields"
    if r.label not in VALID_LABELS:
        return False, "bad-label"
    if r.label in {"Function","Method","Class"}:
        if r.path is None or r.start_line is None or r.end_line is None:
            return False, "missing-span"
        if r.start_line > r.end_line or r.start_line < 1:
            return False, "bad-span"
    return True, None


def _dedupe(rows: list[PlanExecutionResult]) -> list[PlanExecutionResult]:
    """Deduplicate rows by (label, id), preferring richer content.

    Preference is given to rows that include a code snippet and/or a
    docstring, as these provide better context to downstream steps.

    Args:
      rows: Candidate rows that may include duplicates.

    Returns:
      list[PlanExecutionResult]: Unique rows, keeping the best representative
      per ``(label, id)`` key.
    """
    best: dict[tuple[str,str], PlanExecutionResult] = {}
    for r in rows:
        k = (r.label or "", r.id or "")
        cur = best.get(k)
        # WHY: Prefer the row with more context (snippet/docstring) to aid UX.
        score = (r.snippet is not None) + (r.docstring is not None)
        cur_score = ((cur.snippet is not None) + (cur.docstring is not None)) if cur else -1
        if cur is None or score > cur_score:
            best[k] = r
    return list(best.values())


def validate_and_retry(
    *,
    rows: list[PlanExecutionResult],
    required_steps: set[str],
    retry_cb,             # callable(step_name: str, strategy: str) -> list[PlanExecutionResult]
    max_retries: int = 2,
) -> tuple[list[PlanExecutionResult], ValidationReport]:
    """Validate, deduplicate, and retry to satisfy required plan steps.

    The pipeline performs three phases: (1) schema validation, (2) best-effort
    deduplication favoring richer rows, and (3) bounded retries for steps that
    remain empty using progressively relaxed strategies.

    Args:
      rows: Initial rows produced by adapters.
      required_steps: Step names that must be present at least once.
      retry_cb: Callback that runs a relaxed query for ``step`` with a given
        ``strategy`` and returns candidate rows.
      max_retries: Max attempts per strategy before escalating.

    Returns:
      tuple[list[PlanExecutionResult], ValidationReport]: Cleaned rows and a
      report summarizing drops, keeps, and retry issues.

    Raises:
      PlanExecutionError: If any required step remains empty after exhausting
      all strategies and retries.

    Algorithm:
      1. Validate and drop rows failing schema/span checks; record reasons.
      2. Deduplicate by ``(label, id)`` preferring rows with snippet/docstring.
      3. For each missing required step, try strategies in order
         ("relax-params", "fallback-adapter", "simplify-id"), each up to
         ``max_retries`` attempts; on success, merge deduped results; on final
         failure, raise ``PlanExecutionError`` and record an issue.
    """
    rep = ValidationReport()
    cleaned: list[PlanExecutionResult] = []

    # WHY: Drop invalid rows early to avoid retrying or showing bad data.
    # 1) schema checks
    for r in rows:
        ok, reason = _row_ok(r)
        if ok:
            cleaned.append(r)
        else:
            rep.dropped += 1
            rep.reasons[reason] = rep.reasons.get(reason, 0) + 1

    # Keep a copy of valid rows BEFORE deduplication to compute step coverage.
    # WHY:
    #   Coverage of required steps must be determined by "valid but not yet
    #   deduped" rows. If we use the deduped list, two required steps that
    #   produce the SAME (label, id) would collapse into a single row and make
    #   one step appear "missing", which would incorrectly trigger a retry.
    valid_pre_dedupe = list(cleaned)

    # WHY: Prefer rows with more usable context to improve answer quality.
    # 2) dedupe
    cleaned = _dedupe(cleaned)
    rep.kept = len(cleaned)

    # 3) retry missing required steps
    got_by_step = {r.step for r in valid_pre_dedupe}
    missing = [s for s in required_steps if s not in got_by_step]
    for step in missing:
        success = False
        # WHY: Escalate from least to most permissive to preserve precision.
        for strategy in ("relax-params", "fallback-adapter", "simplify-id"):
            for _ in range(max_retries):
                try_rows = retry_cb(step, strategy=strategy) or []
                try_rows = _dedupe([r for r in try_rows if _row_ok(r)[0]])
                if try_rows:
                    cleaned.extend(try_rows)
                    success = True
                    break
            if success:
                break
        if not success:
            msg = f"Required step still empty after retries: {step}"
            rep.issues.append(msg)
            raise PlanExecutionError(msg)

    return cleaned, rep


# Helpers
_PKG_ALIAS = {
    "pil": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "pyyaml": "pyyaml",
    "bs4": "beautifulsoup4",
}

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def _bump_limit(val: int | None, factor: float = 2.0) -> int:
    base = int(val or 50)
    return _clamp_int(int(base * factor), 1, 200)

def _inc_depth(val: int | None) -> int:
    base = int(val or 2)
    return _clamp_int(base + 1, 1, 5)

def _normalize_pkg(name: str) -> str:
    return _PKG_ALIAS.get(name.lower(), name.lower())

def _simplify_qname(q: str) -> Iterable[str]:
    # Generate progressively shorter candidates.
    # E.g., repo.mclass.C.m1 -> repo.mclass.C -> repo.mclass -> repo
    parts = q.split(".")
    for k in range(len(parts), 0, -1):
        yield ".".join(parts[:k])

def make_retry_cb(
    *,
    plan_steps: list[PlanStep],
    runner: Callable[[list[PlanStep], QueryIntent, ResolvedEntity, str | None], list[PlanExecutionResult]],
    intent: QueryIntent,
    resolved: ResolvedEntity,
    repo_root: str | None,
) -> Callable[[str, str], list[PlanExecutionResult]]:
    steps_by_name = {s.name: s for s in plan_steps}

    def _run_step(step: PlanStep) -> list[PlanExecutionResult]:
        # Chạy 1 step với enrich như bình thường để trả về PlanExecutionResult.
        return runner([step], intent, resolved, repo_root)

    def retry_cb(step_name: str, strategy: str) -> list[PlanExecutionResult]:
        base = steps_by_name.get(step_name)
        if not base:
            return []
        step = deepcopy(base)

        # Strategy 1: relax-params
        if strategy == "relax-params":
            # Common limit bump
            lim = step.params.get("limit", intent.limit)
            step.params["limit"] = _bump_limit(lim)

            if step.name == "NEIGHBORHOOD":
                dep = step.params.get("depth", intent.depth)
                step.params["depth"] = _inc_depth(dep)

            if step.name == "ENTRY_FUNCS_BY_KEYWORD":
                kw = (step.params.get("kw") or intent.mention or "").strip()
                if kw and len(kw) < 3:
                    # Cho phép contains (adapter đã hỗ trợ) bằng cách tăng limit
                    step.params["limit"] = _bump_limit(lim, factor=2.5)

            if step.name == "MODULES_DEPENDING_ON_EXTERNAL":
                pkg = step.params.get("package") or intent.mention or ""
                step.params["package"] = _normalize_pkg(pkg)

            return _run_step(step)

        # Strategy 2: fallback-adapter
        if strategy == "fallback-adapter":
            # Map một số fallback an toàn
            src_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
            if step.name == "METHODS_OF_CLASS":
                fb = deepcopy(step)
                fb.name = "NEIGHBORHOOD"
                fb.params = {"id": src_id, "depth": 1, "limit": step.params.get("limit", 50)}
                rows = _run_step(fb)
                # Lọc: Method thuộc class đó (id prefix = "<class_id>.")
                return [r for r in rows if r.label == "Method" and r.id and r.id.startswith(f"{src_id}.")]

            if step.name == "INHERITS_DIRECT":
                fb = deepcopy(step)
                fb.name = "NEIGHBORHOOD"
                fb.params = {"id": src_id, "depth": 1, "limit": step.params.get("limit", 50)}
                rows = _run_step(fb)
                return [r for r in rows if r.label == "Class"]

            if step.name in {"CALLERS_TOP", "CALLEES_TOP"}:
                fb = deepcopy(step)
                fb.name = "NEIGHBORHOOD"
                fb.params = {"id": src_id, "depth": 1, "limit": step.params.get("limit", 50)}
                return _run_step(fb)

            if step.name == "META":
                fb = deepcopy(step)
                fb.name = "NODE_META"
                fb.params = {"ids": [src_id]}
                return _run_step(fb)

            if step.name == "MODULES_DEPENDING_ON_EXTERNAL":
                # Thử alias khác của package trước khi bỏ cuộc
                pkg = step.params.get("package") or intent.mention or ""
                candidates = {pkg.lower(), _normalize_pkg(pkg)}
                out: list[PlanExecutionResult] = []
                for c in candidates:
                    fb = deepcopy(step)
                    fb.params["package"] = c
                    out = _run_step(fb)
                    if out:
                        break
                return out

            return []

        # Strategy 3: simplify-id
        if strategy == "simplify-id":
            src_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
            if not src_id:
                return []
            tried: set[str] = set()
            for cand in _simplify_qname(src_id):
                if cand in tried:
                    continue
                tried.add(cand)
                fb = deepcopy(step)
                fb.params["id"] = cand
                rows = _run_step(fb)
                if rows:
                    return rows
            return []

        # Unknown strategy
        return []

    return retry_cb