"""Entity Resolver.

Resolves free-text mentions (e.g. "main", "pkg.mod.Class") to canonical graph
identifiers (qualified_name/path) using a sequence of parameterized Cypher
queries and a deterministic ranker.

WHY:
    Planner/GraphQuery require precise node ids. This module converts user
    mentions into (label, id, confidence) plus a ranked candidate list when
    ambiguity exists.

SECURITY:
    * All Cypher queries are parameterized to avoid injection.
    * Queries are allow-listed and bounded with LIMITs.

PERF:
    * Queries target indexed properties (qualified_name, name) wherever
      possible. Fuzzy broad search is last resort with tight LIMITs.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.code_graph_rag.agent.models import (
    Candidate,
    QueryIntent,
    ResolvedEntity,
)
from src.code_graph_rag.agent.utils.ranking import finalize_scores
from src.code_graph_rag.agent.utils.utils import run_cypher_query  # provided by agent

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)


# Confidence bands. Keep these in sync with design doc & Planner behavior.
C_ACCEPT = 0.80
C_MAYBE = 0.50
WIN_MARGIN = 0.05


def _rows_exact_qname(q: str) -> list[dict[str, Any]]:
    """Return candidates with exact qualified_name match.

    Args:
        q: The fully-qualified name to match.

    Returns:
        List of dict rows: {"label", "id", "name"}.

    Raises:
        None. Upstream run_cypher_query should raise with context on failure.
    """
    cypher = """
    MATCH (n)
    WHERE (n:Module OR n:Class OR n:Function OR n:Method)
      AND coalesce(n.qualified_name, '') = $q
    RETURN head(labels(n)) AS label,
           coalesce(n.qualified_name, n.path) AS id, n.name AS name
    LIMIT 50
    """
    return run_cypher_query(cypher, {"q": q})


def _rows_exact_name(name: str) -> list[dict[str, Any]]:
    """Return candidates whose simple name equals `name`.

    Searches Function, Method, and Class by exact `n.name = name`.
    Module names vary per exporter, so Module is handled in suffix search.

    Args:
        name: The simple symbol name, e.g. "main".

    Returns:
        List of dict rows: {"label", "id", "name"}.
    """
    cypher = """
    CALL {
        MATCH (n:Function) WHERE n.name = $name
        RETURN 'Function' AS label, n.qualified_name AS id, n.name AS name
        UNION ALL
        MATCH (n:Method) WHERE n.name = $name
        RETURN 'Method'  AS label, n.qualified_name AS id, n.name AS name
        UNION ALL
        MATCH (n:Class) WHERE n.name = $name
        RETURN 'Class'   AS label, n.qualified_name AS id, n.name AS name
    }
    RETURN label, id, name
    LIMIT 200
    """
    return run_cypher_query(cypher, {"name": name})


def _rows_suffix(name: str) -> list[dict[str, Any]]:
    """Return Module candidates based on suffix/equals match.

    WHY:
        Users often mention a module by its stem (e.g., "util"), while the
        graph stores fully-qualified module names.

    Args:
        name: Module stem or name.

    Returns:
        List of dict rows: {"label"="Module", "id", "name"}.
    """
    cypher = """
    CALL {
        MATCH (m:Module)
        WHERE m.qualified_name ENDS WITH ('.' + $name)
            OR m.name = $name
            OR m.qualified_name = $name
        RETURN 'Module' AS label, m.qualified_name AS id, m.name AS name
    }
    RETURN label, id, name
    LIMIT 200
    """
    return run_cypher_query(cypher, {"name": name})


def _rows_contains(token: str) -> list[dict[str, Any]]:
    """Return a broad set of candidates using CONTAINS, for fuzzy ranking.

    WHY:
        As a last resort when exact matches fail, gather plausible candidates
        across labels and defer the final decision to the Python ranker.

    Args:
        token: A meaningful substring (often the longest segment of mention).

    Returns:
        List of dict rows: {"label", "id", "name"} across Module/Class/
        Function/Method.
    """
    cypher = """
    CALL {
        MATCH (n:Module)  WHERE n.qualified_name CONTAINS $t
        RETURN 'Module'  AS label, n.qualified_name AS id, n.name AS name
        UNION ALL
        MATCH (n:Class)   WHERE n.qualified_name CONTAINS $t OR n.name = $t
        RETURN 'Class'   AS label, n.qualified_name AS id, n.name AS name
        UNION ALL
        MATCH (n:Function) WHERE n.qualified_name CONTAINS $t OR n.name = $t
        RETURN 'Function' AS label, n.qualified_name AS id, n.name AS name
        UNION ALL
        MATCH (n:Method)  WHERE n.qualified_name CONTAINS $t OR n.name = $t
        RETURN 'Method'  AS label, n.qualified_name AS id, n.name AS name
    }
    RETURN label, id, name
    LIMIT 400
    """
    return run_cypher_query(cypher, {"t": token})


def _rank(mention: str, rows: list[dict[str, Any]]) -> list[Candidate]:
    """Convert raw rows into ranked `Candidate` models.

    Args:
        mention: Original textual mention.
        rows: Dict rows containing "label", "id", "name".

    Returns:
        A list of `Candidate` (label, id, score, display) sorted by score.

    Algorithm:
        1) Project rows to (label, id, name).
        2) Use finalize_scores() to compute stable ranking.
        3) Wrap into Candidate models with a human-friendly display.
    """
    tuples = [(r["label"], r["id"], r.get("name")) for r in rows]
    log.debug("_rank.tuples: %s", tuples)
    ranked = finalize_scores(tuples, mention)  # [(label, id, score)]
    log.debug("_rank.ranked: %s", ranked)
    return [
        Candidate(label=lab, id=qid, score=sc, display=f"{lab}: {qid}")
        for lab, qid, sc in ranked
    ]


def _choose(
    cands: list[Candidate],
) -> tuple[str | None, str | None, float, str | None]:
    """Decide final pick (if any) from ranked candidates with a win-margin guard.

    Policy:
        - Auto-accept only if:
            top.score >= C_ACCEPT and
            (no runner-up or (top.score - runner_up.score) >= WIN_MARGIN).
        - Else if top.score >= C_MAYBE:
            ambiguous → return candidates with an assumption message.
        - Else:
            very uncertain.

    WHY:
        Exact-name ties across types (e.g., Method vs Function both at 0.90
        base tier) should not auto-resolve merely due to small type boosts.
        The win-margin guard prevents overconfident picks when alternatives
        are close.

    Returns:
        (resolved_label, resolved_id, confidence, assumption_message)
    """
    if not cands:
        return None, None, 0.0, "No candidates found"
    top = cands[0]
    runner_up = cands[1] if len(cands) > 1 else None
    log.debug("_choose.top: %s", top)
    log.debug("_choose.runner_up: %s", runner_up)

    # Only auto-accept if top is confident AND clearly ahead.
    if top.score >= C_ACCEPT and (runner_up is None or (top.score - runner_up.score) >= WIN_MARGIN):
        return top.label, top.id, top.score, None

    if top.score >= C_MAYBE:
        return None, None, top.score, "Low confidence or close alternatives; showing candidates"

    return None, None, top.score, "Very low confidence; likely ambiguous"


def resolve_one(
    mention: str,
) -> tuple[str | None, str | None, float, list[Candidate], str | None]:
    """Resolve a single mention to a node id with confidence and alternatives.

    Heuristic order (early stop on first non-empty result set):
        1) Exact qualified_name
        2) Exact simple name (Function/Method/Class)
        3) Module suffix / equals
        4) CONTAINS (broad) + Python-side ranking

    Args:
        mention: Free-text mention provided by the user.

    Returns:
        (label, id, confidence, candidates, assumption_message)

    Examples:
        >>> resolve_one("proj.app.main")  # doctest: +SKIP
        ('Function', 'proj.app.main', 1.0, [...], None)
    """
    m = (mention or "").strip()
    log.debug("resolve_one.mention: %s", m)
    if not m:
        return None, None, 0.0, [], "Empty mention"

    # 1) Exact qname
    rows: list[dict[str, Any]] = _rows_exact_qname(m)
    log.debug("resolve_one.exact_qname_rows: %s", rows)

    # 2) Exact simple name
    if not rows:
        rows = _rows_exact_name(m)
        log.debug("resolve_one.exact_name_rows: %s", rows)

    # 3) Module suffix
    if not rows:
        rows = _rows_suffix(m)
        log.debug("resolve_one.suffix_rows: %s", rows)

    # 4) Broad contains (pick longest token to reduce noise)
    if not rows:
        token = max(m.split("."), key=len)
        rows = _rows_contains(token)
        log.debug("resolve_one.contains_rows: %s", rows)

    cands = _rank(m, rows)
    lab, rid, conf, assumption = _choose(cands)
    return lab, rid, conf, cands, assumption


def resolve_entity(intent: QueryIntent) -> ResolvedEntity:
    """Resolve `intent.mention` (and optional `mention_dst`) into graph ids.

    WHY:
        Downstream steps (Planner/GraphQuery) require canonical ids. This
        function wraps two calls to `resolve_one` and returns a structured
        `ResolvedEntity` suitable for Agent state.

    Args:
        intent: Parsed user intent carrying `mention` and `mention_dst`.

    Returns:
        A `ResolvedEntity` with:
            - resolved_label/resolved_id/confidence (source)
            - candidates (ranked alternatives)
            - optional assumption message for UI/planner
            - *_dst fields if `mention_dst` is provided

    Examples:
        >>> from src.code_graph_rag.agent.models import QueryIntent
        >>> qi = QueryIntent(action="list_callers", mention="main")
        >>> resolve_entity(qi)  # doctest: +SKIP
        ResolvedEntity(...)
    """
    src_lab, src_id, src_conf, src_cands, src_assump = resolve_one(
        intent.mention or ""
    )
    log.debug("resolve_entity.source: %s, %s, %s, %s, %s", src_lab, src_id, src_conf, src_cands, src_assump)
    dst_lab, dst_id, dst_conf, dst_cands, dst_assump = (None, None, 0.0, [], None)

    log.debug("resolve_entity.intent.mention_dst: %s", intent.mention_dst)
    if intent.mention_dst:
        dst_lab, dst_id, dst_conf, dst_cands, dst_assump = resolve_one(
            intent.mention_dst
        )
        log.debug("resolve_entity.destination: %s, %s, %s, %s, %s", dst_lab, dst_id, dst_conf, dst_cands, dst_assump)

    assumption = src_assump or dst_assump
    return ResolvedEntity(
        resolved_label=src_lab,
        resolved_id=src_id,
        confidence=src_conf,
        candidates=src_cands,
        assumption=assumption,
        resolved_label_dst=dst_lab,
        resolved_id_dst=dst_id,
        confidence_dst=dst_conf,
        candidates_dst=dst_cands,
    )
