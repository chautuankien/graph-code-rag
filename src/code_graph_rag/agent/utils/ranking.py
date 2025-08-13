"""Ranking helpers for EntityResolver.

This module provides deterministic scoring and ranking for candidate graph
entities produced by the resolver's Cypher queries. It prioritizes exact
matches, then suffix/name matches, and finally fuzzy token coverage, with a
small type-based boost to prefer Method > Function > Class > Module.

WHY:
    Natural-language mentions like "main" or "pkg.mod.Class" can map to
    multiple graph nodes. We need a stable, explainable scoring function to
    pick the best candidate (or surface ambiguous options to the UI/Planner).

PERF:
    Pure-Python scoring; operates on already-limited result sets from Cypher
    (LIMIT 50..400). Sorting is O(n log n) for small n.

SECURITY:
    No I/O here. All database interaction is performed in resolver.py with
    parameterized Cypher.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

# Small, explainable priority bump by node type. Keeps behavior deterministic.
# WHY:
#   Short mentions (e.g., "main", "run") are ambiguous across Method/Function/
#   Class/Module. In code Q&A, users typically refer to behaviors (methods/
#   functions). A small, deterministic type-based boost nudges ranking toward
#   executable entities, improving downstream CALLS/CALLEES queries.
#
# DESIGN:
#   Method > Function > Class > Module with tiny margins (0.08/0.05/0.03/0.00).
#   The boost must never overshadow a clearly better textual match; it only
#   breaks near-ties among candidates with similar base_score tiers.
#
# RISK MITIGATION:
#   Scores are clamped to [0,1]. Acceptance still depends on thresholds
#   (e.g., >=0.80 auto-accept). Ambiguous 0.50–0.80 returns candidates for UI.
_TYPE_BOOST: dict[str, float] = {
    "Method": 0.08,
    "Function": 0.05,
    "Class": 0.03,
    "Module": 0.00,
}


def _tokenize(s: str) -> list[str]:
    """Split an identifier into searchable tokens.

    The function normalizes separators ('.', '_') and splits camelCase to
    improve fuzzy matching coverage.

    Args:
        s: Raw string, e.g. "pkg.mod.RequestHandler.handle_request".

    Returns:
        A list of lowercase tokens, e.g. ["pkg", "mod", "request", "handler",
        "handle", "request"].

    Examples:
        >>> _tokenize("MyHTTPHandler.handle_get")
        ['my', 'http', 'handler', 'handle', 'get']
    """
    # Normalize separators first.
    s = s.replace(".", " ").replace("_", " ")
    # Split camelCase and numbers.
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", s)
    return [p.lower() for p in parts if p]


def base_score(mention: str, label: str, qname: str, name: str | None) -> float:
    """Compute a base 0..1 score from multiple matching tiers.

    Tiers (first match wins):
      1. 1.00 if mention == qname (exact qualified_name)
      2. 0.90 if name and mention == name (exact simple name)
      3. 0.85 if name and qname endswith ".name" and mention == name
      4. 0.60..0.80 from token coverage between mention and qname

    WHY:
        Exact matches should dominate. When only partial hints exist, prefer
        candidates whose tokens cover the mention more.

    Args:
        mention: Raw textual mention from the user.
        label: Candidate label ("Module"|"Class"|"Function"|"Method"|...).
        qname: Candidate canonical identifier (usually qualified_name).
        name: Optional simple name of the candidate.

    Returns:
        A float in [0, 1].

    Examples:
        >>> base_score("pkg.mod.main", "Function", "pkg.mod.main", "main")
        1.0
        >>> base_score("main", "Function", "pkg.mod.main", "main")
        0.9
    """
    m = mention.strip()
    if not m:
        return 0.0
    if m == qname:
        return 1.0
    if name and m == name:
        return 0.90
    if name and qname.endswith("." + name) and (m == name):
        return 0.85

    # --- FUZZY RANKING EXPLANATION -----------------------------------------
    # When exact tiers fail, we compute a *soft* similarity using token overlap.
    # 1) Tokenization:
    #    - We call _tokenize() which splits on dot/underscore and camelCase.
    #      Example: "proj.web.RequestHandler.handle_request"
    #      -> ["proj","web","request","handler","handle","request"]
    # 2) Coverage metric:
    #    - coverage = |tokens(mention) ∩ tokens(candidate)| / |tokens(mention)|
    #      This measures how well the candidate *covers* the user's hint.
    #      Example: mention={"request","handler"}, candidate contains both 
    #      tokens(mention) ∩ tokens(candidate) = {"request","handler"} = 2
    #      coverage = 2 / 2 = 1.0 (perfect match)
    # 3) Score mapping:
    #    - We map coverage into the 0.60..0.80 band:
    #        score = 0.60 + 0.20 * coverage
    #      Rationale:
    #        * 0.60 is a weak-hint baseline (some relation but not strong).
    #        * +0.20 rewards full coverage but *keeps fuzzy below exact tiers*:
    #            0.90 (exact simple name) and 0.85 (suffix ".name").
    # 4) Ordering guarantees:
    #    - Fuzzy scores can never outrank exact-name/suffix tiers.
    #    - A later, small type boost (Method/Function > Class > Module) can
    #      break near ties among fuzzy candidates but cannot surpass strong
    #      exact matches. This keeps ranking stable and explainable.
    # -----------------------------------------------------------------------

    # Fallback: fuzzy ranking coverage
    mtoks = set(_tokenize(m))
    qtoks = set(_tokenize(qname))
    log.debug("base_score.mtoks: %s", mtoks)
    log.debug("base_score.qtoks: %s", qtoks)
    if not mtoks or not qtoks:
        return 0.0
    inter = len(mtoks & qtoks)
    log.debug("base_score.intersection: %s", inter)
    cov = inter / max(1, len(mtoks))
    log.debug("base_score.coverage: %s", cov)
    # Base fuzzy band 0.60..0.80 depending on coverage.
    return 0.60 + 0.20 * cov


def with_type_boost(score: float, label: str) -> float:
    """Apply a small, explainable type boost.

    WHY:
        In code Q&A, users more often mean a method or function when using
        short mentions like "main" or "handle". This nudge helps precision
        without masking poor matches.

    Args:
        score: Base score in [0, 1].
        label: Candidate label.

    Returns:
        Clamped score in [0, 1] after adding the label-specific boost.

    Examples:
        >>> with_type_boost(0.78, "Method")
        0.86
    """
    return max(0.0, min(1.0, score + _TYPE_BOOST.get(label, 0.0)))


def finalize_scores(
    rows: Iterable[Tuple[str, str, str | None]], mention: str
) -> list[tuple[str, str, float]]:
    """Score and rank raw candidates deterministically.

    Args:
        rows: Iterable of (label, id_or_qname, simple_name_or_None).
        mention: Original textual mention.

    Returns:
        A sorted list of (label, id, final_score) in descending score order,
        with a stable tie-breaker on the candidate id.

    Algorithm:
        1) For each row, compute base_score().
        2) Add with_type_boost().
        3) Round to 6 decimals for stability/readability.
        4) Sort by (-score, id).

    Examples:
        >>> rows = [("Function", "pkg.mod.main", "main"),
        ...         ("Method", "pkg.App.main", "main")]
        >>> finalize_scores(rows, "main")[0][0]
        'Method'
    """
    log.debug("finalize_scores.rows: %s", rows)
    log.debug("finalize_scores.mention: %s", mention)
    out: list[tuple[str, str, float]] = []
    for label, qid, name in rows:
        s = base_score(mention, label, qid, name)
        s = with_type_boost(s, label)
        out.append((label, qid, round(s, 6)))
    out.sort(key=lambda x: (-x[2], x[1]))
    return out
