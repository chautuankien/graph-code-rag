"""Graph adapters: small, parameterized Cypher queries for CodeGraph RAG.

Each adapter exposes a focused graph query via a small, pure function.
Docstrings follow Google style; comments explain intent and constraints.
"""

from __future__ import annotations
from typing import Any

from src.code_graph_rag.agent.models import QueryIntent, ResolvedEntity, PlanStep
from src.code_graph_rag.agent.adapters.registry import register
from src.code_graph_rag.agent.utils.utils import run_cypher_query
from src.code_graph_rag.utils.logging_setup import get_logger

log = get_logger(__name__)

# -- helper --------------------------------------------------------------------
def _coerce_int(x: Any, lo: int, hi: int, default: int) -> int:
    """Coerce a value to an int and clamp it to a closed interval.

    Falls back to ``default`` if conversion fails, then clamps to ``[lo, hi]``.

    Args:
      x: Value to convert to ``int``.
      lo: Inclusive lower bound.
      hi: Inclusive upper bound.
      default: Value to return when conversion fails.

    Returns:
      int: An integer in the range ``[lo, hi]``.
    """
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))

# -- META ----------------------------------------------------------------------
@register("META")
def meta_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """Fetch canonical metadata for a node by qualified_name or path.

    Args:
      step: Plan step containing parameters (expects ``id``).
      intent: Parsed query intent (may provide fallback ``mention``).
      resolved: Entity resolution results (may provide ``resolved_id``).

    Returns:
      list[dict]: Up to one row with keys ``label``, ``id``, ``name``.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    log.debug("meta_adapter: id=%s", node_id)

    """
    This runs a Cypher query that:
    1. Executes a subquery with UNION ALL branches to match exactly one node
       by identifier.
    2. Tries code entities by `qualified_name = $id` (Module, Class, Function,
       Method) and files by `path = $id`.
    3. Normalizes outputs to `(label, id, name)` for downstream consumers.
    4. Returns at most one row via `LIMIT 1`.

    Notes:
    - With unique identifiers, only one branch should match. If multiple could
      match, the first winning branch determines the result due to `LIMIT 1`.
    - Keep indexes on `qualified_name` (code nodes) and `path` (files) for
      fast lookups in Memgraph.
    - The query is fully parameterized; no string interpolation is used.
    """
    q = """
    CALL {
      MATCH (n:Module  {qualified_name:$id}) RETURN 'Module'  AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      MATCH (n:Class   {qualified_name:$id}) RETURN 'Class'   AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      MATCH (n:Function{qualified_name:$id}) RETURN 'Function'AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      MATCH (n:Method  {qualified_name:$id}) RETURN 'Method'  AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      MATCH (n:File    {path:$id})           RETURN 'File'    AS label, n.path           AS id, n.path AS name
    }
    RETURN label, id, name LIMIT 1
    """
    return run_cypher_query(q, {"id": node_id})

# -- CALLERS_TOP ---------------------------------------------------------------
@register("CALLERS_TOP")
def callers_top_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """List top callers of a function/method/class by qualified name.

    Args:
      step: Plan step containing parameters (``id``, optional ``limit``).
      intent: Parsed query intent providing defaults.
      resolved: Resolution results that may include ``resolved_id``.

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for caller nodes.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("callers_top_adapter: id=%s limit=%s", node_id, limit)

    """
    This runs a Cypher query that:
    1. Follows incoming `CALLS` relationships to the target whose `qualified_name`
       equals `$id`, binding callers to `src`.
    2. Uses `WITH DISTINCT src` to remove duplicate callers in case multiple
       `CALLS` edges exist from the same caller.
    3. Returns:
       - `label`: the first label of the caller node (e.g., "Function",
         "Method", "Class", "Module", or "File").
       - `id`: the caller's `qualified_name` if present, otherwise `path`.
       - `name`: the caller's `name` if present, otherwise `path`.
    4. Sorts results by `label` then `id`, and limits the number of results to
       `$limit`.

    Note: "Top" here means the first N unique callers after sorting. It is not
    ranked by call frequency. To rank by frequency, aggregate and order by the
    count of CALLS edges per caller.
    """
    q = """
    MATCH (src)-[:CALLS]->(dst {qualified_name:$id})
    WITH DISTINCT src
    RETURN labels(src)[0] AS label,
           coalesce(src.qualified_name, src.path) AS id,
           coalesce(src.name, src.path) AS name
    ORDER BY label ASC, id ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"id": node_id, "limit": limit})

# -- CALLEES_TOP ---------------------------------------------------------------
@register("CALLEES_TOP")
def callees_top_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """List top callees of a function/method/class by qualified name.

    Args:
      step: Plan step containing parameters (``id``, optional ``limit``).
      intent: Parsed query intent providing defaults.
      resolved: Resolution results that may include ``resolved_id``.

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for callee nodes.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("callees_top_adapter: id=%s limit=%s", node_id, limit)

    """
    This runs a Cypher query that:
    1. Matches the node whose `qualified_name` equals the given `id`.
    2. Traverses outgoing `CALLS` relationships to find all callees (`dst`).
    3. Uses `WITH DISTINCT dst` to remove duplicates in case multiple CALLS edges
       point to the same callee.
    4. Returns:
       - `label`: the first label of the callee node (e.g., "Function", "Method", "Class").
       - `id`: the callee's `qualified_name` if present, otherwise `path`.
       - `name`: the callee's `name` if present, otherwise `path`.
    5. Sorts results by `label` then `id`, and limits the number of results to `limit`.
    """
    q = """
    MATCH (src {qualified_name:$id})-[:CALLS]->(dst)
    WITH DISTINCT dst
    RETURN labels(dst)[0] AS label,
           coalesce(dst.qualified_name, dst.path) AS id,
           coalesce(dst.name, dst.path) AS name
    ORDER BY label ASC, id ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"id": node_id, "limit": limit})

# -- IMPORTS -------------------------------------------------------------------
@register("IMPORTS")
def imports_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """List outgoing imports from a module by qualified name.

    Args:
      step: Plan step containing parameters (``id``, optional ``limit``).
      intent: Parsed query intent providing defaults.
      resolved: Resolution results that may include ``resolved_id``.

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for imported nodes.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("imports_adapter: id=%s limit=%s", node_id, limit)

    q = """
    MATCH (m:Module {qualified_name:$id})-[:IMPORTS]->(t)
    WITH DISTINCT t
    RETURN labels(t)[0] AS label,
           coalesce(t.qualified_name, t.path) AS id,
           coalesce(t.name, t.path) AS name
    ORDER BY label ASC, id ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"id": node_id, "limit": limit})

# -- NEIGHBORHOOD --------------------------------------------------------------
@register("NEIGHBORHOOD")
def neighborhood_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """Return a k-hop undirected neighborhood around a node.

    Depth is validated (1..5) and inlined in the pattern upper bound.

    Args:
      step: Plan step (expects ``id``, optional ``depth`` and ``limit``).
      intent: Parsed query intent providing defaults.
      resolved: Resolution results that may include ``resolved_id``.

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for neighbors.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    depth = _coerce_int(step.params.get("depth", intent.depth), 1, 5, 2)
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("neighborhood_adapter: id=%s depth=%s limit=%s", node_id, depth, limit)

    """
    This runs a Cypher query that:
    1. Matches the start node `n` whose `qualified_name` equals `$id`.
    2. Walks an undirected, variable-length path of 1..{depth} hops from `n`
       to any neighbor `m`, following any relationship type in either direction.
    3. Uses `WITH DISTINCT m` to deduplicate neighbors reachable via multiple paths.
    4. Returns:
       - `label`: the first label of `m` (e.g., Module, Class, Function, Method, File).
       - `id`: `m.qualified_name` if present, otherwise `m.path`.
       - `name`: `m.name` if present, otherwise `m.path`.
    5. Orders results by `label` then `id`, and limits the number of rows to
       `$limit`.

    Notes:
    - The upper bound of a variable-length pattern cannot be parameterized in
      Cypher; `depth` is safely validated (1..5) and interpolated.
    - The traversal is undirected (`-[]-`), so it discovers neighbors regardless
      of relationship direction.
    """
    q = f"""
    MATCH (n {{qualified_name:$id}})
    MATCH (n)-[*1..{depth}]-(m)
    WITH DISTINCT m
    RETURN labels(m)[0] AS label,
           coalesce(m.qualified_name, m.path) AS id,
           coalesce(m.name, m.path) AS name
    ORDER BY label ASC, id ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"id": node_id, "limit": limit})

# -- INHERITS_DIRECT -----------------------------------------------------------
@register("INHERITS_DIRECT")
def inherits_direct_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """List the direct parent classes of a class.

    Args:
      step: Plan step containing parameters (``id``, optional ``limit``).
      intent: Parsed query intent providing defaults.
      resolved: Resolution results that may include ``resolved_id``.

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for base classes.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("inherits_direct_adapter: id=%s limit=%s", node_id, limit)

    """
    This runs a Cypher query that:
    1. Matches the class node `c` whose `qualified_name` equals `$id`.
    2. Traverses outgoing `INHERITS` relationships to each direct base class
       `base`.
    3. Uses `WITH DISTINCT base` to deduplicate base classes in case of
       duplicated edges.
    4. Returns:
       - `label`: the literal string "Class".
       - `id`: `base.qualified_name` if present, otherwise `base.name`.
       - `name`: `base.name` if present, otherwise `base.qualified_name`.
    5. Orders results by `name` (ascending) and limits the number of rows to
       `$limit`.

    Notes:
    - This lists only immediate parents (no transitive ancestry). To explore the
      full hierarchy, use a variable-length `INHERITS` traversal.
    - Results are normalized via `coalesce` to support nodes that may have only
      `qualified_name` or only `name`.
    """
    q = """
    MATCH (c:Class {qualified_name:$id})-[:INHERITS]->(base)
    WITH DISTINCT base
    RETURN 'Class' AS label,
           coalesce(base.qualified_name, base.name) AS id,
           coalesce(base.name, base.qualified_name) AS name
    ORDER BY name ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"id": node_id, "limit": limit})

# -- OVERRIDDEN_BY -------------------------------------------------------------
@register("OVERRIDDEN_BY")
def overridden_by_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """List immediate subclasses of a given class by inheritance.

    Args:
      step: Plan step containing parameters (``id``, optional ``limit``).
      intent: Parsed query intent providing defaults.
      resolved: Resolution results that may include ``resolved_id``.

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for subclasses.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("overridden_by_adapter: id=%s limit=%s", node_id, limit)


    """
    This runs a Cypher query that:
    1. Matches the base class node `base` whose `qualified_name` equals `$id`.
    2. Traverses incoming `INHERITS` relationships from each subclass `sub`
       to that base class.
    3. Uses `WITH DISTINCT sub` to deduplicate subclasses in case of duplicate
       edges.
    4. Returns:
       - `label`: the literal string "Class".
       - `id`: `sub.qualified_name`.
       - `name`: `sub.name`.
    5. Orders results by `name` (ascending) and limits the number of rows to
       `$limit`.

    Notes:
    - This lists only immediate subclasses (no transitive descendants). To
      include the full hierarchy, use a variable-length `INHERITS` traversal.
    - Method-level overrides are not included here; this query returns classes
      only.
    """
    q = """
    MATCH (base:Class {qualified_name:$id})<-[:INHERITS]-(sub:Class)
    WITH DISTINCT sub
    RETURN 'Class' AS label, sub.qualified_name AS id, sub.name AS name
    ORDER BY name ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"id": node_id, "limit": limit})

# -- METHODS_OF_CLASS ----------------------------------------------------------
@register("METHODS_OF_CLASS")
def methods_of_class_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """List methods defined directly by a class.

    Args:
      step: Plan step containing parameters (``id``, optional ``limit``).
      intent: Parsed query intent providing defaults.
      resolved: Resolution results that may include ``resolved_id``.

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for methods.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("methods_of_class_adapter: id=%s limit=%s", node_id, limit)

    """
    This runs a Cypher query that:
    1. Matches the class node `c` whose `qualified_name` equals `$id`.
    2. Traverses outgoing `DEFINES_METHOD` relationships to each method `m`
       defined directly by that class.
    3. Returns:
       - `label`: the literal string "Method".
       - `id`: `m.qualified_name`.
       - `name`: `m.name`.
    4. Orders results by `name` (ascending) and limits the number of rows to
       `$limit`.

    Notes:
    - Only methods declared on the class are returned (no inherited methods).
    - To include inherited/overridden methods across the hierarchy, combine this
      with `INHERITS` traversals and method resolution logic.
    """
    q = """
    MATCH (c:Class {qualified_name:$id})-[:DEFINES_METHOD]->(m:Method)
    RETURN 'Method' AS label, m.qualified_name AS id, m.name AS name
    ORDER BY name ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"id": node_id, "limit": limit})

# -- MODULES_DEPENDING_ON_EXTERNAL --------------------------------------------
@register("MODULES_DEPENDING_ON_EXTERNAL")
def modules_dep_ext_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """List modules that depend on a given external package.

    The package name is expected in normalized (lowercase, PEP 503) form.

    Args:
      step: Plan step containing parameters (``package``, optional ``limit``).
      intent: Parsed query intent providing defaults.
      resolved: Resolution results (unused here but part of the interface).

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for modules.
    """
    package = (step.params.get("package") or intent.mention or "").lower()
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("modules_dep_ext_adapter: package=%s limit=%s", package, limit)

    """
    This runs a Cypher query that:
    1. Matches each module `m` that has a `DEPENDS_ON_EXTERNAL` edge to an
       external package `p` with `p.name = $package`.
    2. Returns:
       - `label`: the literal string "Module".
       - `id`: `m.qualified_name`.
       - `name`: `m.name`.
    3. Orders results by `id` (ascending) and limits the number of rows to
       `$limit`.

    Notes:
    - The provided `package` is lowercased to align with PEP 503-normalized
      names stored on `ExternalPackage.name`.
    - If duplicate edges exist, consider adding `WITH DISTINCT m` before the
      RETURN to deduplicate modules.
    """
    q = """
    MATCH (m:Module)-[:DEPENDS_ON_EXTERNAL]->(p:ExternalPackage {name:$package})
    RETURN 'Module' AS label, m.qualified_name AS id, m.name AS name
    ORDER BY id ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"package": package, "limit": limit})

# -- NODE_META (batch by ids) --------------------------------------------------
@register("NODE_META")
def node_meta_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """Fetch basic metadata for a list of node ids.

    Accepts ``qualified_name`` values for code nodes and ``path`` for files.

    Args:
      step: Plan step containing parameter ``ids`` (str or list[str]).
      intent: Parsed query intent (unused, reserved for interface parity).
      resolved: Resolution results (unused, reserved for interface parity).

    Returns:
      list[dict]: Rows with ``label``, ``id``, and ``name`` for each id.
    """
    ids = step.params.get("ids") or []
    if not isinstance(ids, list):
        ids = [ids]
    ids = [str(x) for x in ids if x]
    if not ids:
        return []
    log.debug("node_meta_adapter: ids=%s", ids)

    """
    This runs a Cypher query that:
    1. Unwinds the parameter list `$ids` so each element `t` is processed in a
       single batch query (`UNWIND $ids AS t`).
    2. For each `t`, executes a subquery with multiple UNION ALL branches to
       try matching a node by:
         - Module/Class/Function/Method via `qualified_name = t`
         - File via `path = t`
       Each branch starts with `WITH t` to keep `t` in scope.
    3. Normalizes output to `(label, id, name)`:
         - `id`: `qualified_name` for code entities, `path` for files
         - `name`: `name` when present, otherwise `path` for files
    4. Orders rows by `label` then `id` (ascending) for stable results.

    Notes:
    - Duplicates in `$ids` will yield duplicate rows; use
      `UNWIND DISTINCT $ids AS t` or add `DISTINCT` to deduplicate if needed.
    - UNION ALL preserves all matches; if schema allowed overlapping matches,
      multiple rows could be returned for the same `t`.
    - Ensure indexes on `(qualified_name)` for code nodes and `(path)` for files
      to keep lookups fast on Memgraph.
    """
    q = """
    UNWIND $ids AS t
    CALL {
      WITH t
      MATCH (n:Module {qualified_name:t}) RETURN 'Module' AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      WITH t
      MATCH (n:Class {qualified_name:t}) RETURN 'Class' AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      WITH t
      MATCH (n:Function {qualified_name:t}) RETURN 'Function' AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      WITH t
      MATCH (n:Method {qualified_name:t}) RETURN 'Method' AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      WITH t
      MATCH (n:File {path:t}) RETURN 'File' AS label, n.path AS id, n.path AS name
    }
    RETURN label, id, name
    ORDER BY label ASC, id ASC
    """
    return run_cypher_query(q, {"ids": ids})


# -- ENTRY_FUNCS_BY_KEYWORD --------------------------------------------------
@register("ENTRY_FUNCS_BY_KEYWORD")
def entry_funcs_by_keyword_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """Find functions/methods by keyword across names and qualified names.

    Performs case-sensitive and case-insensitive matching on `name` and
    `qualified_name` for functions and methods. Includes exact, prefix, and
    qualified-name suffix checks, and conditionally enables substring search
    for keywords with length >= 3.

    Args:
      step: Plan step containing parameters (``kw``, optional ``limit``).
      intent: Parsed query intent providing defaults (may supply ``mention``).
      resolved: Resolution results (unused; kept for interface parity).

    Returns:
      list[dict]: Rows labeled as ``Function`` or ``Method`` with ``id`` and
        ``name``. Ordered by label then id, limited by ``limit``.
    """
    kw = (step.params.get("kw") or intent.mention or "").strip()
    if not kw:
        return []
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    kwl = kw.lower()
    use_contains = len(kwl) >= 3
    log.debug("entry_funcs_by_keyword_adapter: kw=%s kwl=%s limit=%s use_contains=%s", kw, kwl, limit, use_contains)

    """
    This runs a Cypher query that:
    1. Binds `$kw` and lowercase `$kwl` for matching.
    2. Searches Functions and Methods in two UNION ALL branches using:
       - Exact name, prefix (STARTS WITH), and qname suffix (ENDS WITH '.'+kw).
       - Case-insensitive variants via toLower(...).
       - Optional substring (CONTAINS) when len(kw) >= 3 to control scan cost.
    3. Applies per-branch ORDER BY and LIMIT, then merges and applies a final
       ORDER BY and LIMIT.

    Notes:
    - LIMIT is parameterized for consistency with other adapters.
    - Consider indexes on (name, qualified_name) for :Function and :Method.
    """
    q = f"""
    WITH $kw AS kw, $kwl AS kwl
    CALL {{
      WITH kw, kwl
      MATCH (f:Function)
      WHERE f.name = kw
         OR f.name STARTS WITH kw
         OR f.qualified_name ENDS WITH ('.' + kw)
         OR toLower(f.name) = kwl
         OR toLower(f.name) STARTS WITH kwl
         OR toLower(f.qualified_name) ENDS WITH ('.' + kwl)
         {"OR toLower(f.name) CONTAINS kwl OR toLower(f.qualified_name) CONTAINS kwl" if use_contains else ""}
      RETURN 'Function' AS label, f.qualified_name AS id, f.name AS name
      ORDER BY id ASC
      LIMIT $limit
      UNION ALL
      WITH kw, kwl
      MATCH (m:Method)
      WHERE m.name = kw
         OR m.name STARTS WITH kw
         OR m.qualified_name ENDS WITH ('.' + kw)
         OR toLower(m.name) = kwl
         OR toLower(m.name) STARTS WITH kwl
         OR toLower(m.qualified_name) ENDS WITH ('.' + kwl)
         {"OR toLower(m.name) CONTAINS kwl OR toLower(m.qualified_name) CONTAINS kwl" if use_contains else ""}
      RETURN 'Method' AS label, m.qualified_name AS id, m.name AS name
      ORDER BY id ASC
      LIMIT $limit
    }}
    RETURN label, id, name
    ORDER BY label ASC, id ASC
    LIMIT $limit
    """
    return run_cypher_query(q, {"kw": kw, "kwl": kwl, "limit": limit})