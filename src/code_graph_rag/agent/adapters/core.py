# src/code_graph_rag/agent/adapters/core.py
from __future__ import annotations
from typing import Any

from src.code_graph_rag.agent.models import QueryIntent, ResolvedEntity, PlanStep
from src.code_graph_rag.agent.adapters.registry import register
from src.code_graph_rag.agent.utils.utils import run_cypher_query
from src.code_graph_rag.utils.logging_setup import get_logger

log = get_logger(__name__)

# -- helper --------------------------------------------------------------------
def _coerce_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))

# -- META ----------------------------------------------------------------------
@register("META")
def meta_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    """Fetch canonical metadata for a node by qualified_name/path.

    Returns: rows[label,id,name,extra?] limited to 1.
    """
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    log.debug("meta_adapter: id=%s", node_id)
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
    """List top callers of a function/method/class by qualified_name."""
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
    """List top callees of a function/method/class by qualified_name."""
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
    """Outgoing imports from a module (by qualified_name)."""
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
    """k-hop undirected neighborhood around a node. Depth inserted (1..5)."""
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    depth = _coerce_int(step.params.get("depth", intent.depth), 1, 5, 2)
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("neighborhood_adapter: id=%s depth=%s limit=%s", node_id, depth, limit)

    # NOTE: variable-length upper bound can't be parameterized → safe integer insert
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
    """Direct parents of a class."""
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("inherits_direct_adapter: id=%s limit=%s", node_id, limit)

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
    """Classes/methods that override a given class/method."""
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("overridden_by_adapter: id=%s limit=%s", node_id, limit)

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
    """List methods defined by a class."""
    node_id = step.params.get("id") or resolved.resolved_id or intent.mention or ""
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("methods_of_class_adapter: id=%s limit=%s", node_id, limit)

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
    """Modules that depend on a given external package (normalized name)."""
    package = (step.params.get("package") or intent.mention or "").lower()
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    log.debug("modules_dep_ext_adapter: package=%s limit=%s", package, limit)

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
    """Fetch basic metadata for a list of node ids (qualified_name/path)."""
    ids = step.params.get("ids") or []
    if not isinstance(ids, list):
        ids = [ids]
    ids = [str(x) for x in ids if x]
    if not ids:
        return []
    log.debug("node_meta_adapter: ids=%s", ids)

    q = """
    UNWIND $ids AS t
    CALL {
      WITH t
      MATCH (n:Module {qualified_name:t}) RETURN 'Module' AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      MATCH (n:Class {qualified_name:t}) RETURN 'Class' AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      MATCH (n:Function {qualified_name:t}) RETURN 'Function' AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      MATCH (n:Method {qualified_name:t}) RETURN 'Method' AS label, n.qualified_name AS id, n.name AS name
      UNION ALL
      MATCH (n:File {path:t}) RETURN 'File' AS label, n.path AS id, n.path AS name
    }
    RETURN label, id, name
    ORDER BY label ASC, id ASC
    """
    return run_cypher_query(q, {"ids": ids})

# -- ENTRY_FUNCS_BY_KEYWORD --------------------------------------------------
@register("ENTRY_FUNCS_BY_KEYWORD")
def entry_funcs_by_keyword_adapter(*, step: PlanStep, intent: QueryIntent, resolved: ResolvedEntity) -> list[dict]:
    kw = (step.params.get("kw") or intent.mention or "").strip()
    if not kw:
        return []
    limit = _coerce_int(step.params.get("limit", intent.limit), 1, 200, 50)
    kwl = kw.lower()
    use_contains = len(kwl) >= 3
    log.debug("entry_funcs_by_keyword_adapter: kw=%s kwl=%s limit=%s", kw, kwl, limit)

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
      LIMIT {limit}
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
      LIMIT {limit}
    }}
    RETURN label, id, name
    ORDER BY label ASC, id ASC
    LIMIT {limit}
    """
    # ❗️Chỉ còn dùng kw/kwl làm params (không truyền limit)
    return run_cypher_query(q, {"kw": kw, "kwl": kwl})