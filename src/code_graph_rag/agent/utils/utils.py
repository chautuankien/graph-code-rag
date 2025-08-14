from __future__ import annotations

import os
from typing import Any
import mgclient

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)


def run_cypher_query(
    query: str,
    params: dict[str, Any] | None = None,
    host: str | None = "localhost",
    port: int | None = 7687,
) -> list[dict[str, Any]]:
    """Execute a parameterized Cypher query against Memgraph and return rows.

    WHY:
        Use parameters (e.g., $name) instead of string interpolation to avoid
        Cypher injection and to enable planner reuse.

    Args:
        query: Cypher with $parameters, e.g. "MATCH (n) WHERE n.name=$name RETURN n.name AS name".
        params: Mapping of parameter names to values, e.g. {"name": "main"}.
        host: Override Memgraph host (defaults to MEMGRAPH_HOST or "localhost").
        port: Override Memgraph port (defaults to MEMGRAPH_PORT or 7687).

    Returns:
        A list of dictionaries, one per row, keyed by returned column aliases.

    Raises:
        RuntimeError: If the query fails. The original exception is chained.

    Example:
        rows = run_cypher_query(
            "MATCH (f:Function) WHERE f.name=$name RETURN f.qualified_name AS id, f.name AS name",
            {"name": "main"},
        )
    """
    log.debug("run_cypher_query.params: %s", params)
    conn = mgclient.connect(host=host, port=port)
    cur = conn.cursor()
    try:
        cur.execute(query, params or {})
        # column names: be robust to driver variations (tuple vs object)
        desc = cur.description or ()
        colnames: list[str] = []
        for c in desc:
            name = getattr(c, "name", None)
            if name is None:
                try:
                    name = c[0]  # type: ignore[index]
                except Exception:
                    name = str(c)
            colnames.append(name)

        rows = cur.fetchall() or []
        # build list[dict] row-by-row
        return [dict(zip(colnames, row)) for row in rows]
    except Exception as e:  # pragma: no cover (you can refine specific exceptions)
        raise RuntimeError(f"Cypher execution failed: {e}\nQuery: {query}\nParams: {params}") from e
    finally:
        try:
            cur.close()
        finally:
            conn.close()

