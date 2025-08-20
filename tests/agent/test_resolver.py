import pytest
from src.code_graph_rag.agent.models import QueryIntent
from src.code_graph_rag.agent import resolver as R

from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

@pytest.fixture(autouse=True)
def patch_db(monkeypatch):
    """Monkeypatch `run_cypher_query` to return deterministic fake rows.

    Why:
        - Decouple tests from Memgraph and the filesystem.
        - Precisely control resolver branches (exact qname, exact name,
          module suffix, broad contains) and their returned candidates.

    Behavior:
        - Uses the Cypher text to route to buckets in `fake` based on the
          parameter values, emulating the resolver's allow-listed queries.
    """
    fake = {
        "exact_qname:proj.app.main": [
            {"label":"Function","id":"proj.app.main","name":"main"},
        ],
        "exact_name:main": [
            {"label":"Method","id":"proj.core.App.main","name":"main"},
            {"label":"Function","id":"proj.app.main","name":"main"},
        ],
        "suffix:util": [
            {"label":"Module","id":"proj.pkg.util","name":"util.py"},
        ],
        "contains:handler": [
            {"label":"Function","id":"proj.web.request_handler","name":"request_handler"},
            {"label":"Method","id":"proj.web.RequestHandler.handle","name":"handle"},
        ],
    }

    def fake_run(cypher: str, params=None):
        params = params or {}
        if "coalesce(n.qualified_name, '') = $q" in cypher:
            return fake.get(f"exact_qname:{params.get('q')}", [])
        if "WHERE n.name = name" in cypher and "Function" in cypher:
            return fake.get(f"exact_name:{params.get('name')}", [])
        if "m.qualified_name ENDS WITH '.' + name" in cypher:
            return fake.get(f"suffix:{params.get('name')}", [])
        if "CONTAINS t" in cypher:
            return fake.get(f"contains:{params.get('t')}", [])
        return []

    monkeypatch.setattr(R, "run_cypher_query", fake_run)
    yield

def test_exact_qname_auto_accept():
    """Exact qname should auto-accept with high confidence and no assumption.

    Given ``mention='proj.app.main'``, the fake DB returns a Function node whose
    ``qualified_name`` exactly matches the mention. The resolver must:
      - set ``resolved_id`` to that qname,
      - produce ``confidence >= 0.80``, and
      - return no ``assumption`` message.
    """
    qi = QueryIntent(action="list_callees", mention="proj.app.main")
    res = R.resolve_entity(qi)
    log.debug("test_exact_qname_auto_accept.res: %s", res)

    assert res.resolved_id == "proj.app.main"
    assert res.confidence >= 0.80
    assert not res.assumption


def test_ambiguous_name_yields_candidates():
    """Ambiguous short name should NOT auto-accept; return candidates instead.

    Given ``mention='main'``, the fake DB returns both a Method and a Function
    sharing the same simple name. Although type boosts make their scores close,
    the win-margin guard prevents an overconfident auto-pick. The resolver must:
      - leave ``resolved_id`` as ``None``,
      - return at least two ``candidates``, and
      - provide an ``assumption`` message indicating ambiguity.
    """
    qi = QueryIntent(action="list_callers", mention="main")
    res = R.resolve_entity(qi)
    log.debug("test_ambiguous_name_yields_candidates.res: %s", res)

    assert res.resolved_id is None           # vì score có thể < 0.8
    assert len(res.candidates) >= 2
    assert res.assumption

def test_suffix_module_match():
    """Module stem/suffix should surface Module candidates for inspection.

    Given ``mention='util'``, the resolver queries Modules whose
    ``qualified_name`` ends with ``.util`` (or equals/name match). Expect the
    candidate list to include ``proj.pkg.util``.
    """
    qi = QueryIntent(action="imports", mention="util")
    res = R.resolve_entity(qi)
    log.debug("test_suffix_module_match.res: %s", res)

    assert any("proj.pkg.util" in c.id for c in res.candidates)

def test_contains_token_fuzzy():
    """Fuzzy CONTAINS fallback should still provide useful candidates.

    Given ``mention='handler'`` and ``mention_dst='handle'``, the resolver falls
    back to broad CONTAINS queries and ranks the results via token coverage.
    Expect at least one side (source or destination) to yield candidates or an
    ``assumption`` message, ensuring the Agent can proceed without exact
    matches.
    """
    qi = QueryIntent(action="trace_flow", mention="handler", mention_dst="handle")
    res = R.resolve_entity(qi)
    log.debug("test_contains_token_fuzzy.res: %s", res)

    assert len(res.candidates) > 0
    assert res.candidates_dst or res.assumption