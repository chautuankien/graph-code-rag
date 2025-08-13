"""Build and load the code knowledge graph into Memgraph.

This module parses a repository into nodes/edges, exports Cypher for
auditing/replay, and optionally loads the data into a running Memgraph
instance. It also supports an idempotent bootstrap of constraints/indexes.
"""

from __future__ import annotations

from pathlib import Path
import mgclient

from src.code_graph_rag.utils.logging_setup import get_logger 
from src.code_graph_rag.parser.ast_parser import ASTParser
from src.code_graph_rag.graph.exporter import export_to_cypher

log = get_logger(__name__)

def _exec_many(cur, statements: list[str]) -> None:
    """Execute multiple Cypher statements sequentially.

    Empty lines and lines starting with "//" are skipped. During DDL
    bootstrap, benign "already exists" errors are ignored to keep the
    operation idempotent (e.g., re-running index/constraint creation).

    Args:
        cur: An active Memgraph cursor.
        statements: List of Cypher statements to execute (each ends with ";").

    Raises:
        mgclient.Error: If a non-ignorable error occurs while executing a
            statement.
    """
    for stmt in statements:
        s = stmt.strip()
        if not s or s.startswith("//"):
            continue
        try:
            cur.execute(s)
        except Exception as ex:  # pragma: no cover
            # Why: Ignore idempotent DDL failures (e.g., index exists) so the
            # bootstrap step can be re-run safely without manual cleanup.
            msg = str(ex).lower()
            if "already exists" in msg or ("constraint" in msg and "exists" in msg):
                continue
            raise


def build_knowledge_graph_and_insert_db(
    repo_path: str,
    export_path: str = "graph_export.cypherl",
    host: str = "localhost",
    port: int = 7687,
    username: str | None = None,
    password: str | None = None,
    clear_db: bool = True,
    bootstrap_schema: bool = False,
    bootstrap_file: str | None = None,
) -> None:
    """Build the knowledge graph and load it into Memgraph.

    This function parses the given repository path into graph nodes/edges,
    exports a Cypher script for auditing or replay, and optionally loads the
    data into a Memgraph instance. When requested, it also performs an
    idempotent schema bootstrap (indexes/constraints) before data load.

    Args:
        repo_path: Root directory of the repository to parse.
        export_path: Path to write the Cypher script for data import.
        host: Memgraph hostname.
        port: Memgraph port.
        username: Optional Memgraph username.
        password: Optional Memgraph password.
        clear_db: If True, delete all existing nodes/edges before loading.
        bootstrap_schema: If True, create constraints/indexes prior to import.
        bootstrap_file: Optional path to a .cypher file containing DDL.

    Raises:
        OSError: If files cannot be read/written (e.g., export or bootstrap).
        mgclient.Error: If Memgraph commands fail (connection or execution).

    Example:
        build_knowledge_graph_and_insert_db(
            "./my_repo", "out.cypher", bootstrap_schema=True,
            bootstrap_file="db_bootstrap.cypher",
        )
    """
    repo = Path(repo_path).resolve()
    out = Path(export_path).resolve()
    log.debug(f"repo={repo}, export={out}")

    # 1) Parse repository into nodes and edges.
    parser = ASTParser(str(repo))
    nodes, edges = parser.parse()
    log.debug(f"Parsed {len(nodes)} nodes and {len(edges)} edges.")

    # 2) Export Cypher using natural keys for idempotent MERGE statements.
    export_to_cypher(nodes, edges, out)

    # 3) Connect to Memgraph using provided credentials.
    conn_args = {"host": host, "port": port}
    if username:
        conn_args["user"] = username
    if password:
        conn_args["password"] = password

    conn = mgclient.connect(**conn_args)
    cur = conn.cursor()

    # --- Schema bootstrap DDL (constraints/indexes) --------------------------------
    # When to set `bootstrap_schema`:
    #   - True: first-time setup on a fresh DB/volume OR when the schema changed
    #           (new/updated constraints or indexes). Provide `bootstrap_file` that
    #           contains the latest DDL (e.g., db_bootstrap.cypher).
    #   - False: normal/routine runs. DDL persists; recreating it is unnecessary.
    #
    # IMPORTANT:
    #   - Memgraph does not allow constraint/index manipulation in a multi-command
    #     transaction. Execute DDL with autocommit enabled, or commit after each
    #     statement to avoid errors like:
    #       "Constraint manipulation not allowed in multicommand transactions"
    #       "Index manipulation not allowed in multicommand transactions"
    #   - `clear_db=True` removes data but keeps constraints/indexes.
    #   - Keep DDL out of the exporter; exporter should produce only MERGE/MATCH MERGE
    #     statements for nodes/edges.
    if bootstrap_schema:
        if bootstrap_file:
            # Why: Memgraph requires DDL to run in autocommit; execute each
            # statement separately to avoid multi-command transaction issues.
            conn.autocommit = True
            ddl = Path(bootstrap_file).read_text(encoding="utf-8").split(";")
            ddl = [x.strip() + ";" for x in ddl if x.strip()]
            _exec_many(cur, ddl)
            conn.autocommit = False
        else:
            # Fallback: Create a minimal, idempotent set of indexes.
            _exec_many(
                cur,
                [
                    "CREATE INDEX IF NOT EXISTS ON :Project(name);",
                    "CREATE INDEX IF NOT EXISTS ON :Package(qualified_name);",
                    "CREATE INDEX IF NOT EXISTS ON :Module(qualified_name);",
                    "CREATE INDEX IF NOT EXISTS ON :Class(qualified_name);",
                    "CREATE INDEX IF NOT EXISTS ON :Function(qualified_name);",
                    "CREATE INDEX IF NOT EXISTS ON :Method(qualified_name);",
                    "CREATE INDEX IF NOT EXISTS ON :Folder(path);",
                    "CREATE INDEX IF NOT EXISTS ON :File(path);",
                    "CREATE INDEX IF NOT EXISTS ON :ExternalPackage(name);",
                ],
            )

    # 4) Optionally clear existing data before import (schema remains intact).
    if clear_db:
        cur.execute("MATCH (n) DETACH DELETE n")

    # 5) Load data from the exported Cypher script.
    for line in out.read_text(encoding="utf-8").splitlines():
        stmt = line.strip()
        if not stmt or stmt.startswith("//"):
            continue
        cur.execute(stmt)

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    # Example usage.
    build_knowledge_graph_and_insert_db(
        repo_path="./sample_repo",
        export_path="graph_export.cypherl",
        bootstrap_schema=False,
        bootstrap_file="db_bootstrap.cypher",
    )
