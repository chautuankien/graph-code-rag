import mgclient

def run_cypher_query(query: str) -> list[dict]:
    conn = mgclient.connect(host="localhost", port=7687)
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [col.name for col in cursor.description]
    rows = cursor.fetchall() 

    return [dict(zip(columns, rows))]
