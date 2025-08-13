import mgclient, pathlib

ddl = pathlib.Path("db_bootstrap.cypher").read_text(encoding="utf-8")
# tách theo dấu ; để execute từng câu
statements = [s.strip() for s in ddl.split(";") if s.strip()]

conn = mgclient.connect(host="localhost", port=7687)
conn.autocommit = True # tự động commit sau mỗi câu lệnh
cur = conn.cursor()

for stmt in statements:
    try:
        cur.execute(stmt + ";")
    except Exception as e:
        # Constraint có thể "already exists" nếu chạy lại → bỏ qua
        if "already exists" not in str(e).lower():
            raise
conn.commit()
cur.close(); conn.close()
print("✅ Bootstrap done")
