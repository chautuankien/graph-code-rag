// Uniqueness constraints (khóa tự nhiên)
CREATE CONSTRAINT ON (n:Project) ASSERT n.name IS UNIQUE;
CREATE CONSTRAINT ON (n:Package) ASSERT n.qualified_name IS UNIQUE;
CREATE CONSTRAINT ON (n:Module)  ASSERT n.qualified_name IS UNIQUE;
CREATE CONSTRAINT ON (n:Class)   ASSERT n.qualified_name IS UNIQUE;
CREATE CONSTRAINT ON (n:Function)ASSERT n.qualified_name IS UNIQUE;
CREATE CONSTRAINT ON (n:Method)  ASSERT n.qualified_name IS UNIQUE;
CREATE CONSTRAINT ON (n:Folder)  ASSERT n.path IS UNIQUE;
CREATE CONSTRAINT ON (n:File)    ASSERT n.path IS UNIQUE;
CREATE CONSTRAINT ON (n:ExternalPackage) ASSERT n.name IS UNIQUE;

// Label-property indexes (tăng tốc MATCH/MERGE)
CREATE INDEX ON :Project(name);
CREATE INDEX ON :Package(qualified_name);
CREATE INDEX ON :Module(qualified_name);
CREATE INDEX ON :Class(qualified_name);
CREATE INDEX ON :Function(qualified_name);
CREATE INDEX ON :Method(qualified_name);
CREATE INDEX ON :Folder(path);
CREATE INDEX ON :File(path);
CREATE INDEX ON :ExternalPackage(name);

// (tùy chọn) Label indexes nếu bạn lọc theo nhãn nhiều
CREATE INDEX ON :Module;
CREATE INDEX ON :Class;
