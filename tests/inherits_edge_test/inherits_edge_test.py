class A: pass
class B(A): pass        # B → A (internal)
class C(A, list): pass  # C → A, C → list (list là built-in, nhưng cứ lưu tên raw)
from .base import Base
class D(Base): pass     # D → Base (external)
class E(pkg.Base): pass # E → pkg.Base (qualified, lưu raw)
