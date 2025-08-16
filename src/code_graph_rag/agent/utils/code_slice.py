from pathlib import Path
from src.code_graph_rag.utils.logging_setup import get_logger
log = get_logger(__name__)

def load_code_slice(path: str, start: int, end: int,
                    before: int = 0, after: int = 0,
                    max_lines: int = 120,
                    repo_root: Path | None = None) -> str:
    """Read a bounded UTF-8 snippet from file with context lines.

    Args:
        path: Relative path to file (from repo root).
        start: Start line number of target entity (1-based).
        end: End line number of target entity (1-based).
        before: Context lines before start.
        after: Context lines after end.
        max_lines: Hard cap on total lines returned.
        repo_root: Root path of repository.

    Returns:
        UTF-8 text snippet (with replacement for undecodable bytes).
    """
    if not repo_root:
        repo_root = Path("src")  # default: repo's source folder
    abs_path = repo_root / Path(path)
    if not abs_path.exists():
        return ""

    lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    log.debug("load_code_slice.lines=%d", len(lines))
    log.debug("load_code_slice.lines=%s\n", lines[:10])  # log first 10 lines for debugging
    start_idx = max(0, start - 1 - before)
    end_idx = min(len(lines), end + after)
    log.debug("load_code_slice.start_idx=%d end_idx=%d", start_idx, end_idx)
    if end_idx - start_idx > max_lines:
        end_idx = start_idx + max_lines
    return "".join(lines[start_idx:end_idx])