from pathlib import Path
import sys


def add_titok_root_to_path(titok_root: str) -> Path:
    root = Path(titok_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"TiTok root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
