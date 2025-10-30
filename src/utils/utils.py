from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "ensure_dir",
]
