from __future__ import annotations

from pathlib import Path, PureWindowsPath


def to_posix_path(value: str | Path) -> str:
    """Normalize a path-like value to POSIX separators."""
    text = str(value)
    if text.startswith(("http://", "https://")):
        return text
    if "\\" in text:
        return PureWindowsPath(text).as_posix()
    return Path(text).as_posix()


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a path if needed."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_unlink(path: str | Path) -> None:
    """Remove a file path if it exists."""
    Path(path).unlink(missing_ok=True)
