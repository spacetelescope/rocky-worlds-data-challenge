"""Normalize notebook metadata for public rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

GENERIC_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3,
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}


def normalize_notebook(path: Path) -> bool:
    """Normalize one notebook's top-level kernel metadata.

    Parameters
    ----------
    path : pathlib.Path
        Notebook path to normalize.

    Returns
    -------
    bool
        `True` if the file changed.
    """
    notebook = json.loads(path.read_text())
    original_metadata = notebook.get("metadata", {}).copy()

    metadata = notebook.setdefault("metadata", {})
    metadata.update(GENERIC_METADATA)
    metadata["language_info"].pop("version", None)

    if metadata == original_metadata:
        return False

    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return True


def main(paths: list[str]) -> int:
    """Normalize notebook metadata for each path."""
    changed = False
    for path in paths:
        changed = normalize_notebook(Path(path)) or changed

    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
