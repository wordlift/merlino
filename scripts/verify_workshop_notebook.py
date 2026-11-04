"""Execute every workshop notebook cell in a fresh, CPU-only kernel.

This is a full-flow smoke test of the self-contained Colab artifact, not
just its repository modules. Optional Gemma and remote API calls stay off.
Run from the repository root:
    python scripts/verify_workshop_notebook.py
"""
from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = Path("notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb").resolve()


def main() -> int:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(NOTEBOOK)
    doc = nbformat.read(NOTEBOOK, as_version=4)
    if not any("USE_GEMMA = False" in "".join(c.source) for c in doc.cells if c.cell_type == "code"):
        raise AssertionError("Expected an explicitly disabled optional GPU mode.")
    for name in ("USE_WORDLIFT_RESOLVE = False", "USE_WOGRAPH_REMOTE = False"):
        if not any(name in "".join(c.source) for c in doc.cells if c.cell_type == "code"):
            raise AssertionError(f"Optional remote operation must be disabled by default: {name}")
    print(f"Python {sys.version.split()[0]}, {len(doc.cells)} cells, {NOTEBOOK}", flush=True)
    client = NotebookClient(
        doc, timeout=720, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": str(NOTEBOOK.parent.parent)}},
    )
    try:
        executed = client.execute()
    except CellExecutionError as exc:
        print("Workshop notebook failed during full CPU execution.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    code_count = sum(c.cell_type == "code" for c in executed.cells)
    errors = [(i, o) for i, c in enumerate(executed.cells) if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise AssertionError(f"Notebook has {len(errors)} execution errors: {errors}")
    print(f"PASS: {code_count} code cells completed, no errors; optional GPU and remote integrations disabled.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
