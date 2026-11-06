"""Rebuild the source bundle embedded in the workshop notebook from the repository.

The notebook carries a base64 ZIP of the package, ontologies, shapes, tests and some
docs, so Colab needs no clone. This script refreshes every file already in that bundle
from the working tree, keeping the same file list, order and line width.
Run from the repository root:
    python scripts/rebuild_notebook_bundle.py
"""
from pathlib import Path
import base64
import io
import json
import re
import zipfile

NOTEBOOK = Path("notebooks/Merlino_Gemma4_AOOE_Workshop.ipynb")


def main() -> int:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    index = next(i for i, c in enumerate(nb["cells"]) if "EMBEDDED = (" in "".join(c["source"]))
    source = "".join(nb["cells"][index]["source"])
    start = source.index("EMBEDDED = (") + len("EMBEDDED = (")
    end = source.index("\n)\n")
    block = source[start:end]
    chunks = re.findall(r"'([^']*)'", block)
    width = len(chunks[0])
    indent = re.match(r"\n(\s*)'", block).group(1)
    old = zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(chunks))))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as new:
        for info in old.infolist():
            path = Path(info.filename)
            if not path.is_file():
                raise FileNotFoundError(f"Bundled file missing from the repository: {path}")
            entry = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = info.external_attr
            new.writestr(entry, path.read_bytes())

    encoded = base64.b64encode(buffer.getvalue()).decode()
    lines = [encoded[k:k + width] for k in range(0, len(encoded), width)]
    source = source[:start] + "\n" + "\n".join(f"{indent}'{line}'" for line in lines) + source[end:]
    nb["cells"][index]["source"] = source.splitlines(keepends=True)
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Rebuilt bundle: {len(old.namelist())} files from the repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
