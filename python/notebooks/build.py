"""Generate tutorial notebooks from python/docs/*.md and execute them.

Prose becomes markdown cells; ```python fences become code cells
(doctest-style `>>> ` / `... ` prompts are stripped, expected output dropped).
Usage: python notebooks/build.py [--no-exec]
"""
import pathlib, re, sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
FENCE = re.compile(r"^```(\w*)\s*$")


def strip_prompts(src: str) -> str:
    lines = src.splitlines()
    if not any(l.startswith(">>> ") or l == ">>>" for l in lines):
        return src
    out = []
    for l in lines:
        if l.startswith(">>> ") or l.startswith("... "):
            out.append(l[4:])
        elif l in (">>>", "..."):
            out.append("")
    return "\n".join(out)


def convert(md: pathlib.Path) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    buf, code, other = [], None, False
    def flush():
        text = "\n".join(buf).strip()
        if text:
            nb.cells.append(new_markdown_cell(text))
        buf.clear()
    for line in md.read_text().splitlines():
        m = FENCE.match(line)
        if code is None and not other and m and m.group(1) in ("python", "py", "pycon"):
            flush(); code = []
            continue
        if code is not None:
            if line.strip() == "```":
                src = strip_prompts("\n".join(code)).strip()
                if src:
                    nb.cells.append(new_code_cell(src))
                code = None
            else:
                code.append(line)
            continue
        if m:
            other = not other
        buf.append(line)
    flush()
    # repoint relative images
    for c in nb.cells:
        if c.cell_type == "markdown":
            c.source = re.sub(r"\]\((?!http)([^)]+\.png)\)", r"](../docs/\1)", c.source)
    return nb


def main():
    run = "--no-exec" not in sys.argv
    for md in (DOCS / f"{n}.md" for n in ("simulate", "qec", "shor", "analysis")):  # tutorials only
        nb = convert(md)
        if run:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(HERE)}}).execute()
        out = HERE / f"{md.stem}.ipynb"
        nbformat.write(nb, out)
        print(f"{out.name}: {sum(c.cell_type=='code' for c in nb.cells)} code cells")


if __name__ == "__main__":
    main()
