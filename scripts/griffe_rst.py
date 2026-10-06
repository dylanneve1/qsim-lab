"""Griffe extension for the docs build: render the reST bits of qsimlab's docstrings as Markdown.

The docstrings are written for both ``help()`` and Sphinx-style readers, so they use
``:func:`x``` roles and ``::`` literal blocks. mkdocstrings renders Markdown, so we rewrite
roles to inline code and literal blocks to fenced code before rendering. Docstrings in the
source are not changed.
"""
import re
import griffe

_ROLE = re.compile(r":(?:py:)?(?:func|class|meth|data|attr|mod|obj|exc|const):`~?([^`]+)`")


def _md(text: str) -> str:
    text = _ROLE.sub(lambda m: f"`{m.group(1).split('.')[-1] if '~' in m.group(0) else m.group(1)}`", text)
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("::") and not line.strip().startswith(">>>"):
            head = line.rstrip()[:-2].rstrip()
            out.append(head + (":" if head and not head.endswith(":") else ""))
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            indent = len(lines[i]) - len(lines[i].lstrip())
            block = []
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) >= indent):
                block.append(lines[i][indent:] if lines[i].strip() else "")
                i += 1
            while block and not block[-1]:
                block.pop()
            lang = "pycon" if block and block[0].startswith(">>>") else "text"
            out += ["", f"```{lang}", *block, "```", ""]
            continue
        if line.strip().startswith(">>>") and (not out or not out[-1].strip() or not out[-1].startswith("```")):
            # bare doctest paragraph: fence until the next blank line
            block = []
            while i < len(lines) and lines[i].strip():
                block.append(lines[i].strip() if not lines[i].startswith(" ") else lines[i].lstrip())
                i += 1
            out += ["```pycon", *block, "```"]
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


class RstToMarkdown(griffe.Extension):
    def on_instance(self, *, obj, **kwargs):
        if obj.docstring is not None:
            obj.docstring.value = _md(obj.docstring.value)
