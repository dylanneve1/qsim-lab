#!/usr/bin/env python3
"""Check that every relative link and repo-path mention in the repo resolves.

    tools/check_links.py [--anchors] [--mentions] [--strict]

Run from the repo root. Checks, over tracked files:

* links: every relative Markdown link `[text](target)` / `[id]: target` in a
  *.md file must point at a tracked file or directory (errors) -- untracked
  files (e.g. unpacked data) do not count, since they 404 on GitHub;
* --anchors: `file.md#heading` and `#heading` fragments must match a heading
  slug (GitHub rules) in the target (errors);
* --mentions: repo-root path mentions such as `research/shor/shor.md` or
  `examples/ghz.rs` in *.md / *.rs / *.py / *.sh files must exist (warnings,
  errors with --strict). Paths with glob characters or placeholders are
  skipped; mentions of compressed data are accepted if `<path>.xz` exists.

Exits non-zero if any error is found.
"""
import os
import re
import subprocess
import sys

LINK = re.compile(r"(?<!\\)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)", re.M)
FENCE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)
HEAD = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.M)
ROOTS = ("research", "examples", "src", "tests", "tools", "python", ".github")
MENTION = re.compile(
    r"(?<![\w./-])((?:" + "|".join(re.escape(r) for r in ROOTS) + r")/[\w./+-]*[\w/])"
)


def git_files():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p and os.path.exists(p)]


def slug(h):
    h = re.sub(r"`|\*\*|__|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", h)
    h = h.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h, flags=re.U)
    return h.replace(" ", "-")


_anchor_cache = {}


def anchors(path):
    if path not in _anchor_cache:
        try:
            txt = FENCE.sub("", open(path, encoding="utf-8").read())
        except (OSError, UnicodeDecodeError):
            txt = ""
        seen, out = {}, set()
        for h in HEAD.findall(txt):
            s = slug(h)
            n = seen.get(s, 0)
            out.add(s if n == 0 else f"{s}-{n}")
            seen[s] = n + 1
        out |= set(re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", txt))
        _anchor_cache[path] = out
    return _anchor_cache[path]


TRACKED = set()


def tracked(p):
    """A tracked file, or a directory containing one (untracked/ignored files don't count)."""
    return p in TRACKED


def exists(p):
    if tracked(p):
        return True
    if tracked(p + ".xz") or tracked(p + ".gz"):
        return True
    if p.startswith("examples/") and tracked(p + ".rs"):
        return True  # `examples/<name>` refers to the example binary
    # a prefix written as a pattern stem, e.g. `research/data/ge-shor/bench_`
    d, base = os.path.split(p)
    return base[-1:] in "_-" and os.path.isdir(d) and any(x.startswith(base) for x in os.listdir(d))


def main(argv):
    do_anchors = "--anchors" in argv
    do_mentions = "--mentions" in argv
    strict = "--strict" in argv
    errors, warnings, nlinks = [], [], 0
    files = git_files()
    for f in files:
        TRACKED.add(f)
        d = os.path.dirname(f)
        while d and d not in TRACKED:
            TRACKED.add(d)
            d = os.path.dirname(d)
    for f in files:
        if f.endswith(".md"):
            txt = FENCE.sub("", open(f, encoding="utf-8").read())
            # inline code spans can hold literal "](" — drop them for link parsing
            body = re.sub(r"`[^`\n]*`", "", txt)
            for t in LINK.findall(body) + REFDEF.findall(body):
                if re.match(r"^[a-z][a-z0-9+.-]*:", t, re.I):
                    continue
                nlinks += 1
                path, _, frag = t.partition("#")
                target = os.path.normpath(os.path.join(os.path.dirname(f), path)) if path else f
                if path and not tracked(target):
                    errors.append(f"{f}: broken link -> {t}")
                elif do_anchors and frag and target.endswith(".md") and frag not in anchors(target):
                    errors.append(f"{f}: missing anchor -> {t}")
        if do_mentions and f.endswith((".md", ".rs", ".py", ".sh")):
            try:
                txt = open(f, encoding="utf-8").read()
            except UnicodeDecodeError:
                continue
            for m in set(MENTION.findall(txt)):
                if re.search(r"[*?<>{}]|\.\.\.|XXX|_N\b", m) or m.endswith("/") and exists(m):
                    continue
                if not exists(m.rstrip("/.")):
                    warnings.append(f"{f}: mention of missing path {m}")
    for w in sorted(warnings):
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"checked {nlinks} links in {sum(f.endswith('.md') for f in files)} md files: "
          f"{len(errors)} error(s), {len(warnings)} warning(s)", file=sys.stderr)
    sys.exit(1 if errors or (strict and warnings) else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
