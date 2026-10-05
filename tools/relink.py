#!/usr/bin/env python3
"""Move (or rename) repo files and rewrite every reference to them, repo-wide.

    tools/relink.py [--mv] [--exclude PATH ...] OLD NEW [OLD NEW ...]
    tools/relink.py [--mv] --map pairs.tsv

Run from the repo root. For each OLD -> NEW pair (paths relative to the repo
root; a directory pair moves everything under it):

* with --mv, `git mv OLD NEW` (history is kept; parent dirs are created);
* every relative Markdown link ([text](target) and reference definitions
  `[id]: target`) in every tracked *.md file is re-resolved against the
  file's OLD location and re-expressed relative to its NEW location, so links
  *from* moved files and links *to* moved files both stay correct;
* repo-root path mentions (e.g. `research/<name>.md` in prose, code comments,
  scripts) in tracked *.md, *.rs, *.py, *.sh, *.toml, *.yml files are
  replaced, except in files given with --exclude (e.g. files that open
  branches are editing, to avoid merge conflicts).

Use tools/check_links.py afterwards to confirm nothing dangles.
"""
import os
import re
import subprocess
import sys

LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
REFDEF = re.compile(r"^(\s{0,3}\[[^\]]+\]:\s*)(\S+)(.*)$", re.M)
TEXT_EXT = (".md", ".rs", ".py", ".sh", ".toml", ".yml", ".yaml")


def git_files():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def map_path(p, pairs):
    """Map a repo-relative path through the OLD -> NEW pairs (file or dir prefix)."""
    p = os.path.normpath(p)
    for old, new in pairs:
        if p == old:
            return new
        if p.startswith(old + "/"):
            return new + p[len(old):]
    return p


def is_external(t):
    return re.match(r"^[a-z][a-z0-9+.-]*:", t, re.I) or t.startswith("#") or t.startswith("/")


def rewrite_target(t, old_file, new_file, pairs):
    if is_external(t):
        return t
    path, sep, frag = t.partition("#")
    if not path:
        return t
    trailing = "/" if path.endswith("/") else ""
    abs_old = os.path.normpath(os.path.join(os.path.dirname(old_file), path))
    if abs_old.startswith(".."):
        return t
    abs_new = map_path(abs_old, pairs)
    if abs_new == abs_old and old_file == new_file:
        return t
    rel = os.path.relpath(abs_new, os.path.dirname(new_file) or ".")
    if rel == ".":
        rel = "./"
    rel = rel + trailing if not rel.endswith("/") else rel
    return rel + (sep + frag if sep else "")


def keep_if_already_new(m, new):
    """Replace a mention unless it is already the tail of NEW (idempotence when
    NEW ends with OLD, e.g. foo -> tools/foo)."""
    s, i, old = m.string, m.start(), m.group(0)
    if new.endswith(old) and s[max(0, i - (len(new) - len(old))):m.end()] == new:
        return old
    return new


def main(argv):
    mv = False
    excl = set()
    pairs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--mv":
            mv = True
        elif a == "--exclude":
            i += 1
            excl.add(os.path.normpath(argv[i]))
        elif a == "--map":
            i += 1
            for line in open(argv[i]):
                if line.strip() and not line.startswith("#"):
                    o, n = line.split()[:2]
                    pairs.append((os.path.normpath(o), os.path.normpath(n)))
        else:
            pairs.append((os.path.normpath(a), os.path.normpath(argv[i + 1])))
            i += 1
        i += 1
    if not pairs:
        sys.exit(__doc__)
    # longest OLD first so nested pairs win over directory prefixes
    pairs.sort(key=lambda p: -len(p[0]))

    before = git_files()
    if mv:
        for old, new in pairs:
            if not os.path.exists(old):
                sys.exit(f"missing: {old}")
            os.makedirs(os.path.dirname(new) or ".", exist_ok=True)
            subprocess.run(["git", "mv", old, new], check=True)

    # new path -> old path for every tracked file (moved or not)
    moved_back = {map_path(p, pairs): p for p in before}
    mention_res = [
        (re.compile(r"(?<![\w.-])" + re.escape(old) + r"(?![\w-])"), new) for old, new in pairs
    ]
    changed = 0
    for new_file in git_files():
        if not new_file.endswith(TEXT_EXT) or not os.path.isfile(new_file):
            continue
        old_file = moved_back.get(new_file, new_file)
        try:
            src = open(new_file, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        txt = src
        if new_file.endswith(".md"):
            txt = LINK.sub(lambda m: m.group(1) + rewrite_target(m.group(2), old_file, new_file, pairs) + m.group(3), txt)
            txt = REFDEF.sub(lambda m: m.group(1) + rewrite_target(m.group(2), old_file, new_file, pairs) + m.group(3), txt)
        if os.path.normpath(new_file) not in excl and new_file != "tools/relink.py":
            for rx, new in mention_res:
                txt = rx.sub(lambda m, new=new: keep_if_already_new(m, new), txt)
        if txt != src:
            open(new_file, "w", encoding="utf-8").write(txt)
            changed += 1
            print("rewrote", new_file)
    print(f"{len(pairs)} pair(s), {changed} file(s) rewritten", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
