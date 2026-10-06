#!/usr/bin/env bash
# Assemble site-src/ from python/ so the docs live in one place (python/docs, python/API.md).
set -euo pipefail
rm -rf site-src && mkdir -p site-src
cp python/docs/*.md python/docs/*.png site-src/
cp python/API.md site-src/API.md
cp python/README.md site-src/index.md
cp python/notebooks/README.md site-src/notebooks.md
# Rewrite repo-relative links to site-relative ones.
sed -i 's#(\.\./API\.md#(API.md#g' site-src/*.md
sed -i 's#(docs/\([a-z_]*\.md\)#(\1#g; s#(API\.md#(API.md#g' site-src/index.md
sed -i 's#(docs/\([a-z_]*\.md\)#(\1#g' site-src/API.md
# Links outside python/ point at GitHub.
sed -i -E 's#\((\.\./)+(research/[^)]*)\)#(https://github.com/dylanneve1/qsim-lab/blob/main/\2)#g' site-src/*.md
