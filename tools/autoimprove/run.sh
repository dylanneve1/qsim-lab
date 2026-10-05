#!/bin/sh
# Long-running worker: one queue item per python process, so edits to the
# harness take effect at the next item. touch ~/qsim-ai/STOP to end.
cd ~/qsim-ai
while [ ! -e STOP ]; do
  python3 harness/autoimprove.py queue --once
  sleep 5
done
