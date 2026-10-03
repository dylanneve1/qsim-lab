#!/usr/bin/env python3
"""Decode an exported .stim circuit with Stim sampling + the Tesseract decoder (the decoder
Kishony-Fowler used), to cross-check our circuit against their published numbers and to
benchmark our BP+OSD. usage: tesseract_check.py file.stim shots [max_errors]"""
import sys, json, time
import sinter, stim
from tesseract_decoder import make_tesseract_sinter_decoders_dict
path, shots = sys.argv[1], int(sys.argv[2])
maxerr = int(sys.argv[3]) if len(sys.argv) > 3 else None
c = stim.Circuit.from_file(path)
t = time.time()
stats = sinter.collect(num_workers=int(sys.argv[4]) if len(sys.argv) > 4 else 2,
                       tasks=[sinter.Task(circuit=c, json_metadata={"f": path})],
                       decoders=["tesseract"], max_shots=shots, max_errors=maxerr,
                       custom_decoders=make_tesseract_sinter_decoders_dict())
for s in stats:
    print(json.dumps(dict(file=path, shots=s.shots, errors=s.errors, p_L=s.errors / s.shots,
                          seconds=round(time.time() - t, 1))))
