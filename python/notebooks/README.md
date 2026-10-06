# Tutorial notebooks

Executed Jupyter versions of the guides in `../docs/` (simulate, qec, shor, analysis).
They are generated, not hand-edited: the markdown docs stay the single source of truth
(and are doctested in CI). Regenerate after editing a doc:

```sh
pip install qsimlab nbformat nbclient ipykernel pymatching stim
python build.py            # convert + execute
python build.py --no-exec  # convert only
```
