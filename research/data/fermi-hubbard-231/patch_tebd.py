import re

with open('run_tebd.py', 'r') as f:
    content = f.read()

# Add renorm=1 and norm=psi.norm() normalization before measure
content = content.replace(
    "tebd.split_opts = {'max_bond': chi_max, 'cutoff': 1e-10}",
    "tebd.split_opts = {'max_bond': chi_max, 'cutoff': 1e-10, 'renorm': 1}"
)

# Also normalize psi before measuring
measure_func = """
    def measure(psi):
        psi.normalize()
        # We only need site 15 for our analysis
"""
content = content.replace("    def measure(psi):\n        # We only need site 15 for our analysis", measure_func)

with open('run_tebd.py', 'w') as f:
    f.write(content)
