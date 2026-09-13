"""Make experiments/exp_*.py importable from tests/.

The experiment scripts import each other's shared helper (`import figures`)
the way a script run as `python experiments/exp_foo.py` would resolve it --
via the script's own directory being first on sys.path. Tests import them
as modules instead, so put that directory on sys.path here rather than in
every test file that needs it.
"""

import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
