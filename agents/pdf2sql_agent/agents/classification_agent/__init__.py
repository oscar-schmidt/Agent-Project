"""
Shim package that maps `agents.pdf2sql_agent.src`
to the local `src` namespace in this repository.
"""

from importlib import import_module
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

_src_package = import_module("src")
sys.modules.setdefault(__name__ + ".src", _src_package)

