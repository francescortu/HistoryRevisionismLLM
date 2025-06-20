import sys
from pathlib import Path

# Adjust the sys.path to point to the submodule's library
sys.path.insert(0, str(Path(__file__).resolve().parent / "easyroutine/easyroutine"))

import easyroutine.easyroutine as easyroutine
