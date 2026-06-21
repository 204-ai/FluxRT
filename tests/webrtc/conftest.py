"""
Make src/fluxrt/webrtc/* importable on a laptop WITHOUT torch.

src/fluxrt/__init__.py does `from .stream_processor import *`, which imports
torch/diffusers. To unit-test the torch-free webrtc submodules we install a fake
top-level `fluxrt` package whose __path__ points at the real src/fluxrt, so
`import fluxrt.webrtc.input_ownership` loads the real submodule file but never
runs the heavy package __init__.
"""

import pathlib
import sys
import types

_HERE = pathlib.Path(__file__).resolve()
_TESTS_DIR = _HERE.parent
_REPO = _HERE.parents[2]
_SRC = _REPO / "src"

for _p in (str(_SRC), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Torch-free shim for the `fluxrt` package.
_pkg = types.ModuleType("fluxrt")
_pkg.__path__ = [str(_SRC / "fluxrt")]
sys.modules["fluxrt"] = _pkg


def test_no_torch_leaked():
    """Guard: importing the webrtc modules must not drag in torch."""
    import fluxrt.webrtc.input_ownership  # noqa: F401
    import fluxrt.webrtc.proc  # noqa: F401

    assert "torch" not in sys.modules, "a real fluxrt import leaked torch into the harness"
