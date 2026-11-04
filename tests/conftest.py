import os
import sys
import types

_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "..", "src")

# Make scripts/ (batch_render, batch_routes) importable without installing them.
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

# batch_render imports fluxrt.utils (the parent-side crop). src/fluxrt/__init__.py
# pulls in torch/diffusers, so register a bare `fluxrt` package whose __path__ is
# the real src/fluxrt: submodules load from their real files, the heavy package
# __init__ never runs (same shim as tests/webrtc/conftest.py).
if "fluxrt" not in sys.modules:
    _pkg = types.ModuleType("fluxrt")
    _pkg.__path__ = [os.path.join(_SRC, "fluxrt")]
    sys.modules["fluxrt"] = _pkg

# Same for fluxrt.stream_processor, whose __init__ imports the torch-side
# StreamProcessor: its torch-free modules (the output scheduler) load from
# their real files.
if "fluxrt.stream_processor" not in sys.modules:
    _sp = types.ModuleType("fluxrt.stream_processor")
    _sp.__path__ = [os.path.join(_SRC, "fluxrt", "stream_processor")]
    sys.modules["fluxrt.stream_processor"] = _sp
