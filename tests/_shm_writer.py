"""Child entry point for the cross-process SharedTensor stress test.

A spawned child re-imports this module, not the test, and conftest.py does
not run there: install the same torch-free `fluxrt` shim before importing."""

import os
import sys
import time
import types

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if "fluxrt" not in sys.modules:
    _pkg = types.ModuleType("fluxrt")
    _pkg.__path__ = [os.path.join(_SRC, "fluxrt")]
    sys.modules["fluxrt"] = _pkg

import numpy as np  # noqa: E402

from fluxrt.utils.shared_tensor import SharedTensor  # noqa: E402


def write_alternating(name, shape, versioned, stop, writes, gap=0.0):
    """Write all-0x55 and all-0xAA frames until `stop`, `gap` seconds apart
    (0: back to back); count them in `writes`."""
    t = SharedTensor(shape, name=name)
    if not versioned:
        t._seq = None  # write the way the code did before the seqlock
    frames = [np.full(shape, 0x55, np.uint8), np.full(shape, 0xAA, np.uint8)]
    i = 0
    while not stop.is_set():
        t.copy_from(frames[i & 1])
        i += 1
        writes.value = i
        if gap:
            time.sleep(gap)
    t.close()
