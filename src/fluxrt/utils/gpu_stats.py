"""GPU power, clock and throttle state straight from NVML (no extra package).

On a laptop the driver moves the GPU's power limit and clock with the
platform's power and thermal state, and that sets the frame time as much as
the model does. `/healthz` reports it so the kiosk can show it next to the fps.

`read()` returns None when NVML is not available (no NVIDIA driver, CPU run).
"""

import ctypes
import sys

# nvmlClocksThrottleReasons bits (nvml.h)
_REASONS = {
    0x02: "applications clocks setting",
    0x04: "sw power cap",
    0x08: "hw slowdown",
    0x10: "sync boost",
    0x20: "sw thermal slowdown",
    0x40: "hw thermal slowdown",
    0x80: "hw power brake slowdown",
    0x100: "display clock setting",
}

_lib = None
_handle = None
_failed = False


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def _init(index: int) -> bool:
    global _lib, _handle, _failed
    if _handle is not None:
        return True
    if _failed:
        return False
    try:
        lib = ctypes.CDLL("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)) != 0:
            raise OSError("no such device")
        _lib, _handle = lib, handle
        return True
    except (OSError, AttributeError):
        _failed = True
        return False


def _uint(function: str, *args):
    """One unsigned value from an NVML getter, None if this GPU does not report it."""
    value = ctypes.c_uint()
    try:
        if getattr(_lib, function)(_handle, *args, ctypes.byref(value)) != 0:
            return None
    except AttributeError:
        return None
    return value.value


def read(index: int = 0):
    """Current state of GPU `index`, or None. Costs well under a millisecond."""
    if not _init(index):
        return None
    milliwatts = lambda v: None if v is None else round(v / 1000, 1)
    stats = {
        "power_w": milliwatts(_uint("nvmlDeviceGetPowerUsage")),
        "power_limit_w": milliwatts(_uint("nvmlDeviceGetEnforcedPowerLimit")),
        "clock_mhz": _uint("nvmlDeviceGetClockInfo", ctypes.c_int(0)),
        "clock_max_mhz": _uint("nvmlDeviceGetMaxClockInfo", ctypes.c_int(0)),
        "temp_c": _uint("nvmlDeviceGetTemperature", ctypes.c_int(0)),
        "power_limit_max_w": None,
        "util_pct": None,
        "throttle": [],
    }
    low, high = ctypes.c_uint(), ctypes.c_uint()
    if _lib.nvmlDeviceGetPowerManagementLimitConstraints(_handle, ctypes.byref(low), ctypes.byref(high)) == 0:
        stats["power_limit_max_w"] = milliwatts(high.value)
    utilization = _Utilization()
    if _lib.nvmlDeviceGetUtilizationRates(_handle, ctypes.byref(utilization)) == 0:
        stats["util_pct"] = utilization.gpu
    reasons = ctypes.c_ulonglong()
    if _lib.nvmlDeviceGetCurrentClocksThrottleReasons(_handle, ctypes.byref(reasons)) == 0:
        stats["throttle"] = [name for bit, name in _REASONS.items() if reasons.value & bit]
    return stats


if __name__ == "__main__":
    import json
    import time

    start = time.perf_counter()
    first = read()
    print(json.dumps(first), f"(first read {1000 * (time.perf_counter() - start):.1f} ms)")
    start = time.perf_counter()
    for _ in range(100):
        read()
    print(f"per read {10 * (time.perf_counter() - start):.3f} ms")
