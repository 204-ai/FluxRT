"""GPU power / clock / throttle readout for /healthz (src/fluxrt/utils/gpu_stats.py)."""

import ctypes

from fluxrt.utils import gpu_stats


class _FakeNvml:
    """NVML getters that write fixed values through their out-pointers."""

    def __init__(self, reasons):
        self.reasons = reasons

    def _set(self, value):
        def getter(handle, *args):
            ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_uint)).contents.value = value
            return 0

        return getter

    def __getattr__(self, name):
        values = {
            "nvmlDeviceGetPowerUsage": 84700,  # milliwatts
            "nvmlDeviceGetEnforcedPowerLimit": 85160,
            "nvmlDeviceGetClockInfo": 1230,
            "nvmlDeviceGetMaxClockInfo": 3090,
            "nvmlDeviceGetTemperature": 63,
        }
        if name in values:
            return self._set(values[name])
        raise AttributeError(name)

    def nvmlDeviceGetPowerManagementLimitConstraints(self, handle, low, high):
        ctypes.cast(high, ctypes.POINTER(ctypes.c_uint)).contents.value = 110000
        return 0

    def nvmlDeviceGetUtilizationRates(self, handle, out):
        ctypes.cast(out, ctypes.POINTER(gpu_stats._Utilization)).contents.gpu = 92
        return 0

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, handle, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_ulonglong)).contents.value = self.reasons
        return 0


def _with(monkeypatch, lib):
    monkeypatch.setattr(gpu_stats, "_lib", lib)
    monkeypatch.setattr(gpu_stats, "_handle", ctypes.c_void_p(1))
    monkeypatch.setattr(gpu_stats, "_failed", False)


# What the kiosk shows next to the fps: a GPU held at 85 of 110 W explains a
# slow frame; "idle" must not be reported as a throttle.
def test_reads_watts_clock_and_names_the_active_caps(monkeypatch):
    _with(monkeypatch, _FakeNvml(reasons=0x1 | 0x4 | 0x20))  # idle + sw power cap + sw thermal slowdown
    assert gpu_stats.read() == {
        "power_w": 84.7,
        "power_limit_w": 85.2,
        "clock_mhz": 1230,
        "clock_max_mhz": 3090,
        "temp_c": 63,
        "power_limit_max_w": 110.0,
        "util_pct": 92,
        "throttle": ["sw power cap", "sw thermal slowdown"],
    }


def test_no_nvml_is_none_not_an_error(monkeypatch):
    monkeypatch.setattr(gpu_stats, "_lib", None)
    monkeypatch.setattr(gpu_stats, "_handle", None)
    monkeypatch.setattr(gpu_stats, "_failed", True)
    assert gpu_stats.read() is None
