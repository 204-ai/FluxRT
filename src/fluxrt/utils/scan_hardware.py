import os
import platform
import subprocess
import torch


def _run(cmd):
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def get_cpu_name():
    system = platform.system()

    if system == "Windows":
        # wmic is gone from current Windows 11 builds; the registry always has the name.
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass

    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    return platform.processor() or "Unknown CPU"


def get_gpu_info():
    if not torch.cuda.is_available():
        return "No CUDA GPU"

    gpus = []

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        gpus.append(
            {
                "name": props.name,
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "cc": f"{props.major}.{props.minor}",
            }
        )

    return gpus


def scan_hardware():
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": get_cpu_name(),
        "cpu_cores_logical": os.cpu_count(),
        "gpu": get_gpu_info(),
    }
    return info
