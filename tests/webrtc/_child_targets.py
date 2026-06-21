"""Top-level (picklable) child entry points for the reap tests under spawn."""

import signal
import time


def busy_child(ignore_sigterm=False):
    if ignore_sigterm:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.05)
