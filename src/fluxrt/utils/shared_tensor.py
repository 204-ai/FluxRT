import contextlib
import time

import numpy as np
from multiprocessing import shared_memory

from typing import Union

# A versioned tensor carries a 64-byte trailer after its payload (cache-line
# aligned): a magic number and a sequence counter. The payload stays at offset
# 0 with the same layout, so code that maps the segment by name with the old
# shape (np.ndarray(shape, buffer=shm.buf)) keeps working and simply ignores
# the trailer.
#
# The counter is a seqlock: even = stable, odd = a write is in progress.
# A writer makes it odd, copies, makes it even again; a reader copies between
# two reads of the counter and retries if they differ or were odd. Readers
# never block the writer and never see half of one frame and half of the
# next. Ordering relies on the platform's store/load order (x86-64 keeps
# stores in order and loads in order; glibc's and MSVC's memcpy fence their
# non-temporal stores), which is every machine FluxRT runs on.
_TRAILER = 64
_MAGIC = 0x4B4C51455354524F  # b"ORTSEQLK" read little-endian
# Bounded retries: a reader never waits for a stuck or concurrent writer for
# longer than this; it then returns its last copy (possibly torn) and says so.
# A writer that writes back to back without pause can starve a reader this
# way; FluxRT's writers write one frame per frame time (duty cycle ~2%).
_READ_ATTEMPTS = 16
_READ_DEADLINE_S = 0.02


def _trailer_offset(payload_bytes: int) -> int:
    return (payload_bytes + _TRAILER - 1) // _TRAILER * _TRAILER


class SharedTensor:
    def __init__(
        self,
        shape: np.shape,
        dtype: np.dtype = np.uint8,
        name: str = None,
        create: bool = False,
        versioned: bool = True,
    ):
        """
        Args:
            shape: tuple-like shape of the array.
            dtype: numpy dtype or something convertible (e.g. np.float32).
            name: name of existing shared memory (required if create=False).
            create: if True, create a new SharedMemory block.
            versioned: when creating, add the seqlock trailer (tear-free
                copy_from / to_numpy across processes). When attaching, the
                trailer is used if the segment has one.
        """
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape) * self.dtype.itemsize)
        trailer_at = _trailer_offset(self.size)

        if create:
            total = trailer_at + _TRAILER if versioned else self.size
            self.shm = shared_memory.SharedMemory(create=True, size=total)
            self.name = self.shm.name
            if versioned:
                words = np.ndarray((2,), dtype=np.uint64, buffer=self.shm.buf, offset=trailer_at)
                words[1] = 0
                words[0] = _MAGIC
        else:
            if name is None:
                raise ValueError("name must be provided when create=False")
            self.shm = shared_memory.SharedMemory(name=name)
            self.name = name

        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

        # The segment can be larger than requested (page rounding), so the
        # magic number, not the size alone, says whether a trailer is there.
        self._seq = None
        if self.shm.size >= trailer_at + _TRAILER:
            words = np.ndarray((2,), dtype=np.uint64, buffer=self.shm.buf, offset=trailer_at)
            if int(words[0]) == _MAGIC:
                self._seq = words[1:2]
        # Reads that had to retry because a write overlapped them (this process).
        self.torn_reads_avoided = 0
        self.last_read_sequence = -1

    @property
    def versioned(self) -> bool:
        return self._seq is not None

    def sequence(self) -> int:
        """The write counter (2 per completed write), or 0 when not versioned.
        Cheap: a change means a new frame was written."""
        return int(self._seq[0]) if self._seq is not None else 0

    @contextlib.contextmanager
    def writing(self):
        """Write the payload in place (e.g. cv2's dst=) as one seqlock write.
        One writer per tensor at a time."""
        seq = self._seq
        if seq is None:
            yield self.array
            return
        start = int(seq[0])
        # odd = writing; an odd start (a writer that died mid-write, or a
        # concurrent one) is moved on rather than flipped back to even
        odd = start + 1 if start % 2 == 0 else start + 2
        seq[0] = odd
        try:
            yield self.array
        finally:
            seq[0] = odd + 1

    def copy_from(self, tensor: Union["SharedTensor", np.ndarray]) -> None:
        """
        Copy data from another tensor into this shared tensor.

        Args:
            tensor: numpy ndarray or other SharedTensor
        """
        if isinstance(tensor, SharedTensor):
            src = tensor.to_numpy() if tensor.versioned else tensor.array
        else:
            src = np.asarray(tensor)

        if src.shape != self.shape:
            raise ValueError(
                f"Shape mismatch: source {src.shape} != target {self.shape}"
            )

        if src.dtype != self.dtype:
            src = src.astype(self.dtype, copy=False)

        with self.writing() as dst:
            np.copyto(dst, src)

    def read_into(self, out: np.ndarray, source: np.ndarray = None) -> bool:
        """Copy the payload (or `source`, a view into it such as one frame of
        a batch) into `out` without tearing. Returns False if a consistent
        copy was not obtained within the retry budget (out then holds the last
        attempt)."""
        src = self.array if source is None else source
        seq = self._seq
        if seq is None:
            np.copyto(out, src)
            return True
        deadline = None
        copies = 0
        while True:
            before = int(seq[0])
            if before % 2 == 0:
                np.copyto(out, src)
                if int(seq[0]) == before:
                    # which write this copy is (sequence() at the time)
                    self.last_read_sequence = before
                    return True
                copies += 1
            # Waiting out a write in progress is bounded by time only (a 2 MB
            # write takes ~0.3 ms, far longer than a yield); copies that a
            # write overlapped are bounded by count as well.
            self.torn_reads_avoided += 1
            now = time.perf_counter()
            if deadline is None:
                deadline = now + _READ_DEADLINE_S
            elif now > deadline or copies >= _READ_ATTEMPTS:
                return False  # out holds the last copy, possibly torn
            time.sleep(0)  # let the writer finish

    def to_numpy(self) -> np.ndarray:
        """
        Copies shared tensor to regulat numpy ndarray

        Returns:
            A regular numpy ndarray that is a copy of the shared tensor's data.
        """
        out = np.empty(self.shape, dtype=self.dtype)
        self.read_into(out)
        return out

    def close(self) -> None:
        self.array = None
        self._seq = None
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()

    def close_and_unlink(self) -> None:
        self.close()
        self.unlink()
