"""Raw frames over one WebSocket, for a client on the same machine.

WebRTC between a browser and a server on one box is two VP8 encodes and two
decodes on the CPU for nothing: on the show laptop the server process spent a
third of its time encoding the output and another quarter decoding, converting
and resizing the input. Here both directions are uncompressed RGBX in binary
messages over the loopback, and the control messages of the `ctrl` DataChannel
are the socket's text messages, unchanged.

Binary message, both directions: a 12-byte header, then width * height * 4
bytes of RGBX rows (X is ignored on the way in, 255 on the way out).

    offset 0  4 bytes  b"FRT1"
    offset 4  uint16   width   (little endian)
    offset 6  uint16   height
    offset 8  uint32   sequence (output: the server's frame version)

The socket plays the parts of a peer connection that the rest of the server
looks at (input ownership, control broadcasts, the peer count), so a local
client and WebRTC clients can be connected at the same time.

Torch/aiortc/FastAPI-free to import: the frame codec and the session loop are
unit-tested with a fake socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct

import numpy as np

from fluxrt.webrtc.input_ownership import MediaStreamError

MAGIC = b"FRT1"
HEADER = struct.Struct("<4sHHI")
MAX_SIDE = 4096


def pack_frame(rgbx: np.ndarray, sequence: int) -> bytes:
    """One binary message for an (h, w, 4) uint8 frame."""
    height, width = rgbx.shape[:2]
    return HEADER.pack(MAGIC, width, height, sequence & 0xFFFFFFFF) + rgbx.tobytes()


def unpack_frame(data: bytes) -> np.ndarray:
    """The (h, w, 4) uint8 frame of a binary message (a view, no copy).
    ValueError when the message is not a whole frame."""
    if len(data) < HEADER.size:
        raise ValueError("short frame message")
    magic, width, height, _ = HEADER.unpack_from(data)
    if magic != MAGIC or not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE):
        raise ValueError("not a frame message")
    if len(data) != HEADER.size + width * height * 4:
        raise ValueError("frame message size does not match its header")
    return np.frombuffer(data, dtype=np.uint8, offset=HEADER.size).reshape(height, width, 4)


class RawFrame:
    """What the input path asks of an av.VideoFrame: `to_ndarray("bgr24")`."""

    def __init__(self, rgbx: np.ndarray):
        self.rgbx = rgbx

    def to_ndarray(self, format: str = "bgr24") -> np.ndarray:
        if format != "bgr24":
            raise ValueError(f"unsupported format {format}")
        return np.ascontiguousarray(self.rgbx[:, :, 2::-1])


class SocketTrack:
    """What input ownership asks of an inbound video track: `recv()`. Keeps
    only the newest frame (the pipeline reads the latest anyway)."""

    kind = "video"
    id = "local"

    def __init__(self):
        self._frame = None
        self._ended = False
        self._ready = asyncio.Event()

    def push(self, frame: RawFrame) -> None:
        self._frame = frame
        self._ready.set()

    def end(self) -> None:
        self._ended = True
        self._ready.set()

    async def recv(self) -> RawFrame:
        await self._ready.wait()
        if self._ended:
            raise MediaStreamError
        frame, self._frame = self._frame, None
        self._ready.clear()
        return frame


class SocketChannel:
    """What the control broadcast asks of a DataChannel: `send(text)`."""

    label = "ctrl"

    def __init__(self):
        self.outbox: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_running_loop()

    def send(self, message: str) -> None:
        self._loop.call_soon_threadsafe(self.outbox.put_nowait, message)


class SocketPeer:
    """What the peer set asks of a peer connection: a state and `close()`."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.connectionState = "connected"
        self._fluxrt_channels: set = set()
        self._fluxrt_consume_task = None

    async def close(self) -> None:
        if self.connectionState == "closed":
            return
        self.connectionState = "closed"
        with contextlib.suppress(Exception):
            await self.websocket.close()


async def serve(
    websocket,
    *,
    get_output,
    consume_input,
    on_ctrl,
    on_open,
    peers: set,
    channels: set,
    log,
    poll: float = 0.002,
) -> None:
    """One local client, from accept to disconnect.

    get_output()            -> (version, rgb array or None): the newest output frame
    consume_input(track, peer) -> coroutine that feeds the pipeline from `track`
                               (input ownership decides whether this client steers)
    on_ctrl(text, channel)  handles a control message
    on_open(channel, peer)  sends the state snapshot a new client needs
    """
    await websocket.accept()
    peer = SocketPeer(websocket)
    channel = SocketChannel()
    track = SocketTrack()
    peer._fluxrt_channels.add(channel)
    peers.add(peer)
    channels.add(channel)
    consume = asyncio.ensure_future(consume_input(track, peer))
    peer._fluxrt_consume_task = consume
    on_open(channel, peer)
    log.info("Local client connected")

    async def send_loop():
        sent = -1
        while True:
            while not channel.outbox.empty():
                await websocket.send_text(channel.outbox.get_nowait())
            version, rgb = get_output()
            if rgb is None or version == sent:
                await asyncio.sleep(poll)
                continue
            sent = version
            rgbx = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
            rgbx[:, :, :3] = rgb
            rgbx[:, :, 3] = 255
            await websocket.send_bytes(pack_frame(rgbx, version))

    sender = asyncio.ensure_future(send_loop())
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                on_ctrl(message["text"], channel)
            elif message.get("bytes") is not None:
                try:
                    track.push(RawFrame(unpack_frame(message["bytes"])))
                except ValueError as error:
                    log.warning("Local client: %s", error)
    except Exception as error:  # noqa: BLE001 — a dropped socket ends the session
        log.info("Local client gone: %s", type(error).__name__)
    finally:
        sender.cancel()
        track.end()
        consume.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await consume
        channels.discard(channel)
        peers.discard(peer)
        await peer.close()
        log.info("Local client disconnected")
