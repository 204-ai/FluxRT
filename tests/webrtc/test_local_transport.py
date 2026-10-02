"""Local transport (raw frames over a WebSocket): the frame codec and one
client session, driven with a fake socket. No network, no GPU."""

import asyncio
import logging

import numpy as np
import pytest

from fluxrt.webrtc import local_transport as lt
from fluxrt.webrtc.input_ownership import InputOwnership, consume_peer_input

log = logging.getLogger("test")


def test_frame_codec_round_trip():
    rgbx = np.arange(3 * 5 * 4, dtype=np.uint8).reshape(3, 5, 4)
    message = lt.pack_frame(rgbx, 7)
    assert len(message) == 12 + 3 * 5 * 4
    assert np.array_equal(lt.unpack_frame(message), rgbx)


@pytest.mark.parametrize(
    "message",
    [b"", b"FRT1", b"NOPE" + bytes(8) + bytes(16), lt.pack_frame(np.zeros((2, 2, 4), np.uint8), 0)[:-1]],
)
def test_a_message_that_is_not_a_whole_frame_is_rejected(message):
    with pytest.raises(ValueError):
        lt.unpack_frame(message)


# The pipeline's input path asks a frame for BGR; the browser sends RGBX.
def test_raw_frame_hands_the_input_path_bgr():
    rgbx = np.zeros((1, 2, 4), np.uint8)
    rgbx[0, 0] = (10, 20, 30, 255)
    bgr = lt.RawFrame(rgbx).to_ndarray(format="bgr24")
    assert bgr.shape == (1, 2, 3) and tuple(bgr[0, 0]) == (30, 20, 10)
    assert bgr.flags["C_CONTIGUOUS"]


class FakeSocket:
    """Scripted inbound messages; records what the server sends."""

    def __init__(self, inbound):
        self.inbound = asyncio.Queue()
        for message in inbound:
            self.inbound.put_nowait(message)
        self.texts, self.frames, self.accepted, self.closed = [], [], False, False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return await self.inbound.get()

    async def send_text(self, text):
        self.texts.append(text)

    async def send_bytes(self, data):
        self.frames.append(data)

    async def close(self, code=1000):
        self.closed = True


# One session end to end: the client's frame reaches the pipeline as BGR, its
# control message is handled, output frames come back as RGBX (each version
# once), broadcasts reach it, and everything is released when it leaves — a
# socket that stayed in the peer set would keep the input claimed forever.
def test_a_local_client_steers_the_pipeline_and_gets_the_output():
    async def scenario():
        ownership = InputOwnership(has_server_camera=False)
        received, handled, peers, channels = [], [], set(), set()
        output = {"version": 0, "rgb": None}

        async def sink(frame):
            received.append(frame.to_ndarray(format="bgr24"))

        rgbx_in = np.full((2, 3, 4), 255, np.uint8)
        rgbx_in[:, :, 0] = 200  # red
        socket = FakeSocket([{"type": "websocket.receive", "bytes": lt.pack_frame(rgbx_in, 0)}, {"type": "websocket.receive", "text": "seed:5"}])
        session = asyncio.ensure_future(
            lt.serve(
                socket,
                get_output=lambda: (output["version"], output["rgb"]),
                consume_input=lambda track, peer: consume_peer_input(track, peer, ownership, sink, log=log),
                on_ctrl=lambda text, channel: (handled.append(text), channel.send("ack:" + text)),
                on_open=lambda channel, peer: channel.send("state:steps:2"),
                peers=peers,
                channels=channels,
                log=log,
                poll=0.001,
            )
        )
        await asyncio.sleep(0.05)
        assert socket.accepted and len(peers) == 1 and len(channels) == 1
        assert ownership.is_active()  # this client drives the input
        assert len(received) == 1 and tuple(received[0][0, 0]) == (255, 255, 200)  # BGR
        assert handled == ["seed:5"]
        assert socket.texts == ["state:steps:2", "ack:seed:5"]

        output.update(version=1, rgb=np.full((2, 2, 3), 9, np.uint8))
        await asyncio.sleep(0.03)
        next(iter(channels)).send("state:prompt:x")  # a broadcast from elsewhere
        output.update(version=2, rgb=np.full((2, 2, 3), 7, np.uint8))
        await asyncio.sleep(0.03)
        assert len(socket.frames) == 2  # one message per version, none repeated
        frame = lt.unpack_frame(socket.frames[1])
        assert frame.shape == (2, 2, 4) and tuple(frame[0, 0]) == (7, 7, 7, 255)
        assert socket.texts[-1] == "state:prompt:x"

        socket.inbound.put_nowait({"type": "websocket.disconnect"})
        await asyncio.wait_for(session, timeout=1.0)
        assert not peers and not channels and socket.closed
        assert not ownership.is_active()  # input released for the next client

    asyncio.run(scenario())


def test_a_broken_frame_message_does_not_end_the_session():
    async def scenario():
        ownership = InputOwnership(has_server_camera=False)
        socket = FakeSocket([{"type": "websocket.receive", "bytes": b"garbage"}, {"type": "websocket.receive", "text": "steps:2"}])
        handled = []

        async def sink(frame):
            pass

        session = asyncio.ensure_future(
            lt.serve(
                socket,
                get_output=lambda: (0, None),
                consume_input=lambda track, peer: consume_peer_input(track, peer, ownership, sink, log=log),
                on_ctrl=lambda text, channel: handled.append(text),
                on_open=lambda channel, peer: None,
                peers=set(),
                channels=set(),
                log=log,
                poll=0.001,
            )
        )
        await asyncio.sleep(0.03)
        assert handled == ["steps:2"]
        socket.inbound.put_nowait({"type": "websocket.disconnect"})
        await asyncio.wait_for(session, timeout=1.0)

    asyncio.run(scenario())
