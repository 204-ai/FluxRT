"""The marker-aware jitter buffer (fluxrt.webrtc.jitter): a video frame goes
to the decoder with its last packet, not one frame later; gaps and lost
marker packets behave as in aiortc's buffer; the bytes are the same."""

from aiortc.jitterbuffer import JitterBuffer
from aiortc.rtp import RtpPacket

from fluxrt.webrtc import jitter

Marker = jitter.make_buffer_class()


def packet(seq, ts, marker=False, payload=None):
    p = RtpPacket(payload_type=96, sequence_number=seq, timestamp=ts, marker=int(marker))
    p._data = payload if payload is not None else bytes([seq % 256]) * 10
    return p


def frames_of(buffer, packets):
    """What add() hands the decoder after each packet."""
    out = []
    for p in packets:
        _, frame = buffer.add(p)
        out.append(None if frame is None else (frame.timestamp, frame.data))
    return out


# three frames of 3, 1 and 2 packets, markers on each frame's last packet
STREAM = [packet(100, 9000), packet(101, 9000), packet(102, 9000, True),
          packet(103, 12000, True),
          packet(104, 15000), packet(105, 15000, True),
          packet(106, 18000)]


def test_a_frame_goes_out_with_its_last_packet():
    got = frames_of(Marker(capacity=128, is_video=True), STREAM)
    assert [i for i, f in enumerate(got) if f] == [2, 3, 5]
    assert got[2][0] == 9000 and got[3][0] == 12000 and got[5][0] == 15000


def test_aiortc_s_buffer_waits_for_the_next_frame_and_hands_over_the_same_bytes():
    ours = [f for f in frames_of(Marker(capacity=128, is_video=True), STREAM) if f]
    got = frames_of(JitterBuffer(capacity=128, is_video=True), STREAM)
    assert [i for i, f in enumerate(got) if f] == [3, 4, 6]  # one packet of the next frame later
    assert [f for f in got if f] == ours


def test_a_gap_holds_the_frame_until_the_packet_arrives():
    b = Marker(capacity=128, is_video=True)
    assert frames_of(b, [packet(10, 3000), packet(12, 3000, True)]) == [None, None]
    _, frame = b.add(packet(11, 3000))  # the retransmission
    assert frame is not None and frame.timestamp == 3000 and len(frame.data) == 30


def test_a_frame_whose_marker_packet_was_lost_goes_out_when_the_next_starts():
    b = Marker(capacity=128, is_video=True)
    got = frames_of(b, [packet(20, 3000), packet(21, 3000), packet(22, 6000)])
    assert got[:2] == [None, None] and got[2][0] == 3000


def test_audio_buffers_are_aiortc_s():
    b, a = Marker(capacity=16, prefetch=4), JitterBuffer(capacity=16, prefetch=4)
    audio = [packet(i, 960 * i, True) for i in range(10)]
    assert frames_of(b, audio) == frames_of(a, audio)


def test_install_replaces_the_receiver_s_buffer(monkeypatch):
    from aiortc import rtcrtpreceiver

    monkeypatch.setattr(rtcrtpreceiver, "JitterBuffer", rtcrtpreceiver.JitterBuffer)
    cls = jitter.install()
    assert cls is not None and rtcrtpreceiver.JitterBuffer is cls
