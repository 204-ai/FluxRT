"""Hand a received video frame to the decoder when its last packet arrives
(torch-free).

aiortc's JitterBuffer declares a frame complete only when a packet of the
NEXT frame arrives (a different RTP timestamp). It does not look at the RTP
marker bit, which VP8 (RFC 7741) and H.264 (RFC 6184) senders set on a
frame's last packet (browsers do, and aiortc's own sender does). So every
received frame waited one inter-frame interval before decoding: 33 ms at a
30 fps camera, ~60 ms at the 15-18 fps the kiosk sends at (it caps its uplink
at 1.5x the generated fps), added straight to the input's latency.

This buffer also completes a frame when the contiguous run of packets from
the oldest one ends in a marker packet. The decoder gets the same bytes as
before (the same packets, earlier); a lost or late packet still holds the
frame back until it arrives (NACK / RTX), as before.
"""

from __future__ import annotations

import logging

log = logging.getLogger("fluxrt.webrtc")


def make_buffer_class():
    """A JitterBuffer subclass completing frames on the marker bit, or None
    if this aiortc's buffer does not look like the one this was written
    against (1.x)."""
    from aiortc.jitterbuffer import JitterBuffer, JitterFrame

    probe = JitterBuffer(capacity=4, is_video=True)
    if not all(hasattr(probe, a) for a in ("_origin", "_packets", "_capacity", "_prefetch", "remove", "_remove_frame")):
        return None

    class MarkerJitterBuffer(JitterBuffer):
        def _remove_frame(self, sequence_number: int):
            if not self._is_video or self._prefetch:
                return super()._remove_frame(sequence_number)
            packets = []
            timestamp = None
            complete = False
            for count in range(self._capacity):
                packet = self._packets[(self._origin + count) % self._capacity]
                if packet is None:
                    break  # a gap: wait for it, as aiortc does
                if timestamp is None:
                    timestamp = packet.timestamp
                elif packet.timestamp != timestamp:
                    complete = True  # the next frame started: aiortc's rule
                    break
                packets.append(packet)
                if packet.marker:
                    complete = True  # the frame's last packet
                    break
            if not complete:
                return None
            self.remove(len(packets))
            return JitterFrame(data=b"".join(p._data for p in packets), timestamp=timestamp)

    return MarkerJitterBuffer


def install():
    """Use the marker-aware buffer for every aiortc receiver created after
    this. Returns the class, or None when aiortc's buffer is left in place."""
    cls = make_buffer_class()
    if cls is None:
        log.warning("jitter buffer: unexpected aiortc internals; keeping aiortc's")
        return None
    from aiortc import rtcrtpreceiver

    rtcrtpreceiver.JitterBuffer = cls
    log.info("jitter buffer: video frames decode on their last packet (RTP marker)")
    return cls
