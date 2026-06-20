"""
WebRTC connection/lifecycle logic for the FluxRT streaming server.

Deliberately torch-free so the input-ownership state machine, the per-peer
recv policy, and the subprocess-reap helper can be unit-tested on a laptop
without the diffusion model (or a GPU). scripts/run_webrtc.py wires these
together with aiortc / FastAPI / the StreamProcessor.
"""
