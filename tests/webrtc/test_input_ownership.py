"""Unit tests for InputOwnership — the input-steering state machine.

No aiortc / FastAPI / GPU. `pc` is any object identity.
"""

from fluxrt.webrtc.input_ownership import InputOwnership


def _pc(name):
    return type("PC", (), {"name": name})()


def test_single_peer_claims_and_activates():
    own = InputOwnership(has_server_camera=True)
    a = _pc("a")
    seq = own.register_waiter(a)
    assert own.try_claim(seq, a) is True
    assert own.owner_is(a) is True
    assert own.is_active() is True
    assert own.num_waiters() == 1


def test_only_oldest_seq_claims():
    own = InputOwnership()
    a, b = _pc("a"), _pc("b")
    sa = own.register_waiter(a)
    sb = own.register_waiter(b)
    assert sb > sa
    # b cannot claim while a (older) is waiting
    assert own.try_claim(sb, b) is False
    assert own.try_claim(sa, a) is True
    assert own.owner_is(a) is True
    # b still cannot claim while a owns
    assert own.try_claim(sb, b) is False


def test_release_clears_active_when_idle():
    own = InputOwnership(has_server_camera=True)
    a = _pc("a")
    sa = own.register_waiter(a)
    own.try_claim(sa, a)
    outcome = own.release(sa, a)
    assert outcome.had_owner is True
    assert outcome.became_idle is True
    assert outcome.server_camera_resumes is True
    assert own.is_active() is False
    assert own.num_waiters() == 0


def test_release_no_server_camera_does_not_resume():
    own = InputOwnership(has_server_camera=False)
    a = _pc("a")
    sa = own.register_waiter(a)
    own.try_claim(sa, a)
    outcome = own.release(sa, a)
    assert outcome.became_idle is True
    assert outcome.server_camera_resumes is False


def test_release_keeps_active_while_a_waiter_remains():
    own = InputOwnership()
    a, b = _pc("a"), _pc("b")
    sa = own.register_waiter(a)
    sb = own.register_waiter(b)
    own.try_claim(sa, a)
    # owner a leaves but waiter b is still registered
    outcome = own.release(sa, a)
    assert outcome.had_owner is True
    assert outcome.became_idle is False  # b remains
    assert own.is_active() is True  # NOT cleared — no server-camera flicker
    # b now becomes the oldest and can claim
    assert own.try_claim(sb, b) is True
    assert own.owner_is(b) is True


def test_handoff_to_oldest_waiter():
    own = InputOwnership()
    a, b, c = _pc("a"), _pc("b"), _pc("c")
    sa, sb, sc = (own.register_waiter(x) for x in (a, b, c))
    own.try_claim(sa, a)
    own.release(sa, a)
    # oldest remaining is b
    assert own.try_claim(sc, c) is False
    assert own.try_claim(sb, b) is True


def test_waiter_release_without_ownership_is_clean():
    own = InputOwnership()
    a, b = _pc("a"), _pc("b")
    sa = own.register_waiter(a)
    sb = own.register_waiter(b)
    own.try_claim(sa, a)
    # b never owned; releasing it just pops its seq, ownership of a untouched
    outcome = own.release(sb, b)
    assert outcome.had_owner is False
    assert own.owner_is(a) is True
    assert own.is_active() is True
    assert own.num_waiters() == 1
