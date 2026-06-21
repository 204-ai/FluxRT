"""Regression guard for the c855950 hang.

c855950 evicted the input owner on a blind 5s recv() timeout regardless of
connection state, freezing a healthy owner whose inter-frame gap exceeded 5s
(first keyframe, TURN/DTLS settle, brief stall). These assertions encode the
fix: an owner is released ONLY on a terminal connection state; a connected peer
is never gap-evicted.
"""

import pytest

from fluxrt.webrtc.input_ownership import (
    owner_should_release,
    waiter_should_evict,
)


def test_owner_not_released_when_connected():
    # THE regression guard: a healthy connected owner with a long frame gap stays.
    assert owner_should_release("connected") is False


@pytest.mark.parametrize("state", ["failed", "closed", "disconnected"])
def test_owner_released_only_on_terminal_state(state):
    assert owner_should_release(state) is True


def test_waiter_connected_is_never_evicted_even_past_deadline():
    assert (
        waiter_should_evict(
            connection_state="connected", got_first_frame=False, deadline_passed=True
        )
        is False
    )


def test_waiter_dead_and_past_deadline_is_evicted():
    assert (
        waiter_should_evict(
            connection_state="failed", got_first_frame=False, deadline_passed=True
        )
        is True
    )


def test_waiter_not_evicted_before_deadline():
    assert (
        waiter_should_evict(
            connection_state="failed", got_first_frame=False, deadline_passed=False
        )
        is False
    )


def test_waiter_with_a_frame_is_never_gap_evicted():
    # Once a frame arrived, the waiter is a real view-only peer; never drop it.
    assert (
        waiter_should_evict(
            connection_state="failed", got_first_frame=True, deadline_passed=True
        )
        is False
    )
