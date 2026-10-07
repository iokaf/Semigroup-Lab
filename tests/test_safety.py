"""Protections that matter when the app is hosted for many users."""
import threading
import time

import mpmath as mp
import pytest

from engine.expr import ExpressionError, parse
from engine.model import SemigroupModel
from engine.precision import MP_LOCK


@pytest.mark.parametrize("text", ["9^9^9^9", "(1+z)^100000", "10^1000000*z", "z^(10^5)", "2^(3^5)",
                                  "1" + "0" * 60 + "*z"])
def test_huge_inputs_are_rejected_quickly(text):
    t0 = time.time()
    with pytest.raises(ExpressionError):
        parse(text)
    assert time.time() - t0 < 1.0


@pytest.mark.parametrize("text", ["z^40 - 1", "(1-z^2)*(3-z)/4", "2^(-3)*z", "z^(1/3)", "(z-1)^2/2"])
def test_ordinary_inputs_still_parse(text):
    parse(text)


def test_concurrent_analyses_do_not_change_each_others_precision():
    """Engine code holds MP_LOCK while it uses mpmath; a computation holding the lock keeps its precision."""
    changed = []

    def computation_expecting_50_digits():
        for _ in range(40):
            with MP_LOCK, mp.workdps(50):
                for _ in range(10):
                    time.sleep(0.0005)  # let other sessions run
                    if mp.mp.dps != 50:
                        changed.append(mp.mp.dps)

    def other_session(p):
        for _ in range(3):
            SemigroupModel({"mode": "bp", "tau": "1", "p": p})

    threads = [threading.Thread(target=computation_expecting_50_digits)] + [
        threading.Thread(target=other_session, args=(p,))
        for p in ("sqrt((1+z)/(1-z))", "(1+z)/(1-z) + log(2-z)", "exp(z)+1")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not changed


def test_summary_is_cached_and_returned_as_a_copy():
    m = SemigroupModel({"mode": "generator", "G": "(1-z^2)*(3-z)/4"})
    a = m.summary()
    a["kind"] = "tampered"
    assert m.summary()["kind"] == "hyperbolic"
