"""Regression test: each README benchmark row compares the same public API on both sides.

The VERIFY phase found the bench row "encode 200k" timing `encode_many` (Mojo)
against upstream's `encode` scalar loop, while the README row it feeds implies
upstream's scalar `encode` on both sides. These tests pin the paired helper that
the rows now use, so a row can no longer measure one entry point on one side and
a different one on the other.
"""

import pytest

import bench as bench_module
import geohash2
from geohash2 import geohash as implementation


@pytest.fixture()
def kernel_calls(monkeypatch):
    calls = []
    library = implementation.lib()
    for name in ("mgh_encode_batch", "mgh_decode_batch"):
        original = getattr(library, name)

        def counting(*arguments, _original=original, _name=name):
            calls.append(_name)
            return _original(*arguments)

        monkeypatch.setattr(library, name, counting)
    return calls


def _items(entry, arguments):
    if entry == "encode":
        latitude, longitude, precision = arguments
        return [(lat, lon, precision) for lat, lon in zip(latitude, longitude)]
    return [(value,) for value in arguments[0]]


@pytest.mark.parametrize(
    "entry, arguments",
    [
        ("encode", ((42.6, 57.64911), (-5.6, 10.40744), 5)),
        ("decode_exactly", (["ezs42", "u4pruydqqvj"],)),
    ],
)
def test_paired_scalar_row_runs_mojo_kernel_on_every_element(entry, arguments, kernel_calls):
    reference = bench_module.upstream()
    measured = bench_module.first_repetition(kernel_calls)
    (mojo_time, mojo_result), (python_time, python_result) = measured(entry, reference, *arguments)
    assert mojo_time > 0 and python_time > 0
    expected = [getattr(reference, entry)(*item) for item in _items(entry, arguments)]
    # Scalar API results, not numpy arrays: the row must not run a *_many batch instead.
    assert [type(value) for value in mojo_result] == [type(value) for value in expected]
    assert mojo_result == expected
    assert python_result == expected
    # One FFI dispatch per element in one repetition: two elements must not share a kernel call.
    assert kernel_calls == [f"mgh_{'encode' if entry == 'encode' else 'decode'}_batch"] * len(expected)


def test_scalar_encode_dispatches_one_row_to_the_batch_kernel(monkeypatch):
    """The scalar call is a batch of one; pin that so a Python fallback cannot pass parity."""

    class Recorder:
        def __init__(self, library):
            self.library = library
            self.calls = []

        def mgh_encode_batch(self, *arguments):
            self.calls.append(("encode", arguments))
            return self.library.mgh_encode_batch(*arguments)

        def mgh_decode_batch(self, *arguments):
            self.calls.append(("decode", arguments))
            return self.library.mgh_decode_batch(*arguments)

    recorder = Recorder(implementation.lib())
    monkeypatch.setattr(implementation, "lib", lambda: recorder)
    assert geohash2.encode(42.6, -5.6, 5) == "ezs42"
    assert geohash2.decode_exactly("ezs42")[0] == pytest.approx(42.60498046875)
    (encode_call, encode_args), (decode_call, decode_args) = recorder.calls
    assert encode_call == "encode" and encode_args[2] == 1
    assert decode_call == "decode" and decode_args[1] == 1
