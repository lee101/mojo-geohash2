"""Benchmark Mojo's geohash kernels against upstream geohash2.

Rows are paired by API: the left side calls the same public function as the right
side, once per element of the same array, except for rows explicitly labelled
`*_many`, where the left side is the batch extension (Mojo only) and the right
side is upstream's per-element scalar loop over the identical data.
"""

from __future__ import annotations

import platform

import importlib.metadata
import importlib.util
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import geohash2  # noqa: E402


def upstream():
    package_dir = pathlib.Path(importlib.metadata.distribution("geohash2").locate_file("geohash2"))
    spec = importlib.util.spec_from_file_location(
        "_bench_upstream_geohash2", package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def timeit(function, repetitions=3):
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def _paired(entry, reference, *arguments):
    """Time scalar `entry` on both sides, one call per element of `arguments[0]`."""
    rows = _rows(arguments)
    mojo, mojo_result = timeit(lambda: [getattr(geohash2, entry)(*row) for row in rows])
    python, python_result = timeit(lambda: [getattr(reference, entry)(*row) for row in rows])
    return (mojo, mojo_result), (python, python_result)


def _rows(arguments):
    leading = tuple(argument for argument in arguments if isinstance(argument, (list, tuple, np.ndarray)))
    trailing = tuple(argument for argument in arguments if not isinstance(argument, (list, tuple, np.ndarray)))
    return [(*values, *trailing) for values in zip(*leading)]


def first_repetition(observer):
    """Return a `_paired` wrapper that records the Mojo-side kernel calls of run 1 only."""
    paired = _paired

    def measuring(entry, reference, *arguments):
        start = len(observer)
        result = paired(entry, reference, *arguments)
        del observer[start + len(_rows(arguments)):]
        return result

    return measuring


def row(kernel, mojo_seconds, reference_seconds, reference):
    speedup = reference_seconds / mojo_seconds
    print(
        f"| {kernel} | {mojo_seconds * 1e3:.2f} ms | {reference_seconds * 1e3:.2f} ms | "
        f"{speedup:.2f}x | {reference} |"
    )


def main():
    _upstream = upstream()
    rng = np.random.default_rng(2026)
    count = 200_000
    latitude = rng.uniform(-90.0, 90.0, count)
    longitude = rng.uniform(-180.0, 180.0, count)
    precision = 12

    geohash2.encode_many(latitude[:1], longitude[:1], precision)
    hashes = geohash2.encode_many(latitude, longitude, precision)
    assert hashes.tolist() == [_upstream.encode(lat, lon, precision) for lat, lon in zip(latitude, longitude)]

    print(f"Machine: {platform.platform()}")
    print(f"Data: {count:,} random WGS84 coordinates, precision {precision}; best of 3 runs")
    print("| kernel | mojo-geohash2 | geohash2 1.1 | speedup | reference |")
    print("| --- | ---: | ---: | ---: | --- |")

    mojo, python = _paired("encode", _upstream, latitude, longitude, precision)
    row("encode 200k (scalar API)", mojo[0], python[0], "scalar Python loop, same API")
    mojo, python = _paired("decode_exactly", _upstream, hashes)
    row("decode_exactly 200k (scalar API)", mojo[0], python[0], "scalar Python loop, same API")

    mojo, _ = timeit(lambda: geohash2.encode_many(latitude, longitude, precision))
    python, _ = timeit(lambda: [_upstream.encode(lat, lon, precision) for lat, lon in zip(latitude, longitude)])
    row("encode 200k (encode_many)", mojo, python, "scalar Python loop, same API")

    mojo, _ = timeit(lambda: geohash2.decode_many(hashes, exact=True))
    python, _ = timeit(lambda: [_upstream.decode_exactly(value) for value in hashes])
    row("decode_exactly 200k (decode_many)", mojo, python, "scalar Python loop, same API")


if __name__ == "__main__":
    main()
