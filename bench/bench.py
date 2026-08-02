"""Benchmark Mojo's batched geohash kernels against upstream geohash2."""

from __future__ import annotations

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


def row(kernel, mojo_seconds, reference_seconds, reference):
    speedup = reference_seconds / mojo_seconds
    print(
        f"| {kernel} | {mojo_seconds * 1e3:.2f} ms | {reference_seconds * 1e3:.2f} ms | "
        f"{speedup:.2f}x | {reference} |"
    )


def main():
    reference = upstream()
    rng = np.random.default_rng(2026)
    count = 200_000
    latitude = rng.uniform(-90.0, 90.0, count)
    longitude = rng.uniform(-180.0, 180.0, count)
    precision = 12

    geohash2.encode_many(latitude[:1], longitude[:1], precision)
    hashes = geohash2.encode_many(latitude, longitude, precision)

    print(f"Machine: {__import__('platform').platform()}")
    print(f"Data: {count:,} random WGS84 coordinates, precision {precision}; best of 3 runs")
    print("| kernel | mojo-geohash2 | geohash2 1.1 | speedup | reference |")
    print("| --- | ---: | ---: | ---: | --- |")

    mojo, _ = timeit(lambda: geohash2.encode_many(latitude, longitude, precision))
    python, _ = timeit(lambda: [reference.encode(lat, lon, precision) for lat, lon in zip(latitude, longitude)])
    row("encode 200k", mojo, python, "scalar Python loop")

    mojo, _ = timeit(lambda: geohash2.decode_many(hashes, exact=True))
    python, _ = timeit(lambda: [reference.decode_exactly(value) for value in hashes])
    row("decode_exactly 200k", mojo, python, "scalar Python loop")


if __name__ == "__main__":
    main()
