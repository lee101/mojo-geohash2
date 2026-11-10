"""Profiling harness for the ANALYZE phase of mojo-geohash2.

Measures every public entry point at sizes real callers use (1 scalar .. 1M batch),
splits each batch call into Python glue vs Mojo kernel (direct ctypes, no copies on
the kernel side), and captures machine counters (instructions, cycles, LLC misses,
RSS) so bench/analysis.md can report a measured flops-per-byte figure.

Nothing in src/ or tests/ is touched; this only reads and calls them.

    flock /tmp/mojo-bench.lock pixi run python bench/analysis.py
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import pathlib
import platform
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import geohash2  # noqa: E402
from geohash2 import _lib, geohash as G  # noqa: E402

PRECISION = 12


# ----------------------------------------------------------------------------- reference
def upstream():
    package_dir = pathlib.Path(
        importlib.metadata.distribution("geohash2").locate_file("geohash2")
    )
    spec = importlib.util.spec_from_file_location(
        "_analysis_upstream_geohash2",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- timing
def timeit(function, rounds=5):
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def rss_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def mark(seconds):
    """Format a wall time for the perf-child stderr stanza."""
    return f"{seconds * 1e3:.4f} ms"


# ----------------------------------------------------------------------------- data
def make_data(count: int):
    rng = np.random.default_rng(2026)
    return rng.uniform(-90.0, 90.0, count), rng.uniform(-180.0, 180.0, count)


def decode_buffers(hashes: np.ndarray):
    """Exactly what _decode_fixed builds, so the kernel can be timed alone."""
    count = len(hashes)
    flat = np.frombuffer("".join(hashes.tolist()).encode("ascii"), dtype=np.uint8).reshape(count, PRECISION)
    source = G._DECODE_BYTES[flat].T.copy()
    outputs = [np.empty(count, dtype=np.float64) for _ in range(4)]
    valid = np.empty(count, dtype=np.uint8)
    return source, outputs, valid


def run_decode(source, outputs, valid):
    _lib.lib().mgh_decode_batch(
        _lib.addr(source), source.shape[1], PRECISION,
        *[_lib.addr(o) for o in outputs], _lib.addr(valid),
    )


def run_encode(latitudes, longitudes, encoded):
    _lib.lib().mgh_encode_batch(
        _lib.addr(latitudes), _lib.addr(longitudes), len(latitudes), PRECISION, _lib.addr(encoded)
    )


def _check_one(value: str) -> str:
    """One _check_geohash call whose result the profiler always consumes."""
    return G._check_geohash(value)


# ----------------------------------------------------------------------------- sidecar
SIDECAR = pathlib.Path(__file__).with_name("_analysis_child.py")
DATA = SIDECAR.with_suffix(".json")

SIDECAR_SOURCE = '''
"""Sidecar: fixed-width timing for one kernel call. Not a benchmark."""
import json, sys, time
import numpy as np
sys.path.insert(0, PYTHON_ROOT)
from geohash2 import _lib, geohash as G

with open(DATA_FILE) as handle:
    payload = json.load(handle)
mode, reps = sys.argv[1], int(sys.argv[2])
count = len(payload["lat"])
lat = np.ascontiguousarray(payload["lat"], dtype=np.float64)
lon = np.ascontiguousarray(payload["lon"], dtype=np.float64)
encoded = np.empty((count, 12), dtype=np.uint8)
if mode == "encode":
    call = lambda: _lib.lib().mgh_encode_batch(_lib.addr(lat), _lib.addr(lon), count, 12, _lib.addr(encoded))
else:
    flat = np.frombuffer("".join(payload["hashes"]).encode("ascii"), dtype=np.uint8).reshape(count, 12)
    src = G._DECODE_BYTES[flat].T.copy()
    outs = [np.empty(count) for _ in range(4)]
    valid = np.empty(count, dtype=np.uint8)
    call = lambda: _lib.lib().mgh_decode_batch(_lib.addr(src), count, 12,
        *[_lib.addr(o) for o in outs], _lib.addr(valid))
call()                                   # warm page cache / first-touch
best = float("inf"); total = 0.0
for _ in range(reps):
    start = time.perf_counter(); call(); elapsed = time.perf_counter() - start
    best = min(best, elapsed); total += elapsed
print(json.dumps({"mode": mode, "count": count, "reps": reps,
                  "best_ms": best * 1e3, "mean_ms": total / reps * 1e3}))
'''


def sidecar(mode: str, count: int, reps: int) -> dict:
    latitudes, longitudes = make_data(count)
    hashes = geohash2.encode_many(latitudes, longitudes, PRECISION).reshape(-1).tolist()
    DATA.write_text(json.dumps({"lat": latitudes.tolist(), "lon": longitudes.tolist(), "hashes": hashes}))
    SIDECAR.write_text(SIDECAR_SOURCE.replace("PYTHON_ROOT", repr(str(ROOT / "python")))
                                      .replace("DATA_FILE", repr(str(DATA))))
    done = subprocess.run([sys.executable, str(SIDECAR), mode, str(reps)],
                          capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        return {"error": (done.stderr or done.stdout)[-800:]}
    return json.loads(done.stdout)


def perf_counters(command: list[str]) -> dict:
    events = "task-clock,cycles,instructions,branches,branch-misses,cache-misses,cache-references"
    try:
        done = subprocess.run(["perf", "stat", "-x,", "-e", events, *command],
                              capture_output=True, text=True, timeout=3600)
    except Exception as error:
        return {"error": repr(error)}
    counters, notes = {}, []
    for line in (done.stderr or "").splitlines():
        parts = line.split(",")
        if len(parts) >= 3:
            name = parts[2].strip()
            try:
                value = float(parts[0])
            except ValueError:
                value = None
            if name and name != "event" and value is not None:
                counters[name] = value
                continue
        if "not supported" in line or "Permission" in line or "limited" in line:
            notes.append(line.strip())
    if not counters:
        counters["error"] = (done.stderr or "")[-800:]
    if notes:
        counters["notes"] = notes[:2]
    return counters


# ----------------------------------------------------------------------------- main
def main():
    reference = upstream()
    report = {
        "env": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpu": next(line for line in open("/proc/cpuinfo") if line.startswith("model name")).split(":")[1].strip(),
            "nproc": len(__import__("os").sched_getaffinity(0)),
            "rss_after_import_kib": rss_kib(),
        },
        "rows": [], "glue": [], "transport": [], "sizing": [], "counters": {},
    }

    # ---- 1. encode entry points, batch sizes real callers use
    print("== encode ==", flush=True)
    for count in (1, 1_000, 10_000, 200_000, 1_000_000):
        latitudes, longitudes = make_data(count)
        if count == 1:
            lat0, lon0 = float(latitudes[0]), float(longitudes[0])
            total = timeit(lambda: geohash2.encode(lat0, lon0, PRECISION))[0]
            encoded = np.empty((1, PRECISION), dtype=np.uint8)
            getattr(_lib.lib(), "mgh_encode_batch")(_lib.addr(np.array([lat0])), _lib.addr(np.array([lon0])), 1, PRECISION, _lib.addr(encoded))
            kernel = timeit(lambda: _lib.lib().mgh_encode_batch(
                _lib.addr(np.array([lat0])), _lib.addr(np.array([lon0])), 1, PRECISION, _lib.addr(encoded)))[0]
            strings = timeit(lambda: encoded.tobytes().decode("ascii"))[0]
            row = {"entry": "encode()", "n": 1, "total": total, "kernel": kernel, "strings": strings}
        else:
            total = timeit(lambda: geohash2.encode_many(latitudes, longitudes, PRECISION))[0]
            encoded = np.empty((count, PRECISION), dtype=np.uint8)
            kernel = timeit(lambda: run_encode(latitudes, longitudes, encoded))[0]
            strings = timeit(lambda: np.array([r.tobytes().decode("ascii") for r in encoded], dtype=f"<U{PRECISION}"))[0]
            row = {"entry": "encode_many()", "n": count, "total": total, "kernel": kernel, "strings": strings}
        row["glue"] = max(row["total"] - row["kernel"] - row["strings"], 0.0)
        row["rss_kib"] = rss_kib()
        report["rows"].append(row)
        print(row, flush=True)

    # ---- 2. decode entry points
    print("== decode ==", flush=True)
    for count in (1, 1_000, 10_000, 200_000, 1_000_000):
        latitudes, longitudes = make_data(count)
        hashes = geohash2.encode_many(latitudes, longitudes, PRECISION).reshape(-1)
        if count == 1:
            value = str(hashes[0])
            total = timeit(lambda: geohash2.decode_exactly(value))[0]
            kernel_time = timeit(lambda: _lib.lib().mgh_decode_batch(
                _lib.addr(np.empty((PRECISION, 1), dtype=np.uint8)), 1, PRECISION,
                *[_lib.addr(np.empty(1)) for _ in range(4)], _lib.addr(np.empty(1, dtype=np.uint8))))[0]
            pre = timeit(lambda: G._check_geohash(value))[0]
            row = {"entry": "decode_exactly()", "n": 1, "total": total, "kernel": kernel_time, "pre": pre}
        else:
            total = timeit(lambda: geohash2.decode_many(hashes, exact=True))[0]
            source, outputs, valid = decode_buffers(hashes)
            kernel_time = timeit(lambda: run_decode(source, outputs, valid))[0]
            pre = timeit(lambda: [G._check_geohash(v) for v in hashes])[0]
            strings = timeit(lambda: "".join(hashes.tolist()).encode("ascii"))[0]
            row = {"entry": "decode_many(exact=True)", "n": count, "total": total,
                   "kernel": kernel_time, "pre": pre, "strings": strings}
        row["glue"] = max(row["total"] - row["kernel"] - row["pre"], 0.0)
        row["rss_kib"] = rss_kib()
        report["rows"].append(row)
        print(row, flush=True)

    # ---- 3. decode_many(display) and decode()
    print("== decode display ==", flush=True)
    report["rows"].append({"entry": "decode()", "n": 1,
                           "total": timeit(lambda: geohash2.decode("u4pruydqqvj"))[0]})
    for count in (10_000, 200_000):
        latitudes, longitudes = make_data(count)
        hashes = geohash2.encode_many(latitudes, longitudes, PRECISION).reshape(-1)
        report["rows"].append({"entry": "decode_many(exact=False)", "n": count,
                               "total": timeit(lambda: geohash2.decode_many(hashes, exact=False), rounds=3)[0]})

    # ---- 4. upstream comparison at the same sizes (cheap sizes only; 200k upstream
    #         runs run for minutes on this box and would monopolise the bench lock)
    print("== upstream ==", flush=True)
    for count in (1, 10_000):
        latitudes, longitudes = make_data(count)
        hashes = geohash2.encode_many(latitudes, longitudes, PRECISION).reshape(-1)
        pairs = list(zip(latitudes.tolist(), longitudes.tolist())) if count > 1 else None
        values = hashes.tolist()
        if count == 1:
            lat0, lon0, value = float(latitudes[0]), float(longitudes[0]), values[0]
            mojo_enc = timeit(lambda: geohash2.encode(lat0, lon0, PRECISION))[0]
            ref_enc = timeit(lambda: reference.encode(lat0, lon0, PRECISION))[0]
            mojo_dec = timeit(lambda: geohash2.decode_exactly(value))[0]
            ref_dec = timeit(lambda: reference.decode_exactly(value))[0]
            mojo_disp = timeit(lambda: geohash2.decode(value))[0]
            ref_disp = timeit(lambda: reference.decode(value))[0]
        else:
            mojo_enc = timeit(lambda: geohash2.encode_many(latitudes, longitudes, PRECISION))[0]
            ref_enc = timeit(lambda: [reference.encode(a, b, PRECISION) for a, b in pairs], rounds=3)[0]
            mojo_dec = timeit(lambda: geohash2.decode_many(hashes, exact=True))[0]
            ref_dec = timeit(lambda: [reference.decode_exactly(v) for v in values], rounds=3)[0]
            mojo_disp = timeit(lambda: geohash2.decode_many(hashes, exact=False))[0]
            ref_disp = timeit(lambda: [reference.decode(v) for v in values], rounds=3)[0]
        report["rows"].append({"entry": "upstream encode", "n": count, "mojo": mojo_enc, "upstream": ref_enc})
        report["rows"].append({"entry": "upstream decode_exactly", "n": count, "mojo": mojo_dec, "upstream": ref_dec})
        report["rows"].append({"entry": "upstream decode(display)", "n": count, "mojo": mojo_disp, "upstream": ref_disp})
        print(report["rows"][-3:], file=sys.stderr)

    # ---- 5. transport / copy costs (message-passing between layers)
    print("== transport ==", flush=True)
    for count in (1, 200_000):
        hashes = np.full(count, "u4pruydqqvj", dtype=object)
        join_parts = hashes.tolist()
        encoded = np.empty((count, PRECISION), dtype=np.uint8)
        if count > 1:
            latitudes, longitudes = make_data(count)
            run_encode(latitudes, longitudes, encoded)
            row_strings = lambda: np.array([r.tobytes().decode("ascii") for r in encoded], dtype=f"<U{PRECISION}")
            check = lambda: [_check_one(v) for v in join_parts]
            flat = lambda: G._DECODE_BYTES[np.frombuffer("".join(join_parts).encode("ascii"), dtype=np.uint8).reshape(count, PRECISION)].T.copy()
        else:
            encoded[:] = np.frombuffer(b"u4pruydqqvj", dtype=np.uint8)
            row_strings = lambda: encoded.tobytes().decode("ascii")
            check = lambda: [_check_one(join_parts[0])]
            flat = lambda: G._DECODE_BYTES[np.frombuffer("".join(join_parts).encode("ascii"), dtype=np.uint8).reshape(count, PRECISION)].T.copy()
        report["transport"].append({"kind": "np.asarray(object) on object array", "n": count,
                                    "bytes": count * PRECISION, "time": timeit(lambda: np.asarray(hashes, dtype=object))[0]})
        report["transport"].append({"kind": "object array -> Python list", "n": count,
                                    "bytes": count * PRECISION + count * 8, "time": timeit(hashes.tolist)[0]})
        report["transport"].append({"kind": "_check_geohash loop (per-element Python)", "n": count,
                                    "bytes": count * PRECISION, "time": timeit(check)[0]})
        report["transport"].append({"kind": "join + ascii bytes (input packing)", "n": count,
                                    "bytes": 2 * count * PRECISION, "time": timeit(lambda: "".join(join_parts).encode("ascii"))[0]})
        report["transport"].append({"kind": "byte->value LUT + transpose", "n": count,
                                    "bytes": 2 * (count * PRECISION) + count * PRECISION, "time": timeit(flat)[0]})
        report["transport"].append({"kind": "rows -> Python str (encode_many tail)", "n": count,
                                    "bytes": count * PRECISION + count * 4 * PRECISION, "time": timeit(row_strings)[0]})
        for _ in range(1):
            print(report["transport"][-6], file=sys.stderr)

    # ---- 6. per-call fixed overhead
    print("== fixed overhead ==", flush=True)
    empty_lat, empty_lon = np.empty(0), np.empty(0)
    report["sizing"].append({"kind": "encode_many n=0 (fixed call overhead)",
                             "time": timeit(lambda: geohash2.encode_many(empty_lat, empty_lon, PRECISION))[0]})
    report["sizing"].append({"kind": "decode_many n=0",
                             "time": timeit(lambda: geohash2.decode_many(np.array([], dtype=object), True))[0]})
    report["sizing"].append({"kind": "encode() precision=0 (pure Python shortcut)",
                             "time": timeit(lambda: geohash2.encode(1.0, 2.0, 0))[0]})
    for item in report["sizing"]:
        print(item, flush=True)

    # ---- 7. fixed-width sizing at 1M (decode is a streaming kernel)
    print("== 1M fixed-width sizing ==", flush=True)
    for mode, count, reps in (("encode", 200_000, 200), ("encode", 1_000_000, 3),
                              ("decode", 200_000, 200), ("decode", 1_000_000, 3)):
        result = sidecar(mode, count, reps)
        if "count" in result:
            result["per_call_ms"] = result.pop("best_ms")
            for stale in ("reps", "mean_ms"):
                result.pop(stale, None)
        report["sizing"].append({"kind": f"{mode} n={count:,} x 12", **result})
        print(report["sizing"][-1], file=sys.stderr)

    # ---- 8. machine counters for flops-per-byte (perf is usually unavailable in this container)
    print("== perf ==", flush=True)
    probe = perf_counters(["true"])
    report["counters"]["probe"] = probe
    if "cycles" in probe:
        for mode, reps in (("encode", 400), ("decode", 300)):
            sidecar(mode, 200_000, 1)          # (re)write the 200k payload for this mode
            report["counters"][mode] = perf_counters([sys.executable, str(SIDECAR), mode, str(reps)])
            print(report["counters"][mode], file=sys.stderr)
    else:
        report["counters"]["unavailable"] = "this session has no PMU access"
        print(report["counters"], file=sys.stderr)
    for leftover in (SIDECAR, DATA):
        leftover.unlink(missing_ok=True)

    out = ROOT / "build" / "analysis_raw.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out}")
    print(f"peak RSS {rss_kib() / 1024:.0f} MiB")


if __name__ == "__main__":
    main()
