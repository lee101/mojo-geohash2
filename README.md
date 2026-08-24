# mojo-geohash2

`mojo-geohash2` is a Mojo implementation of [geohash2](https://pypi.org/project/geohash2/)'s coordinate encoder and decoder. It keeps the upstream scalar API intact, then adds batch entry points for the workload where compiled kernels matter: turning large arrays of WGS84 coordinates into geohashes, or decoding them back to bounding-box centres and error margins.

## Covered API

The complete public API in `geohash2` 1.1 is covered with the same names and signatures:

- `encode(latitude, longitude, precision=12) -> str`
- `decode_exactly(geohash) -> (latitude, longitude, latitude_error, longitude_error)`
- `decode(geohash) -> (latitude_string, longitude_string)`

`decode` deliberately returns strings, including quirks such as `('-0.', '-0.')` at the origin, matching upstream. Invalid alphabet characters raise `KeyError` as they do in `geohash2`.

Two NumPy-oriented extensions are included:

- `encode_many(latitude, longitude, precision=12) -> np.ndarray[str]` broadcasts the coordinate arrays.
- `decode_many(geohashes, exact=False)` accepts an array-like of mixed precision hashes. With `exact=True`, it returns four float arrays; otherwise it returns the two upstream-style string arrays.

There are no additional public algorithms in upstream `geohash2` 1.1 to port. This project does not implement neighbouring-cell lookup, spatial joins, GeoJSON helpers, or a database index; those are outside geohash2's encode/decode scope.

## Install and use

This checkout runs with Pixi and the pinned Mojo nightly:

```bash
pixi install
pixi run build
pixi run python - <<'PY'
import geohash2
import numpy as np

hash_ = geohash2.encode(57.64911, 10.40744, precision=11)
print(hash_)                         # u4pruydqqvj
print(geohash2.decode_exactly(hash_))

hashes = geohash2.encode_many(
    np.array([42.6, 57.64911]), np.array([-5.6, 10.40744]), precision=5
)
print(hashes.tolist())               # ['ezs42', 'u4pru']
PY
```

The module builds `dist/libmojo-geohash2.so` on first use from this checkout. Set `MOJO_GEOHASH2_LIB` to an already-built library when a deployment must not compile at import time.

## Benchmarks

Measured with `pixi run bench` on `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`, 200,000 random WGS84 coordinates at precision 12, best of three runs. The reference is the installed PyPI `geohash2` 1.1 scalar loop over the identical data.

| kernel | mojo-geohash2 | geohash2 1.1 | speedup | reference |
| --- | ---: | ---: | ---: | --- |
| encode 200k | 178.32 ms | 3291.93 ms | 18.46x | scalar Python loop |
| decode_exactly 200k | 354.63 ms | 2582.81 ms | 7.28x | scalar Python loop |

The scalar `encode`, `decode`, and `decode_exactly` calls use the same kernels but still pay Python/ctypes and small-array allocation overhead. Use the `*_many` functions when processing many locations.

There is no GPU path. Both measured batch kernels already exceed the optimization
cutoff versus upstream, and their end-to-end work includes CPU-side validation and
conversion to Python strings. Moving only the interval-refinement arithmetic to a
GPU would add transfers and launch overhead without removing that dominant host
work, so a GPU implementation is not justified here.

## How it works

`src/capi.mojo` is one compilation unit containing the base32 interleave and interval-refinement loops. `build/build.sh` emits a C shared library at `dist/libmojo-geohash2.so`. The Python wrapper owns C-contiguous `float64` input/output arrays and `uint8` character buffers for the whole ctypes call, then passes their addresses to Mojo. Decoding writes four contiguous `float64` output arrays: latitude, longitude, and their respective half-interval errors.

The Python layer owns all memory and converts only the finished byte rows to Python strings. It validates the geohash alphabet before dispatch so upstream's `KeyError` behaviour remains visible instead of becoming an FFI error.

## Verification

```bash
pixi run build
pixi run test
pixi run bench
```

The pytest suite loads the separately installed PyPI `geohash2` 1.1 package under an isolated name and checks published vectors, random scalar encode/decode parity, exact error bounds, invalid-input behaviour, and both batch APIs.

MIT. The upstream package is installed only as a test/benchmark reference.
