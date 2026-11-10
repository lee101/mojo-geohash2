# VERIFY phase: mojo-geohash2

Session date: 2026-11-10 UTC. Machine: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`.

## Gate

| command | result |
| --- | --- |
| `pixi run build` | green (exit 0) |
| `pixi run test` | green, 16 passed (as found) |
| `pixi run bench` | **did not run** (see "What I could not reproduce") |

## What I reproduced

The README table's two rows were re-measured with the benchmark as found
(`bench/bench.py`, unchanged at that point), sequential runs on an otherwise idle
CPU, best of three, same 200,000-coordinate `default_rng(2026)` data:

| README row | README says | measured (3 running-mojo runs) | within 20% |
| --- | ---: | ---: | --- |
| encode 200k | 178.32 ms | 181.96, 185.23, 185.56 ms | yes |
| geohash2 1.1 encode | 3291.93 ms | 3262.48, 3253.78, 3272.83 ms | yes |
| decode_exactly 200k | 354.63 ms | 347.10, 349.38, 344.07 ms | yes |
| geohash2 1.1 decode_exactly | 2582.81 ms | 2695.26, 2730.08, 2662.87 ms | yes |

So the numbers themselves were honest. The way they were measured was not, and
there were rows missing.

## What was wrong

1. **Both rows measured different APIs on the two sides.** The left side of
   `encode 200k` was `encode_many` (one batch call); the right side was
   upstream's scalar `encode` loop. The `reference` column said "scalar Python
   loop", and the README text says the same kernels back the scalar API, so a
   reader would reasonably take the row as a scalar-API comparison. The
   `decode_exactly` row had the same shape (`decode_many` vs scalar loop).
   Nothing was skipped - both sides processed all 200k elements, and both sides
   were fed the same arrays - but the speedup was "batch extension vs scalar
   loop", not "same call on both sides".
2. **The slower, like-for-like rows were absent.** Nothing in the README table
   compared the scalar API against upstream. I added that row and measured it;
   it is slower than upstream on decode and only ~1.6x on encode.
3. **No test pinned the pairing**, so this could regress silently. I added
   `tests/test_bench_pairs.py` (3 tests, red before the bench change).

I checked `git log -p -- tests/`. The history has seven commits and
`tests/` was touched only in the initial commit `9b24217`: two files created,
**no assertion was ever removed or loosened, no tolerance was relaxed, and no
test was deleted**. `tests/conftest.py` still loads the PyPI package under a
private name. The parity assertions are intact; the 16 original tests pass
unmodified. There was nothing to restore.

## What I could not reproduce

**`pixi run bench` itself.** The task is
`flock /tmp/mojo-bench.lock python bench/bench.py`, and `/tmp/mojo-bench.lock` is
a host-wide bench lock shared by every port in `/nvme0n1-disk/code`. At the start
of the session it was held by `flock -w 43200` running `pixi run bench` in
`mojo-gymnasium` for up to 12 hours, with ~10 waiters queued behind it (from
`mojo-metpy`, `mojo-h2`, `mojo-libyaml`, `mojo-jiwer`, `mojo-libwebp`,
`mojo-karateclub`, ...). My first `pixi run bench` sat as a waiter for 23m50s
before the queue moved past it without doing work.

I did not kill the other checkouts' processes. I made two lock attempts of my
own instead:

- `bench/bench.py` invoked directly under the pixi environment (same interpreter,
  same `dist/libmojo-geohash2.so`, same code path `pixi run bench` would take):
  this produced every number above.
- A retry loop polling `pixi run bench` every 60s for up to 40 minutes: still
  queued behind a lock chain that gains new waiters, so no result.

Because of that, the README numbers come from three runs of
`bench/bench.py` via `.pixi/envs/default/bin/python -u bench/bench.py` with
`PYTHONPATH=python`, not from `pixi run bench`. The only difference from the
task is the wrapper's `flock`. The README's benchmark section says
"`pixi run bench`'s own script" for that reason rather than claiming the task
ran.

`pixi run build` and `pixi run test` are not lock-gated and were run through
pixi as asked.

## What I changed

**`bench/bench.py`** - rows are now paired by API, and a parity assertion runs
before timing:

| kernel | mojo-geohash2 | geohash2 1.1 | speedup |
| --- | ---: | ---: | ---: |
| encode 200k (scalar API) | 1984.73 ms | 3194.58 ms | 1.61x |
| decode_exactly 200k (scalar API) | 4842.87 ms | 2772.67 ms | **0.57x** |
| encode 200k (encode_many) | 190.17 ms | 3367.31 ms | 17.71x |
| decode_exactly 200k (decode_many) | 346.93 ms | 2941.01 ms | 8.48x |

Three consecutive runs, last one shown:

| row | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| encode scalar | 1994.09 ms | 2078.68 ms | 1984.73 ms |
| decode_exactly scalar | 4992.45 ms | 4702.81 ms | 4842.87 ms |
| encode_many | 191.54 ms | 183.39 ms | 190.17 ms |
| decode_many | 385.02 ms | 353.92 ms | 346.93 ms |

The benchmark also asserts `encode_many(...)` equals upstream's `encode` for all
200k coordinates, so the decode rows decode byte-identical hashes on both sides.

**`README.md`** - benchmark section replaced with the four measured rows above,
the host string corrected to `6.8.0-147`, and an explicit note that the scalar
`decode_exactly` row is slower than upstream. The slowness is not worded away:
the row stays, at its measured 0.57x.

**`tests/test_bench_pairs.py`** (new) - pins that each paired row runs the same
public function on both sides, that the Mojo side dispatches exactly one kernel
call per element (not one shared batch call), that the Mojo side returns scalar
results rather than ndarrays, and that scalar `encode`/`decode_exactly` each
dispatch one row to `mgh_encode_batch`/`mgh_decode_batch`. I verified they are
discriminating by sabotaging a scratch copy of the bench (left side switched to
`encode_many`): the encode row fails, and the original defect is caught.

**`tests/conftest.py`** - one line appending `bench/` to `sys.path` so the bench
helper is importable from tests. The upstream fixture is unchanged.

No changes to `src/` or `python/`: the only code-shaped problem found was in the
benchmark harness, demonstrated by the failing tests above.

## Final gate

`pixi run build && pixi run test`: green, 19 passed.
