"""Loading and building the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_GEOHASH2_LIB") or os.path.join(ROOT, "dist", "libmojo-geohash2.so")

I = ctypes.c_int64

_SIGNATURES = {
    "mgh_encode_batch": ([I, I, I, I, I], None),
    "mgh_decode_batch": ([I, I, I, I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_GEOHASH2_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_GEOHASH2_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile the shared library if it is missing or older than its source."""
    if os.environ.get("MOJO_GEOHASH2_LIB") and os.path.exists(LIB) and not force:
        return LIB
    source = os.path.join(SRC, "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + ["build", "--emit", "shared-lib", source, "-o", LIB]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
