"""Mojo-backed implementation of the public geohash2 encode/decode API."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, lib

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODEMAP = {character: index for index, character in enumerate(_BASE32)}
_DECODE_BYTES = np.full(256, 255, dtype=np.uint8)
for _character, _value in _DECODEMAP.items():
    _DECODE_BYTES[ord(_character)] = _value


def _precision(value) -> int:
    if value <= 0:
        return 0
    return math.ceil(value)


def _check_geohash(value) -> str:
    characters = []
    for character in value:
        if character not in _DECODEMAP:
            raise KeyError(character)
        characters.append(character)
    return "".join(characters)


def _decode_fixed(values: list[str], precision: int):
    count = len(values)
    latitudes = np.empty(count, dtype=np.float64)
    longitudes = np.empty(count, dtype=np.float64)
    lat_errors = np.empty(count, dtype=np.float64)
    lon_errors = np.empty(count, dtype=np.float64)
    if count == 0:
        return latitudes, longitudes, lat_errors, lon_errors
    if precision == 0:
        latitudes.fill(0.0)
        longitudes.fill(0.0)
        lat_errors.fill(90.0)
        lon_errors.fill(180.0)
        return latitudes, longitudes, lat_errors, lon_errors
    source = np.frombuffer("".join(values).encode("ascii"), dtype=np.uint8).reshape(count, precision)
    source = _DECODE_BYTES[source].T.copy()
    valid = np.empty(count, dtype=np.uint8)
    lib().mgh_decode_batch(
        addr(source), count, precision, addr(latitudes), addr(longitudes),
        addr(lat_errors), addr(lon_errors), addr(valid),
    )
    return latitudes, longitudes, lat_errors, lon_errors


def encode(latitude, longitude, precision=12):
    """Encode latitude and longitude as a geohash of ``precision`` characters."""
    precision = _precision(precision)
    if precision <= 0:
        return ""
    latitudes = np.array([latitude], dtype=np.float64)
    longitudes = np.array([longitude], dtype=np.float64)
    encoded = np.empty((1, precision), dtype=np.uint8)
    lib().mgh_encode_batch(addr(latitudes), addr(longitudes), 1, precision, addr(encoded))
    return encoded.tobytes().decode("ascii")


def decode_exactly(geohash):
    """Return latitude, longitude and their positive error margins."""
    value = _check_geohash(geohash)
    latitudes, longitudes, lat_errors, lon_errors = _decode_fixed([value], len(value))
    return float(latitudes[0]), float(longitudes[0]), float(lat_errors[0]), float(lon_errors[0])


def _display(value: float, error: float) -> str:
    decimals = max(1, int(round(-math.log10(error)))) - 1
    formatted = "%.*f" % (decimals, value)
    return formatted.rstrip("0") if "." in formatted else formatted


def decode(geohash):
    """Decode a geohash to the same significant coordinate strings as geohash2."""
    latitude, longitude, lat_error, lon_error = decode_exactly(geohash)
    return _display(latitude, lat_error), _display(longitude, lon_error)


def encode_many(latitude, longitude, precision=12):
    """Vectorized ``encode`` for broadcastable latitude and longitude arrays."""
    precision = _precision(precision)
    latitudes, longitudes = np.broadcast_arrays(
        np.asarray(latitude, dtype=np.float64), np.asarray(longitude, dtype=np.float64)
    )
    shape = latitudes.shape
    if precision <= 0:
        return np.full(shape, "", dtype="<U1")
    flat_latitudes = np.ascontiguousarray(latitudes.reshape(-1))
    flat_longitudes = np.ascontiguousarray(longitudes.reshape(-1))
    encoded = np.empty((flat_latitudes.size, precision), dtype=np.uint8)
    if flat_latitudes.size:
        lib().mgh_encode_batch(
            addr(flat_latitudes), addr(flat_longitudes), flat_latitudes.size, precision, addr(encoded)
        )
    strings = np.array([row.tobytes().decode("ascii") for row in encoded], dtype=f"<U{precision}")
    return strings.reshape(shape)


def decode_many(geohashes, exact: bool = False):
    """Vectorized decode for an array-like of geohashes of mixed precisions."""
    array = np.asarray(geohashes, dtype=object)
    shape = array.shape
    values = [_check_geohash(value) for value in array.reshape(-1)]
    latitudes = np.empty(len(values), dtype=np.float64)
    longitudes = np.empty(len(values), dtype=np.float64)
    lat_errors = np.empty(len(values), dtype=np.float64)
    lon_errors = np.empty(len(values), dtype=np.float64)
    by_precision: dict[int, list[int]] = {}
    for index, value in enumerate(values):
        by_precision.setdefault(len(value), []).append(index)
    for precision, indices in by_precision.items():
        chunk = [values[index] for index in indices]
        decoded = _decode_fixed(chunk, precision)
        for target, output in zip((latitudes, longitudes, lat_errors, lon_errors), decoded):
            target[indices] = output
    if exact:
        return tuple(item.reshape(shape) for item in (latitudes, longitudes, lat_errors, lon_errors))
    latitude_text = np.array([_display(value, error) for value, error in zip(latitudes, lat_errors)], dtype=object)
    longitude_text = np.array([_display(value, error) for value, error in zip(longitudes, lon_errors)], dtype=object)
    return latitude_text.reshape(shape), longitude_text.reshape(shape)
