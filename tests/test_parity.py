import numpy as np
import pytest

import geohash2


@pytest.mark.parametrize(
    ("latitude", "longitude", "precision", "expected"),
    [
        (42.6, -5.6, 5, "ezs42"),
        (57.64911, 10.40744, 11, "u4pruydqqvj"),
        (0.0, 0.0, 12, "7zzzzzzzzzzz"),
        (-90.0, -180.0, 3, "000"),
        (90.0, 180.0, 3, "zzz"),
    ],
)
def test_published_encode_vectors(latitude, longitude, precision, expected):
    assert geohash2.encode(latitude, longitude, precision) == expected


def test_scalar_encode_matches_upstream_random_coordinates(upstream_geohash2):
    rng = np.random.default_rng(2026)
    latitudes = rng.uniform(-90.0, 90.0, 250)
    longitudes = rng.uniform(-180.0, 180.0, 250)
    for precision in (1, 2, 5, 8, 12):
        actual = [geohash2.encode(lat, lon, precision) for lat, lon in zip(latitudes, longitudes)]
        expected = [upstream_geohash2.encode(lat, lon, precision) for lat, lon in zip(latitudes, longitudes)]
        assert actual == expected


def test_decode_exactly_matches_upstream_random_hashes(upstream_geohash2):
    rng = np.random.default_rng(7)
    for precision in (0, 1, 3, 6, 12):
        hashes = [
            upstream_geohash2.encode(lat, lon, precision)
            for lat, lon in zip(rng.uniform(-90, 90, 100), rng.uniform(-180, 180, 100))
        ]
        for value in hashes:
            assert geohash2.decode_exactly(value) == pytest.approx(
                upstream_geohash2.decode_exactly(value), abs=0.0
            )
            assert geohash2.decode(value) == upstream_geohash2.decode(value)


def test_decode_accepts_upstream_iterable_input(upstream_geohash2):
    value = ["e", "z", "s", "4", "2"]
    assert geohash2.decode_exactly(value) == pytest.approx(upstream_geohash2.decode_exactly(value))
    assert geohash2.decode(value) == upstream_geohash2.decode(value)


@pytest.mark.parametrize("value", ["EZS42", "ezs4!", b"ezs42"])
def test_invalid_character_raises_same_keyerror(value, upstream_geohash2):
    with pytest.raises(KeyError) as actual:
        geohash2.decode_exactly(value)
    with pytest.raises(KeyError) as expected:
        upstream_geohash2.decode_exactly(value)
    assert actual.value.args == expected.value.args


def test_encode_precision_edge_cases_match_upstream(upstream_geohash2):
    for precision in (0, -2, True, False, 1.5):
        assert geohash2.encode(0.0, 0.0, precision) == upstream_geohash2.encode(0.0, 0.0, precision)


def test_encode_many_matches_upstream_scalar_loop(upstream_geohash2):
    rng = np.random.default_rng(1)
    latitudes = rng.uniform(-90, 90, size=(31, 17))
    longitudes = rng.uniform(-180, 180, size=(31, 17))
    actual = geohash2.encode_many(latitudes, longitudes, precision=9)
    expected = np.array(
        [upstream_geohash2.encode(lat, lon, 9) for lat, lon in zip(latitudes.flat, longitudes.flat)]
    ).reshape(latitudes.shape)
    assert np.array_equal(actual, expected)


def test_encode_many_broadcasts_like_numpy(upstream_geohash2):
    latitudes = np.array([[42.6], [57.64911]])
    longitudes = np.array([-5.6, 10.40744])
    actual = geohash2.encode_many(latitudes, longitudes, precision=5)
    expected = np.array(
        [[upstream_geohash2.encode(lat, lon, 5) for lon in longitudes] for lat in latitudes[:, 0]]
    )
    assert np.array_equal(actual, expected)


def test_decode_many_matches_upstream_scalar_loop(upstream_geohash2):
    hashes = np.array([["ezs42", "u4pruydqqvj", "7zzzzzz"], ["000", "z", ""]], dtype=object)
    exact = geohash2.decode_many(hashes, exact=True)
    expected = np.array([upstream_geohash2.decode_exactly(value) for value in hashes.flat])
    for got, want in zip(exact, expected.T):
        assert np.array_equal(got, want.reshape(hashes.shape))
    text = geohash2.decode_many(hashes)
    expected_text = np.array([upstream_geohash2.decode(value) for value in hashes.flat], dtype=object)
    assert np.array_equal(text[0], expected_text[:, 0].reshape(hashes.shape))
    assert np.array_equal(text[1], expected_text[:, 1].reshape(hashes.shape))


def test_decode_many_simd_tail_and_parallel_threshold():
    value = "u4pruydqqvj"
    expected = geohash2.decode_exactly(value)
    hashes = np.full(32_771, value, dtype=object)
    actual = geohash2.decode_many(hashes, exact=True)
    for result, want in zip(actual, expected):
        assert np.all(result == want)
