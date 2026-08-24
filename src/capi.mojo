"""C ABI for bulk geohash encoding and exact decoding.

The Python layer owns all buffers.  Addresses rather than typed pointers cross
the ABI because exported Mojo functions cannot be parametric over pointer
origins.
"""

from std.sys import simd_width_of

comptime F64Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def base32_char(value: Int) -> UInt8:
    if value < 10:
        return UInt8(48 + value)
    if value == 10:
        return UInt8(98)
    if value == 11:
        return UInt8(99)
    if value == 12:
        return UInt8(100)
    if value == 13:
        return UInt8(101)
    if value == 14:
        return UInt8(102)
    if value == 15:
        return UInt8(103)
    if value == 16:
        return UInt8(104)
    if value == 17:
        return UInt8(106)
    if value == 18:
        return UInt8(107)
    if value == 19:
        return UInt8(109)
    if value == 20:
        return UInt8(110)
    if value == 21:
        return UInt8(112)
    if value == 22:
        return UInt8(113)
    if value == 23:
        return UInt8(114)
    if value == 24:
        return UInt8(115)
    if value == 25:
        return UInt8(116)
    if value == 26:
        return UInt8(117)
    if value == 27:
        return UInt8(118)
    if value == 28:
        return UInt8(119)
    if value == 29:
        return UInt8(120)
    if value == 30:
        return UInt8(121)
    return UInt8(122)


def base32_value(c: UInt8) -> Int:
    var value = Int(c)
    if value >= 48 and value <= 57:
        return value - 48
    if value == 98:
        return 10
    if value == 99:
        return 11
    if value == 100:
        return 12
    if value == 101:
        return 13
    if value == 102:
        return 14
    if value == 103:
        return 15
    if value == 104:
        return 16
    if value == 106:
        return 17
    if value == 107:
        return 18
    if value == 109:
        return 19
    if value == 110:
        return 20
    if value == 112:
        return 21
    if value == 113:
        return 22
    if value == 114:
        return 23
    if value == 115:
        return 24
    if value == 116:
        return 25
    if value == 117:
        return 26
    if value == 118:
        return 27
    if value == 119:
        return 28
    if value == 120:
        return 29
    if value == 121:
        return 30
    if value == 122:
        return 31
    return -1


@export("mgh_encode_batch")
def mgh_encode_batch(lat_addr: Int, lon_addr: Int, count: Int, precision: Int, dst_addr: Int) abi("C"):
    var latitudes = F64Ptr(unsafe_from_address=lat_addr)
    var longitudes = F64Ptr(unsafe_from_address=lon_addr)
    var dst = U8Ptr(unsafe_from_address=dst_addr)
    for row in range(count):
        var lat_low = -90.0
        var lat_high = 90.0
        var lon_low = -180.0
        var lon_high = 180.0
        var even = True
        var bit = 0
        var character = 0
        var latitude = latitudes.unsafe_load(row)
        var longitude = longitudes.unsafe_load(row)
        for position in range(precision * 5):
            var mask = 16 >> bit
            if even:
                var middle = (lon_low + lon_high) / 2.0
                if longitude > middle:
                    character |= mask
                    lon_low = middle
                else:
                    lon_high = middle
            else:
                var middle = (lat_low + lat_high) / 2.0
                if latitude > middle:
                    character |= mask
                    lat_low = middle
                else:
                    lat_high = middle
            even = not even
            if bit < 4:
                bit += 1
            else:
                dst.unsafe_store(row * precision + position // 5, base32_char(character))
                bit = 0
                character = 0


def decode_range(
    src: U8Ptr, count: Int, precision: Int, latitudes: F64Ptr, longitudes: F64Ptr,
    lat_errors: F64Ptr, lon_errors: F64Ptr, valid: U8Ptr, start: Int, stop: Int,
):
    var row = start
    while row + W <= stop:
        var lat_low = SIMD[DType.float64, W](-90.0)
        var lat_high = SIMD[DType.float64, W](90.0)
        var lon_low = SIMD[DType.float64, W](-180.0)
        var lon_high = SIMD[DType.float64, W](180.0)
        var lat_error = SIMD[DType.float64, W](90.0)
        var lon_error = SIMD[DType.float64, W](180.0)
        var even = True
        for position in range(precision):
            var character = src.unsafe_load[width=W](position * count + row)
            for bit in range(5):
                var mask = UInt8(16 >> bit)
                var upper = (character & SIMD[DType.uint8, W](mask)).ne(0)
                if even:
                    lon_error /= 2.0
                    var middle = (lon_low + lon_high) / 2.0
                    lon_low = upper.select(middle, lon_low)
                    lon_high = upper.select(lon_high, middle)
                else:
                    lat_error /= 2.0
                    var middle = (lat_low + lat_high) / 2.0
                    lat_low = upper.select(middle, lat_low)
                    lat_high = upper.select(lat_high, middle)
                even = not even
        latitudes.unsafe_store(row, (lat_low + lat_high) / 2.0)
        longitudes.unsafe_store(row, (lon_low + lon_high) / 2.0)
        lat_errors.unsafe_store(row, lat_error)
        lon_errors.unsafe_store(row, lon_error)
        valid.unsafe_store(row, SIMD[DType.uint8, W](1))
        row += W
    while row < stop:
        var lat_low = -90.0
        var lat_high = 90.0
        var lon_low = -180.0
        var lon_high = 180.0
        var lat_error = 90.0
        var lon_error = 180.0
        var even = True
        for position in range(precision):
            var character = Int(src.unsafe_load(position * count + row))
            for bit in range(5):
                var mask = 16 >> bit
                if even:
                    lon_error /= 2.0
                    var middle = (lon_low + lon_high) / 2.0
                    if character & mask != 0:
                        lon_low = middle
                    else:
                        lon_high = middle
                else:
                    lat_error /= 2.0
                    var middle = (lat_low + lat_high) / 2.0
                    if character & mask != 0:
                        lat_low = middle
                    else:
                        lat_high = middle
                even = not even
        latitudes.unsafe_store(row, (lat_low + lat_high) / 2.0)
        longitudes.unsafe_store(row, (lon_low + lon_high) / 2.0)
        lat_errors.unsafe_store(row, lat_error)
        lon_errors.unsafe_store(row, lon_error)
        valid.unsafe_store(row, UInt8(1))
        row += 1


@export("mgh_decode_batch")
def mgh_decode_batch(src_addr: Int, count: Int, precision: Int, lat_addr: Int, lon_addr: Int,
                     lat_err_addr: Int, lon_err_addr: Int, valid_addr: Int) abi("C"):
    var src = U8Ptr(unsafe_from_address=src_addr)
    var latitudes = F64Ptr(unsafe_from_address=lat_addr)
    var longitudes = F64Ptr(unsafe_from_address=lon_addr)
    var lat_errors = F64Ptr(unsafe_from_address=lat_err_addr)
    var lon_errors = F64Ptr(unsafe_from_address=lon_err_addr)
    var valid = U8Ptr(unsafe_from_address=valid_addr)
    decode_range(
        src, count, precision, latitudes, longitudes, lat_errors, lon_errors, valid, 0, count
    )
