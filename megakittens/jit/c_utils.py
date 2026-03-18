import ctypes
import struct


def c_int(v: int) -> bytes:
    return struct.pack('<i', v)


def c_float(v: float) -> bytes:
    return struct.pack('<f', v)


def align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def pack_struct(fields: list[tuple[bytes, int, int]]) -> bytearray:
    """Pack (data, size, alignment) tuples into a C struct with padding."""
    offset, max_align = 0, 1
    for _, size, align in fields:
        offset = align_up(offset, align)
        offset += size
        max_align = max(max_align, align)
    buf = bytearray(align_up(offset, max_align))
    offset = 0
    for data, size, align in fields:
        offset = align_up(offset, align)
        buf[offset:offset+size] = data
        offset += size
    return buf


def pack_args(fields: list[tuple[bytes, int, int]]) -> tuple[ctypes.Array, ctypes.Array]:
    """Pack struct fields and return (c_buf, packed_params) for cuLaunchKernelEx.

    Returns both objects because packed_params holds a raw, unreferenced pointer into
    buf_holder's memory. Caller must keep both alive up to the launch call.
    """
    buf = pack_struct(fields)
    c_buf = (ctypes.c_char * len(buf)).from_buffer(buf)
    packed_params = (ctypes.c_void_p * 1)(ctypes.addressof(c_buf))
    return c_buf, packed_params
