from __future__ import annotations

import gzip
from typing import Final

from grammar_codecs import (
    decode_repair,
    decode_sequitur_style,
    encode_repair,
    encode_sequitur_style,
)


try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - environment dependent
    zstd = None


RLE_PAIR_SIZE: Final[int] = 2


def encode_raw(morse: str) -> str:
    if not isinstance(morse, str):
        raise TypeError("morse must be a str")
    return morse


def decode_raw(payload: str) -> str:
    if not isinstance(payload, str):
        raise TypeError("payload must be a str")
    return payload


def encode_rle(morse: str) -> bytes:
    if not isinstance(morse, str):
        raise TypeError("morse must be a str")

    data = morse.encode("ascii")
    if not data:
        return b""

    encoded = bytearray()
    current = data[0]
    run_length = 1
    for byte in data[1:]:
        if byte == current and run_length < 255:
            run_length += 1
            continue
        encoded.extend((run_length, current))
        current = byte
        run_length = 1
    encoded.extend((run_length, current))
    return bytes(encoded)


def decode_rle(payload: bytes) -> str:
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes-like")
    if len(payload) % RLE_PAIR_SIZE != 0:
        raise ValueError("invalid RLE payload length")

    decoded = bytearray()
    for index in range(0, len(payload), RLE_PAIR_SIZE):
        run_length = payload[index]
        byte = payload[index + 1]
        decoded.extend([byte] * run_length)
    return decoded.decode("ascii")


def encode_gzip(morse: str) -> bytes:
    if not isinstance(morse, str):
        raise TypeError("morse must be a str")
    return gzip.compress(morse.encode("ascii"))


def decode_gzip(payload: bytes) -> str:
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes-like")
    return gzip.decompress(payload).decode("ascii")


def encode_repair_grammar(morse: str) -> bytes:
    return encode_repair(morse)


def decode_repair_grammar(payload: bytes) -> str:
    return decode_repair(payload)


def encode_sequitur_grammar(morse: str) -> bytes:
    return encode_sequitur_style(morse)


def decode_sequitur_grammar(payload: bytes) -> str:
    return decode_sequitur_style(payload)


def zstd_available() -> bool:
    return zstd is not None


def encode_zstd(morse: str) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard is not installed")
    if not isinstance(morse, str):
        raise TypeError("morse must be a str")

    compressor = zstd.ZstdCompressor()
    return compressor.compress(morse.encode("ascii"))


def decode_zstd(payload: bytes) -> str:
    if zstd is None:
        raise RuntimeError("zstandard is not installed")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes-like")

    decompressor = zstd.ZstdDecompressor()
    return decompressor.decompress(payload).decode("ascii")
