from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
import lz4.frame

REPRESENTATIONS = ("lz4_whole", "zstd3_whole", "lz4_block64", "lz4_block256", "lz4_block1024")

@dataclass(frozen=True)
class EncodedChunk:
    payload: bytes
    metadata: dict


def _as_float32(values) -> np.ndarray:
    a = np.asarray(values, dtype='<f4')
    if a.ndim != 1:
        raise ValueError('RePressDB chunks are one-dimensional float32 arrays')
    return np.ascontiguousarray(a)


def encode(values, representation: str) -> EncodedChunk:
    a = _as_float32(values)
    raw = a.tobytes(order='C')
    if representation == 'lz4_whole':
        return EncodedChunk(lz4.frame.compress(raw), {'representation': representation, 'count': len(a)})
    if representation == 'zstd3_whole':
        import zstandard as zstd
        return EncodedChunk(zstd.ZstdCompressor(level=3).compress(raw), {'representation': representation, 'count': len(a)})
    if representation.startswith('lz4_block'):
        block = int(representation.replace('lz4_block', ''))
        pieces, offsets = [], [0]
        for i in range(0, len(a), block):
            p = lz4.frame.compress(a[i:i+block].tobytes(order='C'))
            pieces.append(p); offsets.append(offsets[-1] + len(p))
        return EncodedChunk(b''.join(pieces), {'representation': representation, 'count': len(a), 'block_size': block, 'offsets': offsets})
    raise ValueError(f'unknown representation: {representation}')


def decode(payload: bytes, metadata: dict) -> np.ndarray:
    rep = metadata['representation']; count = int(metadata['count'])
    if rep == 'lz4_whole': raw = lz4.frame.decompress(payload)
    elif rep == 'zstd3_whole':
        import zstandard as zstd
        raw = zstd.ZstdDecompressor().decompress(payload)
    elif rep.startswith('lz4_block'):
        offsets = metadata['offsets']; out = bytearray()
        for i in range(len(offsets)-1):
            out.extend(lz4.frame.decompress(payload[offsets[i]:offsets[i+1]]))
        raw = bytes(out)
    else: raise ValueError(f'unknown representation: {rep}')
    arr = np.frombuffer(raw, dtype='<f4', count=count)
    return arr.copy()


def metadata_bytes(metadata: dict) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8')


def metadata_from_bytes(data: bytes) -> dict:
    return json.loads(data.decode('utf-8'))
