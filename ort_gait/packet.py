from __future__ import annotations

from dataclasses import dataclass, field
import struct
import time
from typing import Iterable


MAGIC = b"ORTG"
VERSION = 1
UDP_HEADER = struct.Struct("!4sBBIHH")
TCP_LENGTH = struct.Struct("!I")


class PacketError(ValueError):
    pass


@dataclass(frozen=True)
class CameraChunk:
    camera_index: int
    frame_id: int
    chunk_count: int
    chunk_index: int
    payload: bytes


def fragment_jpeg(
    camera_index: int,
    frame_id: int,
    jpeg: bytes,
    chunk_bytes: int,
) -> Iterable[bytes]:
    if not 0 <= camera_index <= 255:
        raise PacketError("camera_index exceeds one-byte protocol field")
    if not jpeg:
        raise PacketError("cannot fragment an empty JPEG")
    count = (len(jpeg) + chunk_bytes - 1) // chunk_bytes
    if count > 65535:
        raise PacketError("JPEG requires too many UDP chunks")
    for index in range(count):
        start = index * chunk_bytes
        payload = jpeg[start : start + chunk_bytes]
        yield UDP_HEADER.pack(
            MAGIC,
            VERSION,
            camera_index,
            frame_id & 0xFFFFFFFF,
            count,
            index,
        ) + payload


def parse_chunk(packet: bytes) -> CameraChunk:
    if len(packet) <= UDP_HEADER.size:
        raise PacketError("UDP packet is shorter than its header")
    magic, version, camera, frame, count, index = UDP_HEADER.unpack_from(packet)
    if magic != MAGIC or version != VERSION:
        raise PacketError("unknown UDP camera protocol")
    if count == 0 or index >= count:
        raise PacketError("invalid UDP chunk indices")
    return CameraChunk(camera, frame, count, index, packet[UDP_HEADER.size :])


@dataclass
class _PartialFrame:
    chunk_count: int
    created_at: float
    chunks: dict[int, bytes] = field(default_factory=dict)
    byte_count: int = 0


class FrameReassembler:
    """Bounded, timeout-based latest-frame UDP reassembly."""

    def __init__(
        self,
        camera_count: int,
        timeout_s: float,
        max_frame_bytes: int,
        max_inflight: int = 64,
    ) -> None:
        self._camera_count = camera_count
        self._timeout_s = timeout_s
        self._max_frame_bytes = max_frame_bytes
        self._max_inflight = max_inflight
        self._frames: dict[tuple[str, int, int], _PartialFrame] = {}

    def push(
        self,
        packet: bytes,
        peer_ip: str,
        now: float | None = None,
    ) -> tuple[int, bytes] | None:
        moment = time.monotonic() if now is None else now
        self._discard_expired(moment)
        chunk = parse_chunk(packet)
        if chunk.camera_index >= self._camera_count:
            raise PacketError("camera index is not configured")

        key = (peer_ip, chunk.camera_index, chunk.frame_id)
        partial = self._frames.get(key)
        if partial is None:
            if len(self._frames) >= self._max_inflight:
                oldest = min(self._frames, key=lambda item: self._frames[item].created_at)
                del self._frames[oldest]
            partial = _PartialFrame(chunk.chunk_count, moment)
            self._frames[key] = partial
        elif partial.chunk_count != chunk.chunk_count:
            del self._frames[key]
            raise PacketError("chunk count changed inside one frame")

        if chunk.chunk_index not in partial.chunks:
            partial.chunks[chunk.chunk_index] = chunk.payload
            partial.byte_count += len(chunk.payload)
        if partial.byte_count > self._max_frame_bytes:
            del self._frames[key]
            raise PacketError("camera JPEG exceeds configured safety limit")

        if len(partial.chunks) != partial.chunk_count:
            return None
        jpeg = b"".join(partial.chunks[index] for index in range(partial.chunk_count))
        del self._frames[key]
        return chunk.camera_index, jpeg

    def _discard_expired(self, now: float) -> None:
        expired = [
            key
            for key, frame in self._frames.items()
            if now - frame.created_at > self._timeout_s
        ]
        for key in expired:
            del self._frames[key]

    @property
    def inflight_count(self) -> int:
        return len(self._frames)


def pack_quest_frame(jpeg: bytes) -> bytes:
    """Unity contract: exactly 4-byte network-order length followed by JPEG."""
    if not jpeg or len(jpeg) > 0xFFFFFFFF:
        raise PacketError("invalid Quest JPEG length")
    return TCP_LENGTH.pack(len(jpeg)) + jpeg
