import struct

import pytest

from ort_gait.packet import (
    FrameReassembler,
    PacketError,
    fragment_jpeg,
    pack_quest_frame,
)


def test_udp_fragments_reassemble_out_of_order() -> None:
    jpeg = bytes(range(256)) * 20
    packets = list(fragment_jpeg(1, 42, jpeg, chunk_bytes=1200))
    reassembler = FrameReassembler(2, timeout_s=0.25, max_frame_bytes=10_000)

    result = None
    for packet in reversed(packets):
        result = reassembler.push(packet, "10.0.0.2", now=1.0)
    assert result == (1, jpeg)
    assert reassembler.inflight_count == 0


def test_unknown_camera_index_is_rejected() -> None:
    packet = next(iter(fragment_jpeg(3, 1, b"jpeg", chunk_bytes=1200)))
    reassembler = FrameReassembler(2, timeout_s=0.25, max_frame_bytes=100)
    with pytest.raises(PacketError):
        reassembler.push(packet, "10.0.0.2", now=0.0)


def test_incomplete_frame_expires() -> None:
    packets = list(fragment_jpeg(0, 1, b"a" * 2000, chunk_bytes=1200))
    reassembler = FrameReassembler(1, timeout_s=0.25, max_frame_bytes=3000)
    assert reassembler.push(packets[0], "10.0.0.2", now=0.0) is None
    assert reassembler.inflight_count == 1

    next_frame = next(iter(fragment_jpeg(0, 2, b"ok", chunk_bytes=1200)))
    assert reassembler.push(next_frame, "10.0.0.2", now=0.3) == (0, b"ok")
    assert reassembler.inflight_count == 0


def test_quest_frame_has_only_four_byte_length_before_jpeg() -> None:
    jpeg = b"\xff\xd8payload\xff\xd9"
    message = pack_quest_frame(jpeg)
    assert struct.unpack("!I", message[:4])[0] == len(jpeg)
    assert message[4:] == jpeg
