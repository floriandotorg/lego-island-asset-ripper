import io
import logging
import struct
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)

HEADER_SIZE = 8
CHUNK_HEADER_SIZE = 14


class FLC:
    # spell-checker: disable
    class ChunkType(IntEnum):
        CEL_DATA = 3
        COLOR_256 = 4
        DELTA_FLC = 7
        COLOR_64 = 11
        DELTA_FLI = 12
        BLACK = 13
        BYTE_RUN = 15
        FLI_COPY = 16
        PSTAMP = 18
        DTA_BRUN = 25
        DTA_COPY = 26
        DTA_LC = 27
        LABEL = 31
        BMP_MASK = 32
        MLEV_MASK = 33
        SEGMENT = 34
        KEY_IMAGE = 35
        KEY_PAL = 36
        REGION = 37
        WAVE = 38
        USERSTRING = 39
        RGN_MASK = 40
        LABELEX = 41
        SHIFT = 42
        PATHMAP = 43
        PREFIX_TYPE = 0xF100
        SCRIPT_CHUNK = 0xF1E0
        FRAME_TYPE = 0xF1FA
        SEGMENT_TABLE = 0xF1FB
        HUFFMAN_TABLE = 0xF1FC

    # spell-checker: enable

    @dataclass
    class Color:
        r: int
        g: int
        b: int

        def __bytes__(self) -> bytes:
            return struct.pack("<BBB", self.r, self.g, self.b)

    def __init__(self, file: io.BufferedIOBase):
        self._file = file
        self._frames: list[bytes] = []
        self._palette: list[FLC.Color] = [FLC.Color(0, 0, 0)] * 256
        size, type, frames, self._width, self._height, self._delay_ms = struct.unpack("<IHHHH4xI108x", self._file.read(128))
        logger.debug(f"{size=:x} {type=:x} {frames=} {self._width=} {self._height=}")
        if type != 0xAF12:
            raise ValueError(f"Invalid FLC file: {type:x}")
        for _ in range(frames):
            self._read_chunk()

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def frames(self) -> list[bytes]:
        return self._frames

    def fps(self) -> int:
        return 1000 // self._delay_ms

    def _read_chunk(self) -> None:
        chunk_size, chunk_type = struct.unpack("<IH", self._file.read(6))
        logger.debug(f"{FLC.ChunkType(chunk_type).name=}")
        if chunk_type == FLC.ChunkType.FRAME_TYPE:
            chunks, must_be_zero = struct.unpack("<H8s", self._file.read(10))
            if must_be_zero != b"\x00\x00\x00\x00\x00\x00\x00\x00":
                raise ValueError(f"Invalid FLC file: {must_be_zero}")
            if chunks == 0:
                self._frames.append(self._frames[-1])
                return
            for _ in range(chunks):
                self._read_chunk()
        elif chunk_type == FLC.ChunkType.COLOR_256:
            packets = struct.unpack("<H", self._file.read(2))[0]
            n = 0
            for _ in range(packets):
                skip, count = struct.unpack("<BB", self._file.read(2))
                n += skip
                if count == 0:
                    count = 256
                for _ in range(count):
                    self._palette[n] = FLC.Color(*struct.unpack("<BBB", self._file.read(3)))
                    n += 1
        elif chunk_type == FLC.ChunkType.BYTE_RUN:
            frame = bytearray()
            for _ in range(self._height):
                pixels = 0
                self._file.seek(1, io.SEEK_CUR)
                while pixels < self._width:
                    count = struct.unpack("<b", self._file.read(1))[0]
                    if count == 0:
                        raise ValueError("Invalid FLC file: count is 0")
                    elif count < 0:
                        for byte in self._file.read(-count):
                            frame.extend(bytes(self._palette[byte]))
                    else:
                        frame.extend(bytes(self._palette[struct.unpack("<B", self._file.read(1))[0]]) * count)
                    pixels += abs(count)
            self._frames.append(frame)
        elif chunk_type == FLC.ChunkType.DELTA_FLC:
            frame = bytearray(self._frames[-1])
            lines = struct.unpack("<H", self._file.read(2))[0]
            line = 0
            for _ in range(lines):
                pixel = 0
                while True:
                    opcode = struct.unpack("<H", self._file.read(2))[0]
                    code = opcode >> 14
                    if code == 0b00:
                        packets = opcode
                        break
                    elif code == 0b10:
                        raise ValueError("Invalid FLC file: unsupported opcode")
                        # last_value = opcode & 0xFF
                    elif code == 0b11:
                        line -= opcode - 2**16
                    else:
                        raise ValueError("Invalid FLC file: undefined opcode")
                for _ in range(packets):
                    skip, count = struct.unpack("<Bb", self._file.read(2))
                    pixel += skip
                    if count < 0:
                        p1, p2 = struct.unpack("<BB", self._file.read(2))
                        for _ in range(-count):
                            pos = (line * self._width + pixel) * 3
                            frame[pos : pos + 6] = bytes(self._palette[p1]) + bytes(self._palette[p2])
                            pixel += 2
                    elif count > 0:
                        p = self._file.read(count * 2)
                        for n in range(0, count * 2, 2):
                            pos = (line * self._width + pixel + n) * 3
                            frame[pos : pos + 6] = bytes(self._palette[p[n]]) + bytes(self._palette[p[n + 1]])
                        pixel += count * 2
                    else:
                        raise ValueError("Invalid FLC file: count is 0")
                line += 1
            if len(frame) != len(self._frames[0]):
                raise ValueError(f"Error: frame length mismatch {len(frame)} != {len(self._frames[0])}")
            self._frames.append(frame)
        else:
            raise ValueError(f"Invalid FLC file, invalid chunk type: {FLC.ChunkType(chunk_type).name}")
