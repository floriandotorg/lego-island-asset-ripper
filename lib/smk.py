import io
import logging
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)

HEADER_SIZE = 8
CHUNK_HEADER_SIZE = 14


class SMK:
    @dataclass
    class Color:
        r: int
        g: int
        b: int

        def __bytes__(self) -> bytes:
            return struct.pack("<BBB", self.r, self.g, self.b)

    # spell-checker: ignore palmap
    _palmap = bytes([0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C, 0x41, 0x45, 0x49, 0x4D, 0x51, 0x55, 0x59, 0x5D, 0x61, 0x65, 0x69, 0x6D, 0x71, 0x75, 0x79, 0x7D, 0x82, 0x86, 0x8A, 0x8E, 0x92, 0x96, 0x9A, 0x9E, 0xA2, 0xA6, 0xAA, 0xAE, 0xB2, 0xB6, 0xBA, 0xBE, 0xC3, 0xC7, 0xCB, 0xCF, 0xD3, 0xD7, 0xDB, 0xDF, 0xE3, 0xE7, 0xEB, 0xEF, 0xF3, 0xF7, 0xFB, 0xFF])

    def __init__(self, file: io.BufferedIOBase):
        self._file = file
        signature, self._width, self._height, self._frames, self._framerate, flags = struct.unpack("<4sIIIiI80x", self._file.read(104))
        if signature != b"SMK2":
            raise ValueError(f"Invalid SMK file: {signature}")
        if flags != 0:
            raise ValueError(f"Unsupported flags: {flags:x}")
        if self._framerate > 0:
            self._framerate = 1_000 / self._framerate
        elif self._framerate < 0:
            self._framerate = 100_000 / -self._framerate
        else:
            self._framerate = 10
        logger.debug(f"{self._width=} {self._height=} {self._frames=} {self._framerate=}")
        frame_sizes = struct.unpack("<" + "I" * self._frames, self._file.read(self._frames * 4))
        is_keyframe = [frame_size & 0x01 == 1 for frame_size in frame_sizes]
        frame_sizes = tuple(frame_size & ~0x03 for frame_size in frame_sizes)
        logger.debug(f"{is_keyframe=} {frame_sizes=}")
        frame_types = struct.unpack("<" + "b" * self._frames, self._file.read(self._frames))
        if any(type & ~0x01 != 0 for type in frame_types):
            raise ValueError("Audio is not supported")
        has_palette = [frame_type & 0x01 == 1 for frame_type in frame_types]
        self._file.seek(26_862, io.SEEK_CUR)
        self._palette = [self.Color(0, 0, 0)] * 256
        self._read_palette()

    def _read_palette(self) -> None:
        prev_palette = list(self._palette)
        palette_end = self._file.tell() + self._file.read(1)[0] * 4 - 1
        n = 0
        while self._file.tell() < palette_end:
            block = self._file.read(1)[0]
            if block & 0x80:
                n += (block & 0x7F) + 1
                logger.debug(f"skip {(block & 0x7F) + 1} colors")
            elif block & 0x40:
                s = self._file.read(1)[0]
                for i in range((block & 0x3F) + 1):
                    self._palette[n] = prev_palette[s + i]
                    n += 1
                logger.debug(f"read {(block & 0x3F) + 1} colors")
            else:
                b = (block & 0x3F) + 1
                g = self._file.read(1)[0] & 0x3F
                r = self._file.read(1)[0] & 0x3F
                self._palette[n] = SMK.Color(self._palmap[r], self._palmap[g], self._palmap[b])
                n += 1
                logger.debug(f"read {SMK.Color(self._palmap[r], self._palmap[g], self._palmap[b])}")
