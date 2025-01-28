import io
import logging
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

HEADER_SIZE = 8
CHUNK_HEADER_SIZE = 14


class SI:
    class Type(IntEnum):
        Null = -1
        Video = 0x03
        Sound = 0x04
        World = 0x06
        Presenter = 0x07
        Event = 0x08
        Animation = 0x09
        Bitmap = 0x0A
        Object = 0x0B

    class Flags(IntEnum):
        LoopCache = 0x01
        NoLoop = 0x02
        LoopStream = 0x04
        Transparent = 0x08
        Unknown = 0x20

    class FileType(IntEnum):
        WAV = 0x56415720
        STL = 0x4C545320
        FLC = 0x434C4620
        SMK = 0x4B4D5320
        OBJ = 0x4A424F20
        TVE = 0x54564520

    class ChunkFlags(IntEnum):
        Split = 0x10
        End = 0x02

    @dataclass
    class Object:
        type: "SI.Type"
        presenter: str
        name: str
        id: int
        flags: int
        duration: int
        loops: int
        location: tuple[float, float, float]
        direction: tuple[float, float, float]
        up: tuple[float, float, float]
        filename: Optional[str]
        file_type: Optional["SI.FileType"]
        volume: Optional[int]
        data: bytearray = field(default_factory=bytearray)
        chunk_sizes: list[int] = field(default_factory=list)

        def open(self) -> io.BytesIO:
            return io.BytesIO(self.data)

    class Version(IntEnum):
        Version2_1 = 0x00010002
        Version2_2 = 0x00020002

    def __init__(self, file: io.BufferedIOBase):
        self._file = file
        self._buffer_size = 0
        # self._offset_list: list[int] = []
        self._object_list: dict[int, SI.Object] = {}
        self._version: Optional[SI.Version] = None
        self._split_chunk_bytes_written = 0
        self._read_chunk()

    @property
    def object_list(self) -> dict[int, "SI.Object"]:
        return self._object_list

    def _read_null_terminated_string(self) -> str:
        result = bytearray()
        while (b := self._file.read(1)) != b"\0":
            result.extend(b)
        return result.decode("ascii")

    def _read_uint16(self) -> int:
        return struct.unpack("<H", self._file.read(2))[0]

    def _read_uint32(self) -> int:
        return struct.unpack("<I", self._file.read(4))[0]

    def _read_chunk(self) -> None:
        pos = self._file.tell()
        magic, size = struct.unpack("<4sI", self._file.read(8))
        end = self._file.tell() + size

        logger.debug(f"{pos=:08x}, {magic=}, {size=}")

        match magic:
            case b"RIFF":
                if self._file.read(4) != b"OMNI":
                    raise ValueError("Invalid SI file")
            case b"MxHd":
                self._version, self._buffer_size, buffer_count = struct.unpack("<III", self._file.read(12))
                logger.debug(f"{self._version=:08x}, {self._buffer_size=}, {buffer_count=}")
            case b"pad ":
                self._file.seek(size, io.SEEK_CUR)
            case b"MxOf":
                self._file.seek(4, io.SEEK_CUR)
                real_count = size // 4 - 1
                self._file.seek(real_count * 4, io.SEEK_CUR)
                # for _ in range(real_count):
                #     offset = self._read_uint32()
                #     self._offset_list.append(offset)
            case b"LIST":
                if self._file.read(4) == b"MxCh":
                    if self._version == SI.Version.Version2_1:
                        raise ValueError("Version 2.1 is not supported")
                        # self.file.seek(4, io.SEEK_CUR)
                    list_variation = self._read_uint32()
                    if list_variation == b"Act\0" or list_variation == b"RAND":
                        if list_variation == b"RAND":
                            self._file.seek(5, io.SEEK_CUR)
                        self._file.seek(2 * self._read_uint32(), io.SEEK_CUR)
            case b"MxSt" | b"MxDa" | b"WAVE" | b"fmt_" | b"data" | b"OMNI":
                pass
            case b"MxOb":
                type = SI.Type(self._read_uint16())
                presenter = self._read_null_terminated_string()
                self._file.seek(4, io.SEEK_CUR)  # unknown 1
                name = self._read_null_terminated_string()
                id, flags, duration, loops, *coords = struct.unpack("<2I4x2I9d", self._file.read(92))
                self._file.seek(self._read_uint16(), io.SEEK_CUR)
                filename: Optional[str] = None
                file_type: Optional[SI.FileType] = None
                volume: Optional[int] = None
                if type != SI.Type.Presenter and type != SI.Type.World and type != SI.Type.Animation:
                    filename = self._read_null_terminated_string()
                    self._file.seek(12, io.SEEK_CUR)  # unknown 26 - 28
                    file_type = SI.FileType(self._read_uint32())
                    self._file.seek(8, io.SEEK_CUR)  # unknown 29 - 30
                    if type == SI.FileType.WAV:
                        volume = self._read_uint32()
                obj = SI.Object(
                    type,
                    presenter,
                    name,
                    id,
                    flags,
                    duration,
                    loops,
                    location=tuple(coords[:3]),
                    direction=tuple(coords[3:6]),
                    up=tuple(coords[6:]),
                    filename=filename,
                    file_type=file_type,
                    volume=volume,
                )
                self._object_list[id] = obj
                logger.debug(obj)
            case b"MxCh":
                flags, id, total_size = struct.unpack("<HI4xI", self._file.read(CHUNK_HEADER_SIZE))
                size_without_header = size - CHUNK_HEADER_SIZE
                data = self._file.read(size_without_header)
                if not flags & SI.ChunkFlags.End:
                    obj = self._object_list[id]
                    obj.data.extend(data)
                    if self._split_chunk_bytes_written == 0:
                        obj.chunk_sizes.append(total_size)
                    if flags & SI.ChunkFlags.Split:
                        self._split_chunk_bytes_written += size_without_header
                        if self._split_chunk_bytes_written >= total_size:
                            self._split_chunk_bytes_written = 0
            case _:
                raise ValueError(f"Unknown chunk type: {magic}")

        while (self._file.tell() + HEADER_SIZE) < end:
            if self._buffer_size > 0:
                offset = self._file.tell() % self._buffer_size
                if offset + HEADER_SIZE > self._buffer_size:
                    self._file.seek(self._buffer_size - offset, io.SEEK_CUR)

            self._read_chunk()

        self._file.seek(end, io.SEEK_SET)

        if size % 2 == 1:
            self._file.seek(1, io.SEEK_CUR)
