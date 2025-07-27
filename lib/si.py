import io
import logging
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

logger = logging.getLogger(__name__)

HEADER_SIZE = 8
CHUNK_HEADER_SIZE = 14


class SI:
    class Type(IntEnum):
        Null = -1
        Object = 0
        Action = 1
        MediaAction = 2
        Anim = 3
        Sound = 4
        MultiAction = 5
        SerialAction = 6
        ParallelAction = 7
        Event = 8
        SelectAction = 9
        Still = 10
        ObjectAction = 11

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
    class Dimensions:
        width: int
        height: int

        def to_dict(self) -> dict[str, Any]:
            return {"width": self.width, "height": self.height}

    @dataclass
    class Object:
        type: "SI.Type"
        presenter: str
        name: str
        si_file: str
        id: int
        flags: int
        start_time: int
        duration: int
        loops: int
        location: tuple[float, float, float]
        direction: tuple[float, float, float]
        up: tuple[float, float, float]
        filename: Optional[str]
        file_type: Optional["SI.FileType"]
        volume: Optional[int]
        extra_data: str
        data: bytearray = field(default_factory=bytearray)
        chunk_sizes: list[int] = field(default_factory=list)
        children: list["SI.Object"] = field(default_factory=list)
        fps: Optional[int] = None
        num_frames: Optional[int] = None
        dimensions: Optional["SI.Dimensions"] = None
        color_palette: Optional[list[str]] = None
        should_export_palette: bool = False

        def open(self) -> io.BytesIO:
            return io.BytesIO(self.data)

        def to_dict(self) -> dict[str, Any]:
            return {"type": self.type, "presenter": self.presenter, "name": self.name, "siFile": self.si_file, "id": self.id, "flags": self.flags, "startTime": self.start_time, "duration": self.duration, "loops": self.loops, "location": self.location, "direction": self.direction, "up": self.up, "filename": self.filename, "fileType": self.file_type, "volume": self.volume, "extra": self.extra_data, "fps": self.fps, "numFrames": self.num_frames, "dimensions": self.dimensions.to_dict() if self.dimensions else None, "colorPalette": self.color_palette, "children": [child.to_dict() for child in self.children]}

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

    def _read_chunk(self, parents: list["SI.Object"] = []) -> None:
        pos = self._file.tell()
        magic, size = struct.unpack("<4sI", self._file.read(8))
        end = self._file.tell() + size
        current: Optional["SI.Object"] = None

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
                    list_variation = self._file.read(4)
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
                id, flags, start_time, duration, loops, *coords = struct.unpack("<5I9d", self._file.read(92))
                extra_data_length = self._read_uint16()
                if extra_data_length:
                    extra_data = self._file.read(extra_data_length)[:-1].decode("ascii")
                else:
                    extra_data = ""
                filename: Optional[str] = None
                file_type: Optional[SI.FileType] = None
                volume: Optional[int] = None
                if type != SI.Type.ParallelAction and type != SI.Type.SerialAction and type != SI.Type.SelectAction:
                    filename = self._read_null_terminated_string()
                    self._file.seek(12, io.SEEK_CUR)  # unknown 26 - 28
                    file_type = SI.FileType(self._read_uint32())
                    self._file.seek(8, io.SEEK_CUR)  # unknown 29 - 30
                    if file_type == SI.FileType.WAV:
                        volume = self._read_uint32()
                (loc_x, loc_y, loc_z) = tuple(map(float, coords[:3]))
                (dir_x, dir_y, dir_z) = tuple(map(float, coords[3:6]))
                (up_x, up_y, up_z) = tuple(map(float, coords[6:]))
                obj = SI.Object(type, presenter, name, "", id, flags, start_time, duration, loops, (loc_x, loc_y, loc_z), (dir_x, dir_y, dir_z), (up_x, up_y, up_z), filename=filename, file_type=file_type, volume=volume, extra_data=extra_data)
                self._object_list[id] = obj
                parent = parents[-1] if parents else None
                if parent:
                    parent.children.append(obj)
                current = obj
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
                raise ValueError(f"Unknown chunk type: {magic} at {pos=:08x}")

        while (self._file.tell() + HEADER_SIZE) < end:
            if self._buffer_size > 0:
                offset = self._file.tell() % self._buffer_size
                if offset + HEADER_SIZE > self._buffer_size:
                    self._file.seek(self._buffer_size - offset, io.SEEK_CUR)

            self._read_chunk([*parents, current] if current else parents)

        self._file.seek(end, io.SEEK_SET)

        if size % 2 == 1:
            self._file.seek(1, io.SEEK_CUR)
