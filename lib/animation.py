from dataclasses import dataclass
import io
import logging
import struct

logger = logging.getLogger(__name__)


class ReaderHelper:
    _reader: io.BufferedIOBase

    def read_str(self) -> str:
        length = struct.unpack("<I", self._reader.read(4))[0]
        return self._reader.read(length).decode("ascii").rstrip("\x00")

    def read_vertex(self) -> tuple[float, float, float]:
        x, y, z = struct.unpack("<fff", self._reader.read(12))
        return -x, y, z

    def read_u32(self) -> int:
        return struct.unpack("<I", self._reader.read(4))[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self._reader.read(2))[0]

    def read_u8(self) -> int:
        return struct.unpack("<B", self._reader.read(1))[0]

    def read_s32(self) -> int:
        return struct.unpack("<i", self._reader.read(4))[0]

    def read_s16(self) -> int:
        return struct.unpack("<h", self._reader.read(2))[0]

    def read_float(self) -> float:
        return struct.unpack("<f", self._reader.read(4))[0]

    def __init__(self, reader: io.BufferedIOBase):
        self._reader = reader

    @staticmethod
    def from_reader(reader: "io.BufferedIOBase | ReaderHelper") -> "ReaderHelper":
        if isinstance(reader, ReaderHelper):
            return reader
        return ReaderHelper(reader)


@dataclass
class AnimationNode:
    name: str
    translation_keys: list["AnimationNode.VertexKey"]
    rotation_keys: list["AnimationNode.RotationKey"]
    scale_keys: list["AnimationNode.VertexKey"]
    morph_keys: list["AnimationNode.MorphKey"]
    children: list["AnimationNode"]

    @dataclass
    class VertexKey:
        time: int
        flags: int
        vertex: tuple[float, float, float]

    @dataclass
    class RotationKey:
        time: int
        flags: int
        quaternion: tuple[float, float, float, float]

    @dataclass
    class MorphKey:
        time: int
        flags: int
        bool: bool

    @staticmethod
    def _read_time_and_flags(reader: "ReaderHelper") -> tuple[int, int]:
        time_and_flags = reader.read_u32()
        flags = time_and_flags >> 24
        time = time_and_flags & 0xFFFFFF
        return (time, flags)

    @staticmethod
    def read(file: "io.BufferedIOBase | ReaderHelper") -> "AnimationNode":
        reader = ReaderHelper.from_reader(file)
        animation_data_name = reader.read_str()
        logger.debug(f"{animation_data_name=}")

        num_translation_keys = reader.read_u16()
        logger.debug(f"{num_translation_keys=}")
        translation_keys = []
        for _ in range(num_translation_keys):
            time, flags = AnimationNode._read_time_and_flags(reader)

            some_vertex = reader.read_vertex()
            # Set flag, if the vertex is "not almost unset"
            if some_vertex[0] > 1e-05 or some_vertex[0] < -1e-05 or some_vertex[1] > 1e-05 or some_vertex[1] < -1e-05 or some_vertex[2] > 1e-05 or some_vertex[2] < -1e-05:
                flags |= 1

            logger.debug(f"{flags=} {time=}")
            logger.debug(f"{some_vertex=}")
            translation_keys.append(AnimationNode.VertexKey(time, flags, some_vertex))

        num_rotation_keys = reader.read_u16()
        logger.debug(f"{num_rotation_keys=}")
        rotation_keys = []
        for _ in range(num_rotation_keys):
            time, flags = AnimationNode._read_time_and_flags(reader)

            w = reader.read_float()
            x = -reader.read_float()
            y = reader.read_float()
            z = reader.read_float()

            # Set flag, if the vertex is "not almost unset"
            if w != 1.0:
                flags |= 1

            logger.debug(f"{flags=} {time=} {x=} {y=} {z=} {w=}")
            rotation_keys.append(AnimationNode.RotationKey(time, flags, (x, y, z, w)))

        num_scale_keys = reader.read_u16()
        logger.debug(f"{num_scale_keys=}")
        scale_keys = []
        for _ in range(num_scale_keys):
            time, flags = AnimationNode._read_time_and_flags(reader)

            some_vertex = reader.read_vertex()
            # Set flag, if the vertex is "not almost unset"
            if some_vertex[0] > 1.00001 or some_vertex[0] < 0.99999 or some_vertex[1] > 1.00001 or some_vertex[1] < 0.99999 or some_vertex[2] > 1.00001 or some_vertex[2] < 0.99999:
                flags |= 1

            logger.debug(f"{flags=} {time=} {some_vertex=}")
            scale_keys.append(AnimationNode.VertexKey(time, flags, some_vertex))

        num_morph_keys = reader.read_u16()
        logger.debug(f"{num_morph_keys=}")
        morph_keys = []
        for _ in range(num_morph_keys):
            time, flags = AnimationNode._read_time_and_flags(reader)

            some_bool = reader.read_u8() != 0
            logger.debug(f"{flags=} {time=} {some_bool=}")
            morph_keys.append(AnimationNode.MorphKey(time, flags, some_bool))

        num_children = reader.read_u32()
        logger.debug(f"{num_children=}")
        children = []
        for _ in range(num_children):
            children.append(AnimationNode.read(reader))
        return AnimationNode(animation_data_name, translation_keys, rotation_keys, scale_keys, morph_keys, children)


class Animation:
    actors: list[str]
    duration: int
    tree: AnimationNode

    def __init__(self, file: "io.BufferedIOBase | ReaderHelper", parse_scene: int):
        reader = ReaderHelper.from_reader(file)
        actor_count = reader.read_u32()
        logger.debug(f"{actor_count=}")
        self.actors = []
        for _ in range(actor_count):
            name = reader.read_str()
            logger.debug(f"{name=}")
            if name:
                unkn2 = reader.read_u32()
                logger.debug(f"{unkn2=}")
                self.actors += [name]
        self.duration = reader.read_s32()
        logger.debug(f"{self.duration=}")
        assert parse_scene == 0, "parse_scene not implemented"
        self.tree = AnimationNode.read(reader)
        logger.debug(f"{self.tree=}")
