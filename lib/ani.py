from lib.animation import AnimationNode, ReaderHelper

from dataclasses import dataclass
import struct
import io
import logging

logger = logging.getLogger(__name__)

class Ani:
    def __init__(self, file: io.BufferedIOBase | ReaderHelper):
        reader = ReaderHelper.from_reader(file)
        magic = reader.read_u32()
        if magic != 0x11:
            raise Exception("Unknown magic")
        unkn1 = reader.read_float()
        logger.debug(f"{unkn1=}")
        unkn_vector = reader.read_vertex()
        logger.debug(f"{unkn_vector=}")
        parse_scene = reader.read_s32()
        logger.debug(f"{parse_scene=}")
        val3 = reader.read_s32()
        logger.debug(f"{val3=}")
        animation_count = reader.read_u32()
        logger.debug(f"{animation_count=}")
        for _ in range(animation_count):
            name = reader.read_str()
            logger.debug(f"{name=}")
            if name:
                unkn2 = reader.read_u32()
                logger.debug(f"{unkn2=}")
            duration = reader.read_s32()
            logger.debug(f"{duration=}")
            assert parse_scene == 0, "parse_scene not implemented"
            animation = AnimationNode.read(reader)
            logger.debug(f"{animation=}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with open("sba001bu.ani", "rb") as f:
        ani = Ani(f)

