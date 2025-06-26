from lib.animation import Animation, ReaderHelper

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
        animation = Animation(reader, parse_scene)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with open("sba001bu.ani", "rb") as f:
        ani = Ani(f)
