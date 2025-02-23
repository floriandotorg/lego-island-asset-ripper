import io
import logging
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class WDB:
    @dataclass
    class Gif:
        title: str
        width: int
        height: int
        image: bytes

    _images: list[Gif] = []
    _textures: list[Gif] = []

    @property
    def images(self) -> list[Gif]:
        return self._images

    @property
    def textures(self) -> list[Gif]:
        return self._textures

    def _read_gif(self, title: str | None = None) -> Gif:
        if title is None:
            title = self._read_str()
        logger.debug(f"{title=}")

        width, height, num_colors = struct.unpack("<III", self._file.read(12))
        logger.debug(f"{width=} {height=} {num_colors=}")

        colors: list[bytes] = []
        for _ in range(num_colors):
            r, g, b = struct.unpack("<BBB", self._file.read(3))
            colors.append(bytes([r, g, b]))

        image = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                pixel = struct.unpack("<B", self._file.read(1))[0]
                image[y * width * 3 + x * 3 : y * width * 3 + x * 3 + 3] = colors[pixel]

        return self.Gif(title, width, height, image)

    def _read_str(self) -> str:
        length = struct.unpack("<I", self._file.read(4))[0]
        return self._file.read(length).decode("ascii").rstrip("\x00")

    def __init__(self, file: io.BufferedIOBase):
        self._file = file
        num_groups = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{num_groups=}")

        parts_offsets: list[int] = []
        models_offsets: list[int] = []
        for _ in range(num_groups):
            name = self._read_str()
            logger.debug(f"{name=}")

            for is_model in (False, True):
                num_objects = struct.unpack("<I", self._file.read(4))[0]
                logger.debug(f"{num_objects=}")

                for _ in range(num_objects):
                    name = self._read_str()
                    logger.debug(f"{name=}")

                    item_size, offset = struct.unpack("<II", self._file.read(8))
                    logger.debug(f"{item_size=} {offset=}")

                    if is_model:
                        presenter_name = self._read_str()
                        logger.debug(f"{presenter_name=}")
                        location_x, location_y, location_z, direction_x, direction_y, direction_z, up_x, up_y, up_z = struct.unpack("<fffffffffx", self._file.read(37))
                        logger.debug(f"{location_x=} {location_y=} {location_z=} {direction_x=} {direction_y=} {direction_z=} {up_x=} {up_y=} {up_z=}")
                        models_offsets.append(offset)
                    else:
                        parts_offsets.append(offset)

        gif_chunk_size, num_frames = struct.unpack("<II", self._file.read(8))
        logger.debug(f"{gif_chunk_size=} {num_frames=}")

        for _ in range(num_frames):
            self._images.append(self._read_gif())

        for offset in parts_offsets:
            self._file.seek(offset, io.SEEK_SET)
            texture_info_offset = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{texture_info_offset=:}")

            self._file.seek(offset + texture_info_offset, io.SEEK_SET)
            num_textures = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{num_textures=}")

            for _ in range(num_textures):
                texture = self._read_gif()
                self._textures.append(texture)

                if texture.title.startswith("^"):
                    self._textures.append(self._read_gif(title=texture.title[1:]))

        # model_chunk_size, chunk_size, num_bins = struct.unpack("<III", self._file.read(12))
        # logger.debug(f"{model_chunk_size=} {chunk_size=} {num_bins=}")

        # für_später = self._file.tell() - 8

        # for _ in range(num_bins):
        #     name = self._read_str()
        #     logger.debug(f"{name=}")

        #     num_models, end_bin_offset = struct.unpack("<II", self._file.read(8))
        #     logger.debug(f"{num_models=} {end_bin_offset=}")

        #     for _ in range(num_models):
        #         magic, bytes_left_in_subgroup, version = struct.unpack("<III8x", self._file.read(20))
        #         logger.debug(f"{magic=} {bytes_left_in_subgroup=} {version=}")

        #         if magic != 8:
        #             raise ValueError(f"Invalid magic number: {magic}")

        #         file_name = self._read_str()
        #         logger.debug(f"{file_name=}")

        #     self._file.seek(für_später + end_bin_offset, io.SEEK_SET)
