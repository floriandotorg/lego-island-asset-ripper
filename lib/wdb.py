import io
import logging
import struct
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class WDB:
    @dataclass
    class Gif:
        title: str
        width: int
        height: int
        image: bytes

    class Shading(IntEnum):
        WireFrame = 0
        UnlitFlat = 1
        Flat = 2
        Gouraud = 3
        Phong = 4

    _images: list[Gif] = []
    _textures: list[Gif] = []
    _models: list[Gif] = []

    @property
    def images(self) -> list[Gif]:
        return self._images

    @property
    def textures(self) -> list[Gif]:
        return self._textures

    @property
    def models(self) -> list[Gif]:
        return self._models

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

    def _read_vertex(self) -> tuple[float, float, float]:
        x, y, z = struct.unpack("<fff", self._file.read(12))
        return x, y, z

    def _read_vertices(self, count) -> list[tuple[float, float, float]]:
        vertices = []
        for _ in range(count):
            vertices.append(self._read_vertex())
        return vertices

    def _read_animation_tree(self):
        animation_data_name = self._read_str()
        logger.debug(f"{animation_data_name=}")

        num_translation_keys = struct.unpack("<H", self._file.read(2))[0]
        logger.debug(f"{num_translation_keys=}")

        for _ in range(num_translation_keys):
            time_and_flags = struct.unpack("<I", self._file.read(4))[0]
            flags = time_and_flags >> 24
            time = time_and_flags & 0xFFFFFF

            some_vertex = self._read_vertex()
            if some_vertex[0] > 1e-05 or some_vertex[0] < -1e-05 or some_vertex[1] > 1e-05 or some_vertex[1] < -1e-05 or some_vertex[2] > 1e-05 or some_vertex[2] < -1e-05:
                flags |= 1

            logger.debug(f"{flags=} {time=}")
            logger.debug(f"{some_vertex=}")

        num_rotation_keys = struct.unpack("<H", self._file.read(2))[0]
        logger.debug(f"{num_rotation_keys=}")

        for _ in range(num_rotation_keys):
            time_and_flags = struct.unpack("<I", self._file.read(4))[0]
            flags = time_and_flags >> 24
            time = time_and_flags & 0xFFFFFF

            angle = struct.unpack("<f", self._file.read(4))[0]

            some_vertex = self._read_vertex()
            if angle != 1.0:
                flags |= 1

            logger.debug(f"{flags=} {time=} {angle=} {some_vertex=}")

        num_scale_keys = struct.unpack("<H", self._file.read(2))[0]
        logger.debug(f"{num_scale_keys=}")

        for _ in range(num_scale_keys):
            time_and_flags = struct.unpack("<I", self._file.read(4))[0]
            flags = time_and_flags >> 24
            time = time_and_flags & 0xFFFFFF

            some_vertex = self._read_vertex()
            if some_vertex[0] > 1.00001 or some_vertex[0] < 0.99999 or some_vertex[1] > 1.00001 or some_vertex[1] < 0.99999 or some_vertex[2] > 1.00001 or some_vertex[2] < 0.99999:
                flags |= 1

            logger.debug(f"{flags=} {time=} {some_vertex=}")

        num_morph_keys = struct.unpack("<H", self._file.read(2))[0]
        logger.debug(f"{num_morph_keys=}")

        for _ in range(num_morph_keys):
            time_and_flags = struct.unpack("<I", self._file.read(4))[0]
            flags = time_and_flags >> 24
            time = time_and_flags & 0xFFFFFF

            some_bool = struct.unpack("<b", self._file.read(1))[0]
            logger.debug(f"{flags=} {time=} {some_bool=}")

        num_children = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{num_children=}")

        for _ in range(num_children):
            self._read_animation_tree()

    def _read_lod(self):
        unknown8 = struct.unpack("<I", self._file.read(4))[0]
        if unknown8 & 0xffffff04:
            raise Exception(f"{unknown8=:08x}")

        num_meshes = struct.unpack("<I", self._file.read(4))[0]
        if not num_meshes:
            # Clear Flag bit4?
            raise Exception(f"{num_meshes=}")

        # Set Flag bit4?
        num_verts, num_normals = struct.unpack("<HH", self._file.read(4))
        num_normals //= 2
        num_text_verts = struct.unpack("<I", self._file.read(4))[0]

        vertices = self._read_vertices(num_verts)
        normals = self._read_vertices(num_normals)

        uv_coordinates = []
        for _ in range(num_text_verts):
            uv_coordinates.append(struct.unpack("<ff", self._file.read(8)))

        for _ in range(num_meshes):
            num_polys, num_mesh_verts = struct.unpack("<HH", self._file.read(4))
            vertex_indices = [struct.unpack("<III", self._file.read(12)) for _ in range(num_polys)]
            num_texture_indices = struct.unpack("<I", self._file.read(4))[0]
            if num_texture_indices > 0:
                assert num_texture_indices == num_polys * 3
                texture_indices = [struct.unpack("<III", self._file.read(12)) for _ in range(num_polys)]

            red, green, blue, alpha, shading = struct.unpack("<bbbfb3x", self._file.read(3 + 4 + 4))
            texture_name = self._read_str()
            material_name = self._read_str()
            shading = WDB.Shading(shading)
            logger.debug(f"{texture_name=:<30} ({len(texture_name)=:<3}), {material_name=:<30}")

    def __init__(self, file: io.BufferedIOBase):
        self._file = file
        num_worlds = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{num_worlds=}")

        parts_offsets: list[int] = []
        models_offsets: list[int] = []
        for _ in range(num_worlds):
            world_name = self._read_str()
            logger.debug(f"{world_name=}")

            num_parts = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{num_parts=}")

            for _ in range(num_parts):
                world_name = self._read_str()
                logger.debug(f"{world_name=}")

                item_size, offset = struct.unpack("<II", self._file.read(8))
                logger.debug(f"{item_size=} {offset=}")

                parts_offsets.append(offset)

            num_models = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{num_models=}")

            for _ in range(num_models):
                model_name = self._read_str()
                logger.debug(f"{model_name=}")

                size, offset = struct.unpack("<II", self._file.read(8))
                logger.debug(f"{size=} {offset=}")

                presenter_name = self._read_str()
                logger.debug(f"{presenter_name=}")

                location_x, location_y, location_z, direction_x, direction_y, direction_z, up_x, up_y, up_z = struct.unpack("<fffffffffx", self._file.read(37))
                logger.debug(f"{location_x=} {location_y=} {location_z=} {direction_x=} {direction_y=} {direction_z=} {up_x=} {up_y=} {up_z=}")

                models_offsets.append(offset)

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

        for offset in models_offsets:
            self._file.seek(offset, io.SEEK_SET)

            version = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{version=}")

            if version != 19:
                raise ValueError(f"Invalid version: {version}")

            texture_info_offset = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{texture_info_offset=}")

            num_rois = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{num_rois=}")

            num_animations = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{num_animations=}")

            for _ in range(num_animations):
                animation_name = self._read_str()
                logger.debug(f"{animation_name=}")

                unknown = struct.unpack("<I", self._file.read(4))[0]
                logger.debug(f"{unknown=}")

                raise NotImplementedError("Animations were apparently never used")

            duration = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{duration=}")

            self._read_animation_tree()

            model_name = self._read_str()
            logger.debug(f"{model_name=}")

            center = self._read_vertex()
            logger.debug(f"{center=}")

            radius = struct.unpack("<f", self._file.read(4))[0]
            logger.debug(f"{radius=}")

            min = self._read_vertex()
            logger.debug(f"{min=}")

            max = self._read_vertex()
            logger.debug(f"{max=}")

            texture_name = self._read_str()
            logger.debug(f"{texture_name=}")

            defined_elsewhere = struct.unpack("<b", self._file.read(1))[0]
            logger.debug(f"{defined_elsewhere=}")

            if defined_elsewhere != 0:
                roi_name = model_name.rstrip("0123456789")
                logger.debug(f"{roi_name=}")
            else:
                num_lods = struct.unpack("<I", self._file.read(4))[0]
                logger.debug(f"{num_lods=}")
                if num_lods != 0:
                    end_component_offset = struct.unpack("<I", self._file.read(4))[0]
                    for _ in range(num_lods):
                        self._read_lod()

            self._file.seek(offset + texture_info_offset, io.SEEK_SET)
            num_textures, skip_textures = struct.unpack("<II", self._file.read(8))
            logger.debug(f"{num_textures=} {skip_textures=}")

            for _ in range(num_textures):
                texture = self._read_gif()
                self._models.append(texture)

                if texture.title.startswith("^"):
                    self._models.append(self._read_gif(title=texture.title[1:]))

            # world_name = self._read_str()
            # logger.debug(f"{world_name=}")

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
