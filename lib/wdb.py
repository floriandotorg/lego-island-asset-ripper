import io
import logging
import struct
from dataclasses import dataclass
from enum import IntEnum
from itertools import zip_longest
from typing import Optional, cast

from lib.animation import Animation

logger = logging.getLogger(__name__)


class WDB:
    @dataclass
    class Gif:
        title: str
        width: int
        height: int
        image: bytes

    class Shading(IntEnum):
        Flat = 0
        Gouraud = 1
        WireFrame = 2

    @dataclass
    class Color:
        red: int
        green: int
        blue: int
        alpha: float

    @dataclass
    class Model:
        roi: "WDB.Roi"
        animation: Animation

    @dataclass
    class Roi:
        name: str
        lods: list["WDB.Lod"]
        reference: str | None
        children: list["WDB.Roi"]
        texture_name: str

    @dataclass
    class Lod:
        meshes: list["WDB.Mesh"]

    @dataclass
    class Mesh:
        vertices: list[tuple[float, float, float]]
        normals: list[tuple[float, float, float]]
        uvs: list[tuple[float, float]]
        indices: list[int]
        color: "WDB.Color"
        texture_name: str
        material_name: str

    @dataclass
    class Part:
        name: str
        lods: list["WDB.Lod"]

    _images: list[Gif] = []
    _part_textures: list[Gif] = []
    _model_textures: list[Gif] = []
    _models: list[Model] = []
    _parts: list[Part] = []
    _global_parts: list[Part] = []

    @property
    def images(self) -> list[Gif]:
        return self._images

    @property
    def part_textures(self) -> list[Gif]:
        return self._part_textures

    @property
    def model_textures(self) -> list[Gif]:
        return self._model_textures

    @property
    def models(self) -> list[Model]:
        return self._models

    @property
    def parts(self) -> list[Part]:
        return self._parts

    @property
    def global_parts(self) -> list[Part]:
        return self._global_parts

    def model_texture_by_name(self, texture_name: str) -> Gif:
        for texture in self._model_textures:
            if texture.title == texture_name:
                return texture
        raise KeyError(texture_name)

    def part_texture_by_name(self, texture_name: str) -> Gif:
        for texture in self._part_textures:
            if texture.title == texture_name:
                return texture
        raise KeyError(texture_name)

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
        return -x, y, z

    def _read_vertices(self, count) -> list[tuple[float, float, float]]:
        vertices = []
        for _ in range(count):
            vertices.append(self._read_vertex())
        return vertices

    def _read_lod(self) -> Optional["WDB.Lod"]:
        unknown8 = struct.unpack("<I", self._file.read(4))[0]
        if unknown8 & 0xFFFFFF04:
            return None

        num_meshes = struct.unpack("<I", self._file.read(4))[0]
        if not num_meshes:
            return WDB.Lod([])

        num_verts, num_normals = struct.unpack("<HH", self._file.read(4))
        num_normals //= 2
        num_text_verts = struct.unpack("<I", self._file.read(4))[0]

        vertices = self._read_vertices(num_verts)
        normals = self._read_vertices(num_normals)

        uv_coordinates: list[tuple[float, float]] = [cast(tuple[float, float], struct.unpack("<ff", self._file.read(8))) for _ in range(num_text_verts)]

        result = []
        for _ in range(num_meshes):
            num_polys, num_mesh_verts = struct.unpack("<HH", self._file.read(4))
            vertex_indices_packed: list[int] = [struct.unpack("<I", self._file.read(4))[0] for _ in range(num_polys * 3)]
            num_texture_indices = struct.unpack("<I", self._file.read(4))[0]
            if num_texture_indices > 0:
                assert num_texture_indices == num_polys * 3
                texture_indices = [struct.unpack("<I", self._file.read(4))[0] for _ in range(num_polys * 3)]
            else:
                texture_indices = []
            mesh_vertices: list[tuple[float, float, float]] = []
            mesh_normals = []
            mesh_uv = []
            vertex_indices = []
            for vertex_index_packed, texture_index in zip_longest(vertex_indices_packed, texture_indices):
                if vertex_index_packed & 0x80000000:
                    vertex_indices.append(len(mesh_vertices))

                    global_vertex_index = vertex_index_packed & 0x7FFF
                    mesh_vertices.append(vertices[global_vertex_index])
                    global_normal_index = (vertex_index_packed >> 16) & 0x7FFF
                    mesh_normals.append(normals[global_normal_index])
                    if texture_index is not None and uv_coordinates:
                        mesh_uv.append(uv_coordinates[texture_index])
                else:
                    vertex_indices.append(vertex_index_packed & 0x7FFF)
            for i in range(0, len(vertex_indices), 3):
                vertex_indices[i], vertex_indices[i + 2] = vertex_indices[i + 2], vertex_indices[i]
            assert len(vertex_indices) == num_polys * 3
            assert len(mesh_vertices) == num_mesh_verts, f"{len(mesh_vertices)=} != {num_mesh_verts=}"
            assert len(mesh_uv) in [0, num_mesh_verts], f"{len(mesh_uv)=} != {num_polys=}"

            red, green, blue, alpha, shading = struct.unpack("<BBBfB3x", self._file.read(3 + 4 + 4))
            texture_name = self._read_str()
            material_name = self._read_str()
            color = WDB.Color(red, green, blue, alpha)
            shading = WDB.Shading(shading)
            logger.debug(f"{texture_name=:<30} ({len(texture_name)=:<3}), {material_name=:<30}")

            result.append(WDB.Mesh(mesh_vertices, mesh_normals, mesh_uv, vertex_indices, color, texture_name, material_name))

        return WDB.Lod(result)

    def _read_roi(self, scanned_model_names: set[str], offset: int, path: str = "") -> "WDB.Roi":
        model_name = self._read_str()
        logger.debug(f"{model_name=}")

        if path:
            path += "/"
        path += model_name

        logger.debug(f"Reading '{path}'")

        if model_name in scanned_model_names:
            # TODO: Either this is okay, or determine how we can avoid this
            logger.warning(f"Already scanned model '{model_name}'!")
        scanned_model_names.add(model_name)

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

        lods: list[WDB.Lod] = []
        reference: str | None = None
        if defined_elsewhere != 0:
            roi_name = model_name.rstrip("0123456789")
            logger.debug(f"{roi_name=}")
            reference = roi_name
        else:
            num_lods = struct.unpack("<I", self._file.read(4))[0]
            logger.debug(f"{num_lods=}")
            if num_lods != 0:
                end_component_offset = struct.unpack("<I", self._file.read(4))[0]
                lods = [lod for lod in (self._read_lod() for _ in range(num_lods)) if lod is not None]
                logger.debug(f"Loaded {len(lods)} for {path}")
                self._file.seek(offset + end_component_offset)

        num_rois = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{num_rois=}")
        children = []
        for _ in range(num_rois):
            children.append(self._read_roi(scanned_model_names, offset, path))

        return WDB.Roi(model_name, lods, reference, children, texture_name)

    def _read_part(self, offset: int) -> list["WDB.Part"]:
        result = []

        texture_info_offset, num_rois = struct.unpack("<II", self._file.read(8))
        logger.debug(f"{texture_info_offset=:}")

        for _ in range(num_rois):
            roi_name = self._read_str()
            logger.debug(f"{roi_name=}")

            num_lods, roi_info_offset = struct.unpack("<II", self._file.read(8))
            logger.debug(f"{num_lods=} {roi_info_offset=}")

            lods = []
            for _ in range(num_lods):
                lod = self._read_lod()
                if lod is not None:
                    lods.append(lod)

            result.append(WDB.Part(roi_name, lods))

        self._file.seek(offset + texture_info_offset, io.SEEK_SET)
        num_textures = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{num_textures=}")

        for _ in range(num_textures):
            texture = self._read_gif()
            self._part_textures.append(texture)

            if texture.title.startswith("^"):
                self._part_textures.append(self._read_gif(title=texture.title[1:]))

        return result

    def _read_model(self) -> None:
        offset = self._file.tell()

        version = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{version=}")

        if version != 19:
            raise ValueError(f"Invalid version: {version}")

        texture_info_offset = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{texture_info_offset=}")

        num_rois = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{num_rois=}")

        animation = Animation(self._file, 0)

        scanned_model_names: set[str] = set()
        roi = self._read_roi(scanned_model_names, offset)
        self._models.append(WDB.Model(roi, animation))

        self._file.seek(offset + texture_info_offset, io.SEEK_SET)
        num_textures, skip_textures = struct.unpack("<II", self._file.read(8))
        logger.debug(f"{num_textures=} {skip_textures=}")

        for _ in range(num_textures):
            texture = self._read_gif()
            self._model_textures.append(texture)

            if texture.title.startswith("^"):
                self._model_textures.append(self._read_gif(title=texture.title[1:]))

    def __init__(self, file: io.BufferedIOBase, read_si_model=False):
        self._file = file

        if read_si_model:
            self._read_model()
            return

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

        gif_chunk_size, num_gifs = struct.unpack("<II", self._file.read(8))
        logger.debug(f"{gif_chunk_size=} {num_gifs=}")

        for _ in range(num_gifs):
            self._images.append(self._read_gif())

        model_chunk_size = struct.unpack("<I", self._file.read(4))[0]
        logger.debug(f"{model_chunk_size=}")

        self._global_parts = self._read_part(self._file.tell())

        for offset in parts_offsets:
            self._file.seek(offset, io.SEEK_SET)
            self._parts.extend(self._read_part(offset))

        scanned_offsets = set()
        for offset in models_offsets:
            if offset in scanned_offsets:
                logger.debug(f"Already scanned offset {offset}, skipping")
                continue
            scanned_offsets.add(offset)

            self._file.seek(offset, io.SEEK_SET)
            self._read_model()
