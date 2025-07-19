import io
import json
import logging
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing import Pool
from tkinter import filedialog
from typing import Any, BinaryIO, Callable, Union

from lib.animation import AnimationNode
from lib.flc import FLC
from lib.iso import ISO9660
from lib.si import SI
from lib.smk import SMK
from lib.wdb import WDB

logger = logging.getLogger(__name__)
log_level = logging.INFO


class ColorSpace(IntEnum):
    RGB = 2
    RGBA = 6


def write_png(width: int, height: int, data: bytes, color: ColorSpace, stream: BinaryIO):
    def write_chunk(tag, data):
        stream.write(struct.pack(">I", len(data)))
        stream.write(tag)
        stream.write(data)
        stream.write(struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    # PNG file signature
    stream.write(b"\x89PNG\r\n\x1a\n")

    # IHDR chunk
    ihdr = struct.pack(">IIBBBBB", width, height, 8, int(color), 0, 0, 0)
    write_chunk(b"IHDR", ihdr)

    # Prepare raw image data (add filter byte 0 at start of each row)
    match color:
        case ColorSpace.RGB:
            byte_per_pixel = 3
        case ColorSpace.RGBA:
            byte_per_pixel = 4
        case _:
            raise ValueError(f"Invalid value for parameter color: {color}")

    if len(data) != width * height * byte_per_pixel:
        raise ValueError(f"Expected {width * height * byte_per_pixel} bytes but got {len(data)}")

    raw = b""
    stride = width * byte_per_pixel
    for y in range(height):
        raw += b"\x00" + data[y * stride : (y + 1) * stride]

    # IDAT chunk (compressed image data)
    compressed = zlib.compress(raw)
    write_chunk(b"IDAT", compressed)

    # IEND chunk
    write_chunk(b"IEND", b"")


class GLBWriter:
    USHORT = 5123
    FLOAT = 5126
    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = ARRAY_BUFFER + 1

    class Node:
        def __init__(self) -> None:
            self._data: dict[str, Any] = {}

        @property
        def data(self) -> dict[str, Any]:
            return self._data

    class Parent(Node):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._data["name"] = name
            self._children: list[GLBWriter.Node] = []

        @property
        def children(self) -> list["GLBWriter.Node"]:
            return self._children

        def add_child(self, node: "GLBWriter.Node"):
            self._children.append(node)

        def __bool__(self) -> bool:
            return bool(self._children)

    def __init__(self) -> None:
        self._bin_chunk_data = bytearray()
        self._buffer_views: list[dict] = []
        self._accessors: list[dict[str, Any]] = []

        self._json_textures: list[dict] = []
        self._json_images: list[dict] = []
        self._json_meshes: list[dict] = []
        self._json_materials: list[dict] = []

        self._textures: list[tuple[int, WDB.Gif]] = []

        self._root_node: GLBWriter.Node | None = None

    @property
    def root_node(self) -> "GLBWriter.Node | None":
        return self._root_node

    @root_node.setter
    def root_node(self, node: "GLBWriter.Node") -> None:
        self._root_node = node

    @staticmethod
    def _extend_gltf_chunk(type: bytes, content: bytes) -> bytes:
        result = bytearray()
        result.extend(struct.pack("<I4s", len(content), type))
        result.extend(content)
        return bytes(result)

    def _append_buffer_view(self, data: bytes, target: int | None) -> int:
        buffer_view_offset = len(self._bin_chunk_data)
        self._bin_chunk_data.extend(data)
        length = len(self._bin_chunk_data) - buffer_view_offset
        while len(self._bin_chunk_data) % 4:
            self._bin_chunk_data.append(0)
        buffer_view_index = len(self._buffer_views)
        buffer_view = {"buffer": 0, "byteOffset": buffer_view_offset, "byteLength": length}
        if target is not None:
            buffer_view["target"] = target
        self._buffer_views.append(buffer_view)
        return buffer_view_index

    def _append_accessor(self, fmt: str, data: list, target: int | None, componentType: int, type: str) -> int:
        chunk_data = bytearray()
        for entry in data:
            if not isinstance(entry, tuple):
                entry = (entry,)
            chunk_data.extend(struct.pack(fmt, *entry))
        buffer_view_index = self._append_buffer_view(chunk_data, target)
        accessor_index = len(self._accessors)
        self._accessors.append({"bufferView": buffer_view_index, "componentType": componentType, "count": len(data), "type": type})
        return accessor_index

    def add_mesh(self, mesh: WDB.Mesh, texture: (WDB.Gif | None), name: str) -> "GLBWriter.Node":
        mesh_index = len(self._json_meshes)

        vertex_index = self._append_accessor("<fff", mesh.vertices, GLBWriter.ARRAY_BUFFER, GLBWriter.FLOAT, "VEC3")
        min_vertex = [min(vertex[axis] for vertex in mesh.vertices) for axis in range(0, 3)]
        max_vertex = [max(vertex[axis] for vertex in mesh.vertices) for axis in range(0, 3)]
        self._accessors[-1].update(
            {
                "min": min_vertex,
                "max": max_vertex,
            }
        )
        normal_index = self._append_accessor("<fff", mesh.normals, GLBWriter.ARRAY_BUFFER, GLBWriter.FLOAT, "VEC3")
        index_index = self._append_accessor("<H", mesh.indices, GLBWriter.ELEMENT_ARRAY_BUFFER, GLBWriter.USHORT, "SCALAR")

        json_mesh_data: dict[str, Any] = {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": vertex_index,
                        "NORMAL": normal_index,
                    },
                    "indices": index_index,
                    "material": len(self._json_materials),
                }
            ],
            "name": name,
        }
        json_material = {"pbrMetallicRoughness": {"baseColorFactor": [mesh.color.red / 255, mesh.color.green / 255, mesh.color.blue / 255, 1 - mesh.color.alpha]}}
        self._json_meshes.append(json_mesh_data)
        self._json_materials.append(json_material)
        if mesh.uvs:
            uv_index = self._append_accessor("<ff", mesh.uvs, GLBWriter.ARRAY_BUFFER, GLBWriter.FLOAT, "VEC2")
            json_mesh_data["primitives"][0]["attributes"]["TEXCOORD_0"] = uv_index
        else:
            assert not mesh.texture_name

        if texture:
            with io.BytesIO() as texture_file:
                write_png(texture.width, texture.height, texture.image, ColorSpace.RGB, texture_file)
                texture_data = texture_file.getvalue()
            texture_index = self._append_buffer_view(texture_data, None)
            self._json_materials[mesh_index]["pbrMetallicRoughness"] = {"baseColorTexture": {"index": len(self._json_textures)}}
            self._json_textures.append({"source": len(self._json_images)})
            self._json_images.append({"mimeType": "image/png", "bufferView": texture_index})

        mesh_node = GLBWriter.Node()
        mesh_node.data["mesh"] = mesh_index
        return mesh_node

    def build(self) -> bytearray:
        """Builds the glb file and returns the contents."""
        if self._root_node is None:
            raise Exception("No node defined")

        nodes: list[dict[str, Any]] = []

        def append_node(node: GLBWriter.Node):
            node_data = dict(node.data)
            nodes.append(node_data)
            if isinstance(node, GLBWriter.Parent):
                children_indices: list[int] = []
                node_data["children"] = children_indices
                for child in node.children:
                    children_indices.append(len(nodes))
                    append_node(child)

        append_node(self._root_node)

        json_data = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(self._bin_chunk_data)}],
            "bufferViews": self._buffer_views,
            "accessors": self._accessors,
            "meshes": self._json_meshes,
            "materials": self._json_materials,
            "nodes": nodes,
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        }

        if self._json_images:
            json_data["images"] = self._json_images
        if self._json_textures:
            json_data["textures"] = self._json_textures

        json_chunk_data = bytearray(json.dumps(json_data).encode("utf8"))
        while len(json_chunk_data) % 4:
            json_chunk_data.extend(b" ")

        contents = bytearray()
        contents.extend(GLBWriter._extend_gltf_chunk(b"JSON", json_chunk_data))
        contents.extend(GLBWriter._extend_gltf_chunk(b"BIN\0", self._bin_chunk_data))
        return contents

    def write(self, filename: str) -> None:
        """Writes the glb file. Important: Adding meshes after calling this is not supported."""
        contents = self.build()
        with open(filename, "wb") as file:
            file.write(struct.pack("<4sII", b"glTF", 2, 4 * 3 + len(contents)))
            file.write(contents)


def write_gltf2_mesh(mesh: WDB.Mesh, texture: (WDB.Gif | None), name: str, filename: str) -> None:
    writer = GLBWriter()
    writer.root_node = writer.add_mesh(mesh, texture, name)
    writer.write(filename)


def _add_lod(lod: WDB.Lod, lod_name: str, writer: GLBWriter, texture_by_name: Callable[[str], WDB.Gif]) -> GLBWriter.Parent | None:
    if not lod.meshes:
        return None
    lod_node = GLBWriter.Parent(lod_name)
    for mesh_index, mesh in enumerate(lod.meshes):
        if mesh.uvs:
            texture = texture_by_name(mesh.texture_name)
        else:
            texture = None
        mesh_node = writer.add_mesh(mesh, texture, f"{lod_name}_M{mesh_index}")
        lod_node.add_child(mesh_node)
    return lod_node


def write_gltf2_lod(lod: WDB.Lod, lod_name: str, filename: str, texture_by_name: Callable[[str], WDB.Gif]) -> None:
    writer = GLBWriter()
    root = _add_lod(lod, lod_name, writer, texture_by_name)
    if root:
        writer.root_node = root
        writer.write(filename)


def write_gltf2_model(wdb: WDB, model: WDB.Model, filename: str, all_lods: bool) -> None:
    writer = GLBWriter()

    def add_lod(lods: list[WDB.Lod], name: str, parent: GLBWriter.Parent, is_model: bool) -> None:
        start = 0 if all_lods else len(lods) - 1
        for lod_index, lod in enumerate(lods[start:], start):
            lod_name = f"{name}_L{lod_index}"
            lod_node = _add_lod(lod, lod_name, writer, wdb.model_texture_by_name if is_model else wdb.part_texture_by_name)
            if lod_node:
                parent.add_child(lod_node)

    def add_roi(roi: WDB.Roi, animation: (AnimationNode | None)) -> GLBWriter.Parent | None:
        roi_node = GLBWriter.Parent(roi.name)

        transformation: dict[str, Any] = {}
        if animation:
            if animation.translation_keys:
                if len(animation.translation_keys) > 1:
                    logger.warning(f"Found {len(animation.translation_keys)} translations for {roi.name}")
                if animation.translation_keys[0].time != 0:
                    logger.warning(f"First translation key for {roi.name} is not at time 0")
                else:
                    # TODO: What to do with flags
                    transformation["translation"] = animation.translation_keys[0].vertex
            if animation.rotation_keys:
                if len(animation.rotation_keys) > 1:
                    logger.warning(f"Found {len(animation.rotation_keys)} rotations for {roi.name}")
                if animation.rotation_keys[0].time != 0:
                    logger.warning(f"First rotation key for {roi.name} is not at time 0")
                else:
                    # TODO: What to do with flags and time
                    transformation["rotation"] = animation.rotation_keys[0].quaternion
        roi_node.data.update(transformation)

        add_lod(roi.lods, roi.name, roi_node, True)

        if roi.reference:
            logger.info(f"{roi.name} (in {model.roi.name}) references another roi: '{roi.reference}'")
            for part in wdb.parts:
                if part.name.lower() == roi.reference.lower():
                    add_lod(part.lods, part.name, roi_node, False)
                    break
            else:
                logger.warning(f"Cannot find '{roi.reference}' (in {model.roi.name}) referenced from {roi.name} in parts list")

        for child in roi.children:
            if animation:
                child_animation = [x for x in animation.children if x.name.lower() == child.name.lower()]
                if len(child_animation) > 1:
                    logger.warning(f"Found {len(child_animation)} animations for {child.name}, using first")
            else:
                child_animation = []
            child_node = add_roi(child, child_animation[0] if child_animation else None)
            if child_node:
                roi_node.add_child(child_node)

        if roi_node:
            return roi_node
        return None

    root_node = add_roi(model.roi, model.animation.tree)
    if root_node:
        writer.root_node = root_node

    writer.write(filename)


def _export_wdb_roi(wdb: WDB, roi: WDB.Roi, root_name: str, prefix: str, path_prefix="extract/WORLD.WDB/models/") -> int:
    prefix = f"{prefix}{roi.name}"
    result = 0
    for lod_index, lod in enumerate(roi.lods):
        lod_name = f"{prefix}_L{lod_index}"
        for mesh_index, mesh in enumerate(lod.meshes):
            if mesh.texture_name != "":
                texture = wdb.model_texture_by_name(mesh.texture_name)
            else:
                texture = None
            assert (texture is not None) == bool(mesh.uvs), f"{texture=} == {len(mesh.uvs)}; {texture is not None=}; {bool(mesh.uvs)=}"
            mesh_name = f"{lod_name}_M{mesh_index}"
            write_gltf2_mesh(mesh, texture, mesh_name, f"{path_prefix}/{root_name}/parts/{mesh_name}.glb")
            result += 1
        write_gltf2_lod(lod, lod_name, f"{path_prefix}/{root_name}/parts/{lod_name}.glb", wdb.model_texture_by_name)
        result += 1
    for child in roi.children:
        _export_wdb_roi(wdb, child, root_name, f"{prefix}_R", path_prefix)
    return result


def export_wdb_model(wdb: WDB, model: WDB.Model, path_prefix="extract/WORLD.WDB/models/") -> int:
    def count_meshes(roi: WDB.Roi) -> int:
        return sum(len(lod.meshes) for lod in roi.lods) + sum(count_meshes(child) for child in roi.children)

    if count_meshes(model.roi) < 1:
        logger.warning(f"Model {model.roi.name} has no meshes")
        return 0

    file_count = 0
    os.makedirs(f"{path_prefix}/{model.roi.name}/parts", exist_ok=True)
    file_count += _export_wdb_roi(wdb, model.roi, model.roi.name, "", path_prefix)
    write_gltf2_model(wdb, model, f"{path_prefix}/{model.roi.name}/model.glb", False)
    file_count += 1
    write_gltf2_model(wdb, model, f"{path_prefix}/{model.roi.name}/all_lods.glb", True)
    file_count += 1
    return file_count


def export_wdb_part(wdb: WDB, part: WDB.Part, path="parts") -> int:
    file_count = 0
    os.makedirs(f"extract/WORLD.WDB/{path}/{part.name}", exist_ok=True)
    for lod_index, lod in enumerate(part.lods):
        write_gltf2_lod(lod, f"{part.name}_L{lod_index}", f"extract/WORLD.WDB/{path}/{part.name}/L{lod_index}.glb", wdb.part_texture_by_name)
        file_count += 1
    return file_count


def write_bitmap(filename: str, obj: SI.Object) -> None:
    with open(filename, "wb") as file:
        file.write(struct.pack("<2sIII", b"BM", len(obj.data), 0, obj.chunk_sizes[0] + 14))
        file.write(obj.data)


def write_flc(dest_file: io.BufferedIOBase, obj: SI.Object) -> None:
    src_file = obj.open()
    for n, chunk_size in enumerate(obj.chunk_sizes):
        chunk = src_file.read(chunk_size)
        if n == 0:
            dest_file.write(chunk)
            continue
        if chunk_size == 20:
            dest_file.write(b"\x10\x00\x00\x00\xfa\xf1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
            continue
        dest_file.write(chunk[20:])


def write_gif(gif: WDB.Gif, filename: str) -> None:
    with open(filename, "wb") as file:
        width = gif.width
        height = gif.height
        pad = b"\x00" * ((4 - (width * 3) % 4) % 4)
        header_size = 54
        bf_size = header_size + (width * 3 + len(pad)) * height
        bi_size = bf_size - header_size

        # BMP Header (14 bytes)
        file.write(struct.pack("<2sIHHI", b"BM", bf_size, 0, 0, header_size))

        # DIB Header (40 bytes)
        file.write(
            struct.pack(
                "<IIiHHIIIIII",
                40,  # DIB header size
                width,  # Width
                -height,  # Height
                1,  # Color planes
                24,  # Bits per pixel (RGB = 24)
                0,  # No compression
                bi_size,  # Image size
                0,  # Horizontal resolution (pixels/meter)
                0,  # Vertical resolution (pixels/meter)
                0,  # Number of colors in palette
                0,  # Important colors
            )
        )

        bgr_frame = bytearray(len(gif.image))
        bf = memoryview(bgr_frame)
        rf = memoryview(gif.image)
        # Swap R and B:
        bf[0::3] = rf[2::3]  # B
        bf[1::3] = rf[1::3]  # G
        bf[2::3] = rf[0::3]  # R

        if pad:
            row_size = width * 3
            file.write(b"".join(bgr_frame[i : i + row_size] + pad for i in range(0, len(bgr_frame), row_size)))
        else:
            file.write(bgr_frame)


def write_flc_sprite_sheet(flc: FLC, filename: str) -> None:
    with open(filename, "wb") as file:
        width = flc.width
        height = flc.height * len(flc.frames)
        pad = b"\x00" * ((4 - (width * 3) % 4) % 4)
        header_size = 54
        bf_size = header_size + (width * 3 + len(pad)) * height
        bi_size = bf_size - header_size

        # BMP Header (14 bytes)
        file.write(struct.pack("<2sIHHI", b"BM", bf_size, 0, 0, header_size))

        # DIB Header (40 bytes)
        file.write(
            struct.pack(
                "<IIiHHIIIIII",
                40,  # DIB header size
                width,  # Width
                -height,  # Height
                1,  # Color planes
                24,  # Bits per pixel (RGB = 24)
                0,  # No compression
                bi_size,  # Image size
                0,  # Horizontal resolution (pixels/meter)
                0,  # Vertical resolution (pixels/meter)
                0,  # Number of colors in palette
                0,  # Important colors
            )
        )

        for frame in flc.frames:
            bgr_frame = bytearray(len(frame))
            bf = memoryview(bgr_frame)
            rf = memoryview(frame)
            # Swap R and B:
            bf[0::3] = rf[2::3]  # B
            bf[1::3] = rf[1::3]  # G
            bf[2::3] = rf[0::3]  # R

            if pad:
                row_size = width * 3
                file.write(b"".join(bgr_frame[i : i + row_size] + pad for i in range(0, len(bgr_frame), row_size)))
            else:
                file.write(bgr_frame)


def write_smk_avi(video: Union[SMK, FLC], filename: str) -> None:
    with open(filename, "wb") as file:
        pad = b"\x00" * ((4 - (video.width * 3) % 4) % 4)
        total_frame_size = (video.width + len(pad)) * video.height * 3

        file.write(
            struct.pack(
                "<4sI4s4sI4s4sIIIIIIIIIII16x4sI4s4sI4s4sIIIIIIIIIIII4sIIIiHHIIIIII",
                b"RIFF",  # RIFF signature
                0,  # File size (filled later)
                b"AVI ",  # AVI signature
                b"LIST",
                4 + 64 + 124,  # Size of LIST chunk
                b"hdrl",
                b"avih",
                56,  # Size of avih chunk
                1_000_000 // int(video.fps),  # Microseconds per frame
                total_frame_size,  # Max bytes per second
                1,  # Padding granularity
                0,  # Flags
                len(video.frames),  # Total frames
                0,  # Initial frames
                1,  # Number of streams
                total_frame_size,  # Suggested buffer size
                video.width,  # Width
                video.height,  # Height
                b"LIST",
                116,
                b"strl",
                b"strh",
                56,
                b"vids",  # Video stream type
                b"DIB ",  # Video codec (uncompressed)
                0,  # Flags
                0,  # Priority + Language
                0,  # Initial frames
                1,  # Scale
                int(video.fps),  # Rate
                0,  # Start
                len(video.frames),  # Length
                total_frame_size,  # Suggested buffer size
                0,  # Quality
                total_frame_size,  # Sample size
                0,  # rcFrame
                0,  # rcFrame: right, bottom
                b"strf",
                40,
                40,
                video.width,  # Width
                -video.height,  # Height (negative for top-down)
                1,  # Color planes
                24,  # Bits per pixel (RGB = 24)
                0,  # No compression
                total_frame_size,  # Image size
                0,  # Horizontal resolution (pixels/meter)
                0,  # Vertical resolution (pixels/meter)
                0,  # Number of colors in palette
                0,  # Important colors
            )
        )

        file.write(
            struct.pack(
                "<4sI4s",
                b"LIST",
                len(video.frames) * (total_frame_size + 8) + 4,
                b"movi",
            )
        )

        for frame in video.frames:
            file.write(
                struct.pack(
                    "<4sI",
                    b"00db",
                    total_frame_size,
                )
            )

            bgr_frame = bytearray(len(frame))
            bf = memoryview(bgr_frame)
            rf = memoryview(frame)
            # Swap R and B:
            bf[0::3] = rf[2::3]  # B
            bf[1::3] = rf[1::3]  # G
            bf[2::3] = rf[0::3]  # R

            if pad:
                row_size = video.width * 3
                file.write(b"".join(bgr_frame[i : i + row_size] + pad for i in range(0, len(bgr_frame), row_size)))
            else:
                file.write(bgr_frame)

        file_size = file.tell()
        file.seek(4, io.SEEK_SET)
        file.write(struct.pack("<I", file_size - 8))


def write_si(filename: str, obj: SI.Object) -> int:
    os.makedirs(f"extract/{filename}", exist_ok=True)

    match obj.file_type:
        case SI.FileType.OBJ:
            if obj.presenter != "LegoModelPresenter" or not obj.data:
                return 0

            model_files = 0
            wdb = WDB(io.BytesIO(obj.data), read_si_model=True)
            for model in wdb.models:
                model_files += export_wdb_model(wdb, model, f"extract/{filename}/models")
            os.makedirs(f"extract/{filename}/models/textures", exist_ok=True)
            for texture in wdb.model_textures:
                write_gif(texture, f"extract/{filename}/models/textures/{obj.id}.bmp")
            return model_files + len(wdb.model_textures)

        case SI.FileType.WAV:

            def extend_wav_chunk(type: bytes, content: bytes) -> bytes:
                result = bytearray()
                result.extend(struct.pack("<4sI", type, len(content)))
                result.extend(content)
                if (len(content) % 2) == 1:
                    result.append(0)
                return bytes(result)

            with open(f"extract/{filename}/{obj.id}.wav", "wb") as file:
                content = bytearray()
                content.extend(b"WAVE")
                content.extend(extend_wav_chunk(b"fmt ", obj.data[: obj.chunk_sizes[0]]))
                content.extend(extend_wav_chunk(b"data", obj.data[obj.chunk_sizes[0] :]))
                file.write(extend_wav_chunk(b"RIFF", content))
            return 1
        case SI.FileType.STL:
            write_bitmap(f"extract/{filename}/{obj.id}.bmp", obj)
            return 1
        case SI.FileType.FLC:
            mem_file = io.BytesIO()
            write_flc(mem_file, obj)
            mem_file.seek(0)
            with open(f"extract/{filename}/{obj.id}.flc", "wb") as file:
                file.write(mem_file.getvalue())
            mem_file.seek(0)
            try:
                flc = FLC(mem_file)
                write_flc_sprite_sheet(flc, f"extract/{filename}/{obj.id}_frames{len(flc.frames)}_fps{flc.fps}.bmp")
                write_smk_avi(flc, f"extract/{filename}/{obj.id}.avi")
            except Exception as e:
                logger.error(f"Error writing {filename}_{obj.id}.flc: {e}")
                return 1
            return 3
        case SI.FileType.SMK:
            with open(f"extract/{filename}/{obj.id}.smk", "wb") as file:
                file.write(obj.data)
            smk = SMK(io.BytesIO(obj.data))
            write_smk_avi(smk, f"extract/{filename}/{obj.id}.avi")
            return 2
        case _:
            return 0


def get_iso_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]

    path = filedialog.askopenfilename(
        title="Select ISO file",
        filetypes=[("ISO files", "*.iso"), ("All files", "*.*")],
    )
    if not path:
        sys.exit("No file selected")
    return path


@dataclass
class File:
    si: SI
    name: str
    weight: int

    def _obj_weight(self, obj: SI.Object) -> int:
        match obj.file_type:
            case SI.FileType.FLC:
                frames, width, height = struct.unpack("<6xHHH", obj.data[0:12])
                return (width * height * frames) / 10_000
            case SI.FileType.SMK:
                width, height, frames = struct.unpack("<4xIII", obj.data[0:16])
                return (width * height * frames) / 2_000
            case _:
                return 10

    def __init__(self, si: SI, name: str):
        self.si = si
        self.name = name
        self.weight = sum(self._obj_weight(obj) for obj in self.si.object_list.values())

    def __hash__(self) -> int:
        return hash(self.name)


def process_file(file: File) -> int:
    logger.info(f"Extracting {file.name} ..")
    result = sum(write_si(os.path.basename(file.name), obj) for obj in file.si.object_list.values())
    logger.info(f"Extracting {file.name} .. [done]")
    return result


def process_files(files: list[File]) -> int:
    logging.basicConfig(level=log_level)
    return sum(process_file(file) for file in files)


def balanced_chunks(data: list[File], n: int) -> list[list[File]]:
    data = sorted(data, key=lambda x: x.weight, reverse=True)
    chunks: list[list[File]] = [[] for _ in range(n)]
    sums = [0] * n
    for item in data:
        i = sums.index(min(sums))
        chunks[i].append(item)
        sums[i] += item.weight
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=log_level)
    os.makedirs("extract", exist_ok=True)

    si_files: list[File] = []
    wdb_files: list[io.BytesIO] = []
    with ISO9660(get_iso_path()) as iso:
        for file in iso.filelist:
            if not file.endswith(".SI") and not file.endswith(".WDB"):
                continue

            try:
                mem_file = io.BytesIO()
                mem_file.write(iso.open(file).read())
                mem_file.seek(0, io.SEEK_SET)
                if file.endswith(".SI"):
                    si_files.append(File(SI(mem_file), file))
                elif file.endswith(".WDB"):
                    wdb_files.append(mem_file)
                else:
                    raise ValueError(f"Unknown file type: {file}")
            except ValueError:
                logger.error(f"Error opening {file}")

    cpus = os.cpu_count()
    if cpus is None:
        cpus = 1

    exported_files = 0
    with Pool(processes=cpus) as pool:
        results = pool.map_async(process_files, balanced_chunks(si_files, cpus))

        logger.info("Exporting WDB models ..")
        os.makedirs("extract/WORLD.WDB", exist_ok=True)
        os.makedirs("extract/WORLD.WDB/images", exist_ok=True)
        os.makedirs("extract/WORLD.WDB/part_textures", exist_ok=True)
        os.makedirs("extract/WORLD.WDB/model_textures", exist_ok=True)
        os.makedirs("extract/WORLD.WDB/parts", exist_ok=True)
        os.makedirs("extract/WORLD.WDB/global_parts", exist_ok=True)
        for wdb_file in wdb_files:
            wdb = WDB(wdb_file)
            for model in wdb.models:
                exported_files += export_wdb_model(wdb, model)
            for part in wdb.parts:
                exported_files += export_wdb_part(wdb, part)
            for part in wdb.global_parts:
                exported_files += export_wdb_part(wdb, part, "global_parts")
            for image in wdb.images:
                write_gif(image, f"extract/WORLD.WDB/images/{image.title}.bmp")
            for texture in wdb.part_textures:
                write_gif(texture, f"extract/WORLD.WDB/part_textures/{texture.title}.bmp")
            for model_texture in wdb.model_textures:
                write_gif(model_texture, f"extract/WORLD.WDB/model_textures/{model_texture.title}.bmp")
            exported_files += len(wdb.images) + len(wdb.part_textures) + len(wdb.model_textures)
        logger.info("Exporting WDB models .. [done]")

        exported_files += sum(results.get())

    logger.info(f"Exported {exported_files} files")
