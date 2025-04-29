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
from typing import Any, BinaryIO, Union

from lib.flc import FLC
from lib.iso import ISO9660
from lib.si import SI
from lib.smk import SMK
from lib.wdb import WDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def write_gltf2_mesh(mesh: WDB.Mesh, mesh_name: str, texture: WDB.Gif | None, filename: str) -> None:
    _write_gltf2([(mesh, texture)], mesh_name, filename, True)


def write_gltf2_lod(wdb: WDB, lod: WDB.Lod, lod_name: str, filename: str) -> None:
    meshes_textures: list[tuple[WDB.Mesh, (WDB.Gif | None)]] = []
    for mesh in lod.meshes:
        if mesh.uvs:
            texture = wdb.texture_by_name(mesh.texture_name)
        else:
            texture = None
        meshes_textures.append((mesh, texture))
    _write_gltf2(meshes_textures, lod_name, filename, False)


def _write_gltf2(meshes_textures: list[tuple[WDB.Mesh, (WDB.Gif | None)]], name: str, filename: str, mesh_export: bool) -> None:
    def extend_gltf_chunk(type: bytes, content: bytes) -> bytes:
        result = bytearray()
        result.extend(struct.pack("<I4s", len(content), type))
        result.extend(content)
        return bytes(result)

    assert not mesh_export or len(meshes_textures) == 1

    meshes: list[WDB.Mesh]
    textures: list[(WDB.Gif | None)]
    meshes, textures = zip(*meshes_textures)

    USHORT = 5123
    FLOAT = 5126
    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = ARRAY_BUFFER + 1

    bin_chunk_data = bytearray()
    buffer_views = []
    accessors: list[dict[str, Any]] = []

    def append_bin_chunk(data: bytes, target: int | None) -> int:
        buffer_view_offset = len(bin_chunk_data)
        bin_chunk_data.extend(data)
        length = len(bin_chunk_data) - buffer_view_offset
        while len(bin_chunk_data) % 4:
            bin_chunk_data.append(0)
        buffer_view_index = len(buffer_views)
        buffer_view = {"buffer": 0, "byteOffset": buffer_view_offset, "byteLength": length}
        if target is not None:
            buffer_view["target"] = target
        buffer_views.append(buffer_view)
        return buffer_view_index

    def extend_bin_chunk(fmt: str, data: list, target: int | None, componentType: int, type: str) -> int:
        chunk_data = bytearray()
        for entry in data:
            if not isinstance(entry, tuple):
                entry = (entry,)
            chunk_data.extend(struct.pack(fmt, *entry))
        buffer_view_index = append_bin_chunk(chunk_data, target)
        accessors.append({"bufferView": buffer_view_index, "componentType": componentType, "count": len(data), "type": type})
        return buffer_view_index

    json_textures = []
    json_images = []
    json_meshes = []
    json_materials = []
    for mesh_index, mesh in enumerate(meshes):
        vertex_index = extend_bin_chunk("<fff", mesh.vertices, ARRAY_BUFFER, FLOAT, "VEC3")
        min_vertex = [min(vertex[axis] for vertex in mesh.vertices) for axis in range(0, 3)]
        max_vertex = [max(vertex[axis] for vertex in mesh.vertices) for axis in range(0, 3)]
        accessors[-1].update({
            "min": min_vertex,
            "max": max_vertex,
        })
        normal_index = extend_bin_chunk("<fff", mesh.normals, ARRAY_BUFFER, FLOAT, "VEC3")
        index_index = extend_bin_chunk("<H", mesh.indices, ELEMENT_ARRAY_BUFFER, USHORT, "SCALAR")

        json_mesh_data = {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": vertex_index,
                        "NORMAL": normal_index,
                    },
                    "indices": index_index,
                    "material": len(json_materials)
                }
            ],
            "name": name if mesh_export else f"{name}_M{mesh_index}"
        }
        json_material = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [mesh.color.red / 255, mesh.color.green / 255, mesh.color.blue / 255, 1 - mesh.color.alpha]
            }
        }
        json_meshes.append(json_mesh_data)
        json_materials.append(json_material)
        if mesh.uvs:
            uv_index = extend_bin_chunk("<ff", mesh.uvs, ARRAY_BUFFER, FLOAT, "VEC2")
            json_meshes[mesh_index]["primitives"][0]["attributes"]["TEXCOORD_0"] = uv_index
        else:
            assert not mesh.texture_name

    for mesh_index, texture in enumerate(textures):
        if texture:
            with io.BytesIO() as texture_file:
                write_png(texture.width, texture.height, texture.image, ColorSpace.RGB, texture_file)
                texture = texture_file.getvalue()
            texture_index = append_bin_chunk(texture, None)
            json_materials[mesh_index]["pbrMetallicRoughness"] = {
                "baseColorTexture": {
                    "index": len(json_textures)
                }
            }
            json_textures.append({
                "source": len(json_images)
            })
            json_images.append({
                "mimeType": "image/png",
                "bufferView": texture_index
            })

    nodes = []
    if not mesh_export:
        nodes.append({
            "name": name,
            "children": list(range(1, len(json_meshes) + 1)),
        })
    nodes.extend({ "mesh": index } for index in range(len(json_meshes)))

    json_data = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_chunk_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": json_meshes,
        "materials": json_materials,
        "nodes": nodes,
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    if json_images:
        json_data["images"] = json_images
    if json_textures:
        json_data["textures"] = json_textures

    json_chunk_data = bytearray(json.dumps(json_data).encode("utf8"))
    while len(json_chunk_data) % 4:
        json_chunk_data.extend(b" ")

    contents = bytearray()
    contents.extend(extend_gltf_chunk(b"JSON", json_chunk_data))
    contents.extend(extend_gltf_chunk(b"BIN\0", bin_chunk_data))

    with open(filename, "wb") as file:
        file.write(struct.pack("<4sII", b"glTF", 2, 4 * 3 + len(contents)))
        file.write(contents)


def export_wdb_model(wdb: WDB, model: WDB.Model) -> int:
    result = 0
    for lod_index, lod in enumerate(model.lods):
        lod_name = f"{model.name}_L{lod_index}"
        for mesh_index, mesh in enumerate(lod.meshes):
            if mesh.texture_name != "":
                texture = wdb.texture_by_name(mesh.texture_name)
            else:
                texture = None
            assert (texture is not None) == bool(mesh.uvs), f"{texture=} == {len(mesh.uvs)}; {texture is not None=}; {bool(mesh.uvs)=}"
            mesh_name = f"{lod_name}_M{mesh_index}"
            write_gltf2_mesh(mesh, mesh_name, texture, f"extract/{mesh_name}.glb")
            result += 1
        write_gltf2_lod(wdb, lod, lod_name, f"extract/{lod_name}.glb")
        result += 1
    return result


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


def write_si(filename: str, obj: SI.Object) -> bool:
    match obj.file_type:
        case SI.FileType.WAV:

            def extend_wav_chunk(type: bytes, content: bytes) -> bytes:
                result = bytearray()
                result.extend(struct.pack("<4sI", type, len(content)))
                result.extend(content)
                if (len(content) % 2) == 1:
                    result.append(0)
                return bytes(result)

            with open(f"extract/{filename}_{obj.id}.wav", "wb") as file:
                content = bytearray()
                content.extend(b"WAVE")
                content.extend(extend_wav_chunk(b"fmt ", obj.data[: obj.chunk_sizes[0]]))
                content.extend(extend_wav_chunk(b"data", obj.data[obj.chunk_sizes[0] :]))
                file.write(extend_wav_chunk(b"RIFF", content))
            return True
        case SI.FileType.STL:
            write_bitmap(f"extract/{filename}_{obj.id}.bmp", obj)
        case SI.FileType.FLC:
            mem_file = io.BytesIO()
            write_flc(mem_file, obj)
            mem_file.seek(0)
            with open(f"extract/{filename}_{obj.id}.flc", "wb") as file:
                file.write(mem_file.getvalue())
            mem_file.seek(0)
            try:
                flc = FLC(mem_file)
                write_flc_sprite_sheet(flc, f"extract/{filename}_{obj.id}_frames{len(flc.frames)}_fps{flc.fps}.bmp")
                write_smk_avi(flc, f"extract/{filename}_{obj.id}.avi")
            except Exception as e:
                logger.error(f"Error writing {filename}_{obj.id}.flc: {e}")
                return False
            return True
        case SI.FileType.SMK:
            with open(f"extract/{filename}_{obj.id}.smk", "wb") as file:
                file.write(obj.data)
            smk = SMK(io.BytesIO(obj.data))
            write_smk_avi(smk, f"extract/{filename}_{obj.id}.avi")
            return True
    return False


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
    result = sum(1 if write_si(os.path.basename(file.name), obj) else 0 for obj in file.si.object_list.values())
    logger.info(f"Extracting {file.name} .. [done]")
    return result


def process_files(files: list[File]) -> int:
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
        for wdb_file in wdb_files:
            wdb = WDB(wdb_file)
            for model in wdb.models:
                exported_files += export_wdb_model(wdb, model)
        logger.info("Exporting WDB models .. [done]")

        exported_files += sum(results.get())

    logger.info(f"Exported {exported_files} files")
