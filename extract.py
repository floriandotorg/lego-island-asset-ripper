from enum import IntEnum
import io
import logging
import os
import struct
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from tkinter import filedialog
from typing import BinaryIO, Union
import zlib

from lib.flc import FLC
from lib.iso import ISO9660
from lib.si import SI
from lib.smk import SMK

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ColorSpace(IntEnum):
    RGB = 2
    RGBA = 6


def write_png(width: int, height: int, data: bytes, color: ColorSpace, stream: BinaryIO):
    def write_chunk(tag, data):
        stream.write(struct.pack('>I', len(data)))
        stream.write(tag)
        stream.write(data)
        stream.write(struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

    # PNG file signature
    stream.write(b'\x89PNG\r\n\x1a\n')

    # IHDR chunk
    ihdr = struct.pack('>IIBBBBB',
        width, height, 8, int(color), 0, 0, 0
    )
    write_chunk(b'IHDR', ihdr)

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

    raw = b''
    stride = width * byte_per_pixel
    for y in range(height):
        raw += b'\x00' + data[y * stride:(y+1) * stride]

    # IDAT chunk (compressed image data)
    compressed = zlib.compress(raw)
    write_chunk(b'IDAT', compressed)

    # IEND chunk
    write_chunk(b'IEND', b'')


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
    files: list[File] = []
    with ISO9660(get_iso_path()) as iso:
        for file in iso.filelist:
            if not file.endswith(".SI"):
                continue

            try:
                mem_file = io.BytesIO()
                mem_file.write(iso.open(file).read())
                mem_file.seek(0, io.SEEK_SET)
                files.append(File(SI(mem_file), file))
            except ValueError:
                logger.error(f"Error opening {file}")

    cpus = os.cpu_count()
    if cpus is None:
        cpus = 1

    with Pool(processes=cpus) as pool:
        exported_files = sum(pool.map(process_files, balanced_chunks(files, cpus)))

    logger.info(f"Exported {exported_files} files")
