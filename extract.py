import io
import logging
import os
import random
import struct
import sys
from multiprocessing import Pool
from tkinter import filedialog

from lib.flc import FLC
from lib.iso import ISO9660
from lib.si import SI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extend_chunk(type: bytes, content: bytes) -> bytes:
    result = bytearray()
    result.extend(struct.pack("<4sI", type, len(content)))
    result.extend(content)
    if (len(content) % 2) == 1:
        result.append(0)
    return bytes(result)


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


def write_si(filename: str, obj: SI.Object) -> bool:
    match obj.file_type:
        case SI.FileType.WAV:
            with open(f"extract/{filename}_{obj.id}.wav", "wb") as file:
                content = bytearray()
                content.extend(b"WAVE")
                content.extend(extend_chunk(b"fmt ", obj.data[: obj.chunk_sizes[0]]))
                content.extend(extend_chunk(b"data", obj.data[obj.chunk_sizes[0] :]))
                file.write(extend_chunk(b"RIFF", content))
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
            except Exception as e:
                logger.error(f"Error writing {filename}_{obj.id}.flc: {e}")
                return False
            return True
        case SI.FileType.SMK:
            with open(f"extract/{filename}_{obj.id}.smk", "wb") as file:
                file.write(obj.data)
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


def process_file(file: str) -> int:
    exported_files = 0
    with ISO9660(get_iso_path()) as iso:
        logger.info(f"Open: {file}")
        try:
            si = SI(iso.open(file))
        except ValueError:
            logger.error(f"Error opening {file}")
            return 0
        for obj in si.object_list.values():
            if write_si(os.path.basename(file), obj):
                exported_files += 1
    return exported_files


def process_files(files: list[str]) -> int:
    return sum(process_file(file) for file in files)


if __name__ == "__main__":
    with ISO9660(get_iso_path()) as iso:
        files = [file for file in iso.filelist if file.endswith(".SI")]
        random.shuffle(files)

    cpus = os.cpu_count()
    if cpus is None:
        cpus = 1

    with Pool(processes=cpus) as pool:
        chunks = [files[i : i + cpus] for i in range(0, len(files), cpus)]
        exported_files = sum(pool.map(process_files, chunks))

    logger.info(f"Exported {exported_files} files")
