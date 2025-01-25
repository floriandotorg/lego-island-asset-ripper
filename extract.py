import io
import logging
import os.path
import struct
import sys
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
        width = flc.width()  # Assuming width from STL format
        height = flc.height() * len(flc.frames())  # Calculate height based on RGB data
        header_size = 54  # Standard BMP header size
        image_size = width * height * 3  # 3 bytes per pixel (RGB)
        file_size = header_size + image_size

        # BMP Header (14 bytes)
        file.write(struct.pack("<2sIHHI", b"BM", file_size, 0, 0, header_size))

        # DIB Header (40 bytes)
        file.write(
            struct.pack(
                "<IIIHHIIIIII",
                40,  # DIB header size
                width,  # Width
                height,  # Height
                1,  # Color planes
                24,  # Bits per pixel (RGB = 24)
                0,  # No compression
                image_size,  # Image size
                0,  # Horizontal resolution (pixels/meter)
                0,  # Vertical resolution (pixels/meter)
                0,  # Number of colors in palette
                0,  # Important colors
            )
        )

        pad = b"\x00" * (4 - (width * 3) % 4)

        for frame in flc.frames():
            bgr_frame = bytearray(len(frame))
            bf = memoryview(bgr_frame)
            rf = memoryview(frame)
            bf[0::3] = rf[2::3]
            bf[1::3] = rf[1::3]
            bf[2::3] = rf[0::3]

            if pad:
                for n in range(height):
                    file.write(bgr_frame[n * width * 3 : (n + 1) * width * 3])
                    file.write(pad)
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
                write_flc_sprite_sheet(flc, f"extract/{filename}_{obj.id}_frames{len(flc.frames())}_fps{flc.fps()}.bmp")
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


exported_files = 0
with ISO9660(get_iso_path()) as iso:
    for file in iso.filelist():
        filename = os.path.basename(file)
        if filename.endswith(".SI"):
            logger.info(f"Open: {filename}")
            try:
                si = SI(iso.open(file))
            except ValueError:
                logger.error(f"Error opening {filename}")
                continue
            for obj in si.object_list().values():
                if write_si(filename, obj):
                    exported_files += 1
logger.info(f"Exported {exported_files} files")
