import io
import logging
import os.path
import struct
import sys
from tkinter import filedialog

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


def write_si(obj: SI.Object) -> bool:
    match obj.file_type:
        case SI.FileType.WAV:
            with open(f"extract/{filename}_{obj.id}.wav", "wb") as file:
                content = bytearray()
                content.extend(b"WAVE")
                content.extend(extend_chunk(b"fmt ", obj.data[:obj.first_chunk_size]))
                content.extend(extend_chunk(b"data", obj.data[obj.first_chunk_size:]))
                file.write(extend_chunk(b"RIFF", content))
            return True
        case SI.FileType.STL:
            write_bitmap(f"extract/{filename}_{obj.id}.bmp", obj)
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


def write_bitmap(filename: str, obj: SI.Object) -> None:
    with open(filename, "wb") as file:
        file.write(
            struct.pack("<2sIII", b"BM", len(obj.data), 0, obj.first_chunk_size + 14)
        )
        file.write(obj.data)


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
                if write_si(obj):
                    exported_files += 1
logger.info(f"Exported {exported_files} files")
