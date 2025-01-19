import io
import logging
import os.path
import struct
import sys
from contextlib import contextmanager
from tkinter import filedialog

from lib.iso import ISO9660
from lib.si import SI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@contextmanager
def write_chunk(type: bytes, file: io.BufferedIOBase):
    file.write(struct.pack("<4sI", type, 0))
    start_position = file.tell()
    yield file
    end_position = file.tell()
    size = end_position - start_position
    file.seek(start_position - 4)
    file.write(struct.pack("<I", size))
    file.seek(end_position)


def write_si(obj: SI.Object) -> bool:
    match obj.file_type:
        case SI.FileType.WAV:
            with open(f"extract/{filename}_{obj.id}.wav", "wb") as file:
                with obj.open() as obj_data:
                    with write_chunk(b"RIFF", file):
                        file.write(b"WAVE")
                        with write_chunk(b"fmt ", file):
                            file.write(obj_data.read(obj.first_chunk_size))
                        with write_chunk(b"data", file):
                            file.write(obj_data.read())
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
