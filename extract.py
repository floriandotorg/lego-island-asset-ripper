from lib.iso import ISO9660
from lib.si import SI

import io
import os.path

from contextlib import contextmanager
import struct


@contextmanager
def write_chunk(type: bytes, f: io.BufferedIOBase):
    f.write(struct.pack("4sI", type, 0))
    start_position = f.tell()
    yield f
    end_position = f.tell()
    size = end_position - start_position
    f.seek(start_position - 4)
    f.write(struct.pack("I", size))
    f.seek(end_position)


def write_si(obj: SI.Object) -> bool:
    match obj.file_type:
        case SI.FileType.WAV:
            with open(f"extract/{filename}_{obj.id}.wav", "wb") as f:
                with obj.open() as obj_data:
                    with write_chunk(b"RIFF", f):
                        f.write(b"WAVE")
                        with write_chunk(b"fmt ", f):
                            f.write(obj_data.read(obj.first_chunk_size))
                        with write_chunk(b"data", f):
                            f.write(obj_data.read())
            return True
    return False


exported_files = 0
with ISO9660("./LEGO_ISLANDI.ISO") as iso:
    for file in iso.filelist():
        filename = os.path.basename(file)
        if filename.endswith(".SI"):
            print(f"Open: {filename}")
            try:
                si = SI(iso.open(file))
            except ValueError:
                print("ERROR")
                continue
            for obj in si.object_list().values():
                if write_si(obj):
                    exported_files += 1
print(f"Exported {exported_files} files")

