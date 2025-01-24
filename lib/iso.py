import io
import mmap
import struct
from enum import Enum

SECTOR_SIZE = 2048


class ISO9660:
    class File(io.BufferedIOBase):
        def __init__(self, mm, loc, len):
            self.mm = mm
            self.loc = loc
            self.len = len
            self.pos = 0

        def read(self, n: int | None = None, /) -> bytes:
            if self.pos >= self.len:
                return b""

            if n is None or self.pos + n > self.len:
                n = self.len - self.pos

            self.mm.seek(self.loc * SECTOR_SIZE + self.pos)
            self.pos += n
            return self.mm.read(n)

        def readall(self) -> bytes:
            return self.read(None)

        def seek(self, pos: int, whence: int = 0) -> int:
            if whence == 0:
                new_pos = pos
            elif whence == 1:
                new_pos = self.pos + pos
            elif whence == 2:
                new_pos = self.len - pos
            else:
                raise ValueError("Invalid whence")
            if new_pos < 0:
                raise ValueError("Negative seek position")
            self.pos = new_pos
            return self.pos

        def readable(self) -> bool:
            return True

        def writable(self) -> bool:
            return False

        def seekable(self) -> bool:
            return True

        def isatty(self) -> bool:
            return False

        def fileno(self) -> int:
            raise OSError()

        def tell(self) -> int:
            return self.pos

        def truncate(self, size: int | None = None, /) -> int:
            raise OSError()

    class Variant(Enum):
        ISO9660 = 1
        Joliet = 2

    def __init__(self, path: str, variant: Variant = Variant.Joliet) -> None:
        self.path_to_loc: dict[str, dict[str, int]] = {}
        self.file = open(path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), length=0, access=mmap.ACCESS_READ)
        match variant:
            case ISO9660.Variant.ISO9660:
                if self._get_volume_descriptor_type(16) != 1:
                    raise ValueError(
                        "Primary volume descriptor must be primary volume descriptor"
                    )
            case ISO9660.Variant.Joliet:
                if self._get_volume_descriptor_type(17) != 2:
                    raise ValueError(
                        "Primary volume descriptor must be supplementary volume descriptor"
                    )
        self.variant = variant
        self.mm.seek(151, io.SEEK_CUR)
        pvd_loc, pvd_len = struct.unpack("<I4xI", self.mm.read(12))
        self._read_dir(pvd_loc, pvd_len)

    def _get_volume_descriptor_type(self, sector: int) -> int | None:
        self.mm.seek(sector * SECTOR_SIZE)
        type, identifier, version = struct.unpack("b5sb", self.mm.read(7))
        if identifier != b"CD001" or version != 1:
            raise ValueError("Not a valid ISO 9660 file")
        return type

    def _read_dir(self, start: int, total_len: int, path: str = "") -> None:
        n = 0
        while n < total_len:
            self.mm.seek(start * SECTOR_SIZE + n)
            rec_len, loc, len, flags, name_len = struct.unpack(
                "<bxI4xI11xb6xb", self.mm.read(33)
            )
            if rec_len < 1:
                n = (n // SECTOR_SIZE + 1) * SECTOR_SIZE
                continue
            n += rec_len
            name_bytes = self.mm.read(name_len)
            if name_bytes == b"\x00" or name_bytes == b"\x01":
                continue
            if self.variant == ISO9660.Variant.ISO9660:
                name = name_bytes.decode("ascii")
            elif self.variant == ISO9660.Variant.Joliet:
                name = name_bytes.decode("utf-16be")
            name = name.strip(";1")
            filename = path + name
            self.path_to_loc[filename] = {"loc": loc, "len": len}
            if flags & 0b10:
                self._read_dir(loc, len, filename + "/")

    def open(self, path: str) -> File:
        return self.File(
            self.mm, self.path_to_loc[path]["loc"], self.path_to_loc[path]["len"]
        )

    def close(self) -> None:
        self.mm.close()
        self.file.close()

    def filelist(self) -> list[str]:
        return list(self.path_to_loc.keys())

    def __enter__(self) -> "ISO9660":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
