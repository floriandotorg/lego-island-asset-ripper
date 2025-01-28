import io
import logging
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

HEADER_SIZE = 8
CHUNK_HEADER_SIZE = 14


class SMK:
    @dataclass
    class Color:
        r: int
        g: int
        b: int

        def __bytes__(self) -> bytes:
            return struct.pack("<BBB", self.r, self.g, self.b)

    # spell-checker: ignore palmap
    _palmap = bytes([0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C, 0x41, 0x45, 0x49, 0x4D, 0x51, 0x55, 0x59, 0x5D, 0x61, 0x65, 0x69, 0x6D, 0x71, 0x75, 0x79, 0x7D, 0x82, 0x86, 0x8A, 0x8E, 0x92, 0x96, 0x9A, 0x9E, 0xA2, 0xA6, 0xAA, 0xAE, 0xB2, 0xB6, 0xBA, 0xBE, 0xC3, 0xC7, 0xCB, 0xCF, 0xD3, 0xD7, 0xDB, 0xDF, 0xE3, 0xE7, 0xEB, 0xEF, 0xF3, 0xF7, 0xFB, 0xFF])

    class BlockType(IntEnum):
        Mono = 0
        Full = 1
        Void = 2
        Solid = 3

    # spell-checker: ignore sizetable
    _sizetable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 128, 256, 512, 1024, 2048]

    def __init__(self, file: io.BufferedIOBase):
        self._current_byte: int = 0
        self._current_bit: int = 0
        self._file = file
        signature, self._width, self._height, self._num_frames, self._framerate, flags, trees_size = struct.unpack("<4sIIIiI28xI48x", self._file.read(104))
        if signature != b"SMK2":
            raise ValueError(f"Invalid SMK file: {signature}")
        if flags != 0:
            raise ValueError(f"Unsupported flags: {flags:x}")
        if self._framerate > 0:
            self._framerate = 1_000 / self._framerate
        elif self._framerate < 0:
            self._framerate = 100_000 / -self._framerate
        else:
            self._framerate = 10
        logger.debug(f"{self._width=} {self._height=} {self._num_frames=} {self._framerate=}")
        frame_sizes = struct.unpack("<" + "I" * self._num_frames, self._file.read(self._num_frames * 4))
        is_keyframe = [frame_size & 0x01 == 1 for frame_size in frame_sizes]
        frame_sizes = tuple(frame_size & ~0x03 for frame_size in frame_sizes)
        logger.debug(f"{is_keyframe=} {frame_sizes=}")
        frame_types = struct.unpack("<" + "b" * self._num_frames, self._file.read(self._num_frames))
        if any(type & ~0x01 != 0 for type in frame_types):
            raise ValueError("Audio is not supported")
        has_palette = [frame_type & 0x01 == 1 for frame_type in frame_types]

        end_of_trees = self._file.tell() + trees_size

        self._init_bit_reader()

        self._mmap = self._read_big_tree()
        # spell-checker: ignore mclr
        self._mclr = self._read_big_tree()
        self._full = self._read_big_tree()
        self._type = self._read_big_tree()

        self._file.seek(end_of_trees)

        blocks_per_frame = math.ceil(self._width / 4) * math.ceil(self._height / 4)

        if self._width % 4 != 0 or self._height % 4 != 0:
            raise ValueError("Width and height must be divisible by 4 or these idiots need to fix their code")

        self._frames: list[bytes] = []
        for frame in range(self._num_frames):
            if has_palette[frame]:
                self._palette = [self.Color(0, 0, 0)] * 256
                self._read_palette()

            self._init_bit_reader()

            current_block = 0
            current_frame = bytearray(self._frames[-1] if self._frames else b"\xff" * self._width * self._height * 3)
            while current_block < blocks_per_frame:
                current_block += self._read_chain(current_frame, current_block)

            self._frames.append(bytes(current_frame))

            break

    @property
    def frames(self) -> list[bytes]:
        return self._frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> int:
        return self._framerate

    def _read_chain(self, frame: bytearray, current_block: int) -> int:
        block = self._read_bits_until_found(self._type)
        type = SMK.BlockType(block & 0x03)
        size = SMK._sizetable[(block & 0xFC) >> 2]
        extra = (block & 0xFF00) >> 8
        print(f"{block=:04x} {type.name} size={(block & 0xFC) >> 2} ({size}) {extra=:x}")
        match type:
            case SMK.BlockType.Mono:
                for n in range(size):
                    colors = self._read_bits_until_found(self._mclr)
                    map = self._read_bits_until_found(self._mmap)
                    color1 = bytes(self._palette[colors & 0xFF])
                    color2 = bytes(self._palette[colors >> 8 & 0xFF])
                    color_rgb = bytearray(b"\00" * 16 * 3)
                    for n in range(16):
                        color_rgb[n * 3 : n * 3 + 3] = color1 if map & 0x01 else color2
                        map >>= 1
                    y_offset = (current_block + n) // math.ceil(self._width / 4) * 4
                    x_offset = (current_block + n) % math.ceil(self._width / 4) * 4
                    for y in range(4):
                        p = (x_offset + (y_offset + y) * self._width) * 3
                        frame[p : p + 3 * 4] = color_rgb
            case SMK.BlockType.Full:
                for n in range(size):
                    colors_rgb = bytearray(b"\00" * 16 * 3)
                    for n in [2, 0, 6, 4, 10, 8, 14, 12]:
                        print("full tree")
                        colors = self._read_bits_until_found(self._full)
                        print("full tree end")
                        colors_rgb[n * 3 : n * 3 + 3] = bytes(self._palette[colors & 0xFF])
                        colors_rgb[n * 3 + 3 : n * 3 + 6] = bytes(self._palette[colors >> 8 & 0xFF])
                    y_offset = (current_block + n) // math.ceil(self._width / 4) * 4
                    x_offset = (current_block + n) % math.ceil(self._width / 4) * 4
                    for y in range(4):
                        p = (x_offset + (y_offset + y) * self._width) * 3
                        frame[p : p + 3 * 4] = colors_rgb
            case SMK.BlockType.Void:
                pass
            case SMK.BlockType.Solid:
                colors_rgb = bytearray(bytes(self._palette[extra]) * 4)
                for n in range(size):
                    y_offset = (current_block + n) // math.ceil(self._width / 4) * 4
                    x_offset = (current_block + n) % math.ceil(self._width / 4) * 4
                    for y in range(4):
                        p = (x_offset + (y_offset + y) * self._width) * 3
                        frame[p : p + 3 * 4] = colors_rgb
        return size

    def print_table(self, data: dict[str, int]) -> None:
        max_width = max(len(k) for k in data.keys())
        print(f"{'Key'.ljust(max_width)} | {'Value'.ljust(10)}")
        print("-" * (max_width + 10 + 3))

        for k, v in data.items():
            print(f"{k.ljust(max_width)} | {f"0b{v:08b}".ljust(10)}")

    def _read_palette(self) -> None:
        prev_palette = list(self._palette)
        palette_end = self._file.tell() + self._file.read(1)[0] * 4 - 1
        n = 0
        while self._file.tell() <= palette_end:
            block = self._file.read(1)[0]
            if block & 0x80:
                c = (block & 0x7F) + 1
                n += c
                logger.debug(f"skip {c} colors")
            elif block & 0x40:
                c = (block & 0x3F) + 1
                s = self._file.read(1)[0]
                for i in range(c):
                    self._palette[n] = prev_palette[s + i]
                    n += 1
                logger.debug(f"read {c} colors starting at {s}")
            else:
                b = (block & 0x3F) + 1
                g = self._file.read(1)[0] & 0x3F
                r = self._file.read(1)[0] & 0x3F
                self._palette[n] = SMK.Color(self._palmap[r], self._palmap[g], self._palmap[b])
                n += 1
                logger.debug(f"read {SMK.Color(self._palmap[r], self._palmap[g], self._palmap[b])}")

    def _init_bit_reader(self) -> None:
        self._current_byte = self._file.read(1)[0]
        self._current_bit = 0

    def _read_bit(self) -> int:
        result = (self._current_byte >> self._current_bit) & 1

        if self._current_bit >= 7:
            self._current_byte = self._file.read(1)[0]
            self._current_bit = 0
        else:
            self._current_bit += 1

        return result

    def _read_8_bits(self) -> int:
        if self._current_bit == 0:
            result = self._current_byte
            self._current_byte = self._file.read(1)[0]
            return result

        result = self._current_byte >> self._current_bit
        self._current_byte = self._file.read(1)[0]
        return result | self._current_byte << (8 - self._current_bit) & 0xFF

    @dataclass
    class HuffmanNode:
        value: Optional[int] = None
        zero: Optional["SMK.HuffmanNode"] = None
        one: Optional["SMK.HuffmanNode"] = None
        caches: Optional[tuple[int, int, int]] = None
        cached: bool = False

        def __repr__(self) -> str:
            return self._repr_helper()

        def _repr_helper(self, prefix: str = "", is_left: bool = True) -> str:
            result = []

            branch = "└── " if is_left else "┌── "
            result.append(prefix + branch + (f"[{self.value}]" if self.value is not None else "*"))

            new_prefix = prefix + ("    " if is_left else "│   ")

            if self.one:
                result.append(self.one._repr_helper(new_prefix, False))
            if self.zero:
                result.append(self.zero._repr_helper(new_prefix, True))

            return "\n".join(result)

    def _build_huffman(self, high_byte_tree: Optional[HuffmanNode] = None, low_byte_tree: Optional[HuffmanNode] = None, caches: Optional[tuple[int, int, int]] = None) -> HuffmanNode:
        if (high_byte_tree is None and low_byte_tree is None) and self._read_bit() == 0:
            raise ValueError("Tree is not present")

        root = SMK.HuffmanNode()
        root.caches = caches
        self._read_huffman(root, root, high_byte_tree, low_byte_tree)

        if (high_byte_tree is None and low_byte_tree is None) and self._read_bit() != 0:
            raise ValueError("Error reading tree")

        return root

    def _tree_lookup(self, tree: HuffmanNode, bits: list[int]) -> HuffmanNode:
        for bit in bits:
            if bit == 0:
                if tree.zero is None:
                    raise ValueError("Invalid bit sequence")
                tree = tree.zero
            else:
                if tree.one is None:
                    raise ValueError("Invalid bit sequence")
                tree = tree.one
        return tree

    def _read_bits_until_found(self, tree: HuffmanNode) -> int:
        bits = []
        while True:
            bits.append(self._read_bit())
            node = self._tree_lookup(tree, bits)
            if node.value is not None:
                value = node.value

                if node.cached:
                    if tree.caches is None:
                        raise ValueError("No caches")

                    print(f"cache: {node.value:x}")

                    value = tree.caches[node.value]

                if tree.caches is not None and tree.caches[0] != value:
                    tree.caches = (value, tree.caches[0], tree.caches[1])
                    print(f"caches: {tree.caches[0]:x} {tree.caches[1]:x} {tree.caches[2]:x}")

                return value

    def _read_huffman(self, tree: HuffmanNode, current_node: HuffmanNode, high_byte_tree: Optional[HuffmanNode], low_byte_tree: Optional[HuffmanNode]) -> None:
        flag = self._read_bit()
        if flag != 0:
            current_node.zero = SMK.HuffmanNode()
            self._read_huffman(tree, current_node.zero, high_byte_tree, low_byte_tree)

            current_node.one = SMK.HuffmanNode()
            self._read_huffman(tree, current_node.one, high_byte_tree, low_byte_tree)
        else:
            if high_byte_tree is None or low_byte_tree is None:
                current_node.value = self._read_8_bits()
            else:
                current_node.value = self._read_bits_until_found(low_byte_tree) | (self._read_bits_until_found(high_byte_tree) << 8)

                if tree.caches is None:
                    raise ValueError("No caches")

                if current_node.value in tree.caches:
                    current_node.cached = True
                    current_node.value = tree.caches.index(current_node.value)

    # def _huffman_tree_to_table(self, tree: HuffmanNode) -> dict[str, int]:
    #     table: dict[str, int] = {}
    #     self._huffman_tree_to_table_helper(tree, 0, 0, table)
    #     return table

    # def _huffman_tree_to_table_helper(self, tree: HuffmanNode, key: int, key_length: int, table: dict[str, int]) -> None:
    #     if tree.value is not None:
    #         key_str = f"{key:0{key_length}b}"
    #         if key_str in table:
    #             raise ValueError(f"Duplicate key: {key_str}")
    #         table[key_str] = tree.value
    #     else:
    #         if tree.zero:
    #             self._huffman_tree_to_table_helper(tree.zero, key << 1, key_length + 1, table)
    #         if tree.one:
    #             self._huffman_tree_to_table_helper(tree.one, (key << 1) | 1, key_length + 1, table)

    def _read_big_tree(self) -> HuffmanNode:
        if self._read_bit() == 0:
            raise ValueError("Big tree is not present")

        low_byte_tree = self._build_huffman()
        # logger.debug("\n" + repr(low_byte_tree))
        # low_byte_table = self._huffman_tree_to_table(low_byte_tree)
        # self.print_table(low_byte_table)

        high_byte_tree = self._build_huffman()
        # logger.debug("\n" + repr(high_byte_tree))
        # high_byte_table = self._huffman_tree_to_table(high_byte_tree)
        # self.print_table(high_byte_table)

        c1 = self._read_8_bits() | (self._read_8_bits() << 8)
        c2 = self._read_8_bits() | (self._read_8_bits() << 8)
        c3 = self._read_8_bits() | (self._read_8_bits() << 8)

        print(f"cache 0: {c1:x}")
        print(f"cache 1: {c2:x}")
        print(f"cache 2: {c3:x}")

        big_tree = self._build_huffman(high_byte_tree, low_byte_tree, (c1, c2, c3))
        # logger.debug("\n" + repr(big_tree))
        # big_table = self._huffman_tree_to_table(big_tree)
        # self.print_table(big_table)

        big_tree.caches = (0, 0, 0)

        if self._read_bit() != 0:
            raise ValueError("Error reading big tree")

        return big_tree
