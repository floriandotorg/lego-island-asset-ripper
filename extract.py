import argparse
import io
import itertools
import json
import logging
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from tkinter import filedialog
from typing import Any

import cv2
import dotenv
import numpy as np
from PIL import Image

from lib.flc import FLC
from lib.iso import ISO9660
from lib.si import SI
from lib.smk import SMK
from lib.wdb import WDB

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
log_level = logging.INFO

transparent_color = [255, 0, 255]


def replace_ext(path, new_ext):
    root, _ = os.path.splitext(path)
    return root + new_ext


def ffmpeg(ffmpeg_args: list[str], input_bytes: bytes | None = None) -> None:
    params = ["ffmpeg"]
    if input_bytes:
        params.append("-i")
        params.append("pipe:0")
    params.append("-y")
    params.extend(ffmpeg_args)
    print(" ".join(params))
    proc = subprocess.Popen(params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(input=input_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode('utf-8', errors='ignore')}")


def imagemagick(imagemagick_args: list[str]) -> None:
    params = ["magick"]
    params.extend(imagemagick_args)
    print(" ".join(params))
    proc = subprocess.Popen(params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"imagemagick failed: {stderr.decode('utf-8', errors='ignore')}")


def write_png(data: bytes, width: int, height: int, filename: str, flip: bool = False) -> None:
    image = Image.frombytes("RGB", (width, height), data).convert("RGBA")
    if flip:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    img_data = np.array(image)
    mask = (img_data[:, :, :3] == transparent_color).all(axis=-1)
    img_data[mask, 3] = 0
    image = Image.fromarray(img_data, "RGBA")
    image.save(filename)
    imagemagick([filename, "-quality", "80", "-define", "heif:8", replace_ext(filename, ".avif")])


def get_image(obj: SI.Object) -> Image.Image:
    mem_file = io.BytesIO()
    mem_file.write(struct.pack("<2sIII", b"BM", len(obj.data), 0, obj.chunk_sizes[0] + 14))
    mem_file.write(obj.data)
    return Image.open(mem_file)


def write_bitmap(filename: str, obj: SI.Object) -> None:
    image = get_image(obj)
    img_array = np.array(image.convert("RGB"))
    filtered_img = cv2.bilateralFilter(img_array, 5, 50, 50)
    image = Image.fromarray(filtered_img)
    write_png(image.convert("RGB").tobytes(), image.width, image.height, filename)


def write_gif(gif: WDB.Gif, filename: str) -> None:
    write_png(gif.image, gif.width, gif.height, filename, flip=True)


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
    all_frames = bytearray()
    for frame in flc.frames:
        all_frames.extend(frame)
    write_png(all_frames, flc.width, flc.height * len(flc.frames), filename)


def get_iso_path(argument: str | None) -> str:
    if argument:
        return argument

    path = filedialog.askopenfilename(
        title="Select ISO file",
        filetypes=[("ISO files", "*.iso"), ("All files", "*.*")],
    )
    if not path:
        sys.exit("No file selected")
    return path


def write_video(filename: str, obj: SI.Object, mem_file: io.BytesIO, fps: int, width: int, height: int) -> None:
    folder_name = f"frames_{filename}_{obj.id}"
    os.makedirs(folder_name, exist_ok=True)
    ffmpeg([f"{folder_name}/frame_%05d.png"], mem_file.getvalue())
    for frame_file in os.listdir(folder_name):
        frame_path = f"{folder_name}/{frame_file}"
        image = cv2.imread(frame_path)
        if image is None:
            raise Exception(f"Error reading {frame_path}")
        filtered = cv2.bilateralFilter(image, 10, 75, 75)
        cv2.imwrite(frame_path, filtered)
    # h264 needs width and height to be divisible by 2
    ffmpeg(["-i", f"{folder_name}/frame_%05d.png", "-framerate", str(fps), "-vf", f"scale=w={width if width % 2 == 0 else width * 2}:h={height if height % 2 == 0 else height * 2}", "-c:v", "libx264", "-crf", "18", "-preset", "veryslow", "-pix_fmt", "yuv420p", "-avoid_negative_ts", "make_zero", "-fflags", "+genpts", "-movflags", "+faststart", f"extract/{filename}/{obj.id}.mp4"])
    shutil.rmtree(folder_name)


if __name__ == "__main__":
    logging.basicConfig(level=log_level)
    os.makedirs("extract", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("iso", nargs="?", help="path to the iso file (if not provided, does show file open dialog)")
    parser.add_argument("-E", "--no-extract", action="store_true", help="does not extract and convert the contents from ISO file")
    parser.add_argument("--isle", metavar="ISLEDECOMP", help="provide the file path to the decompilation project and generate typescript files from the header files")

    args = parser.parse_args()

    @dataclass
    class File:
        si: SI
        name: str

    def write_si(filename: str, obj: SI.Object) -> int:
        filename = filename.lower().replace(".si", "")

        os.makedirs(f"extract/{filename}", exist_ok=True)

        if args.no_extract:
            return 0

        match obj.file_type:
            case SI.FileType.OBJ:
                if obj.presenter == "LegoAnimPresenter" or obj.presenter == "LegoLocomotionAnimPresenter" or obj.presenter == "LegoCarBuildAnimPresenter":
                    with open(f"extract/{filename}/{obj.id}.ani", "wb") as f:
                        f.write(obj.data)
                    return 1

                if obj.presenter == "LegoPathPresenter":
                    with open(f"extract/{filename}/{obj.id}.gph", "wb") as f:
                        f.write(obj.data)
                    return 1

                if obj.presenter != "LegoModelPresenter" or not obj.data:
                    return 0

                # model_files = 0
                # wdb = WDB(io.BytesIO(obj.data), read_si_model=True)
                # for model in wdb.models:
                #     model_files += export_wdb_model(wdb, model, f"extract/{filename}/models")
                # os.makedirs(f"extract/{filename}/models/textures", exist_ok=True)
                # for texture in wdb.model_textures:
                #     write_gif(texture, f"extract/{filename}/models/textures/{obj.id}.png")
                # return model_files + len(wdb.model_textures)
                return 0

            case SI.FileType.WAV:

                def extend_wav_chunk(type: bytes, content: bytes) -> bytes:
                    result = bytearray()
                    result.extend(struct.pack("<4sI", type, len(content)))
                    result.extend(content)
                    if (len(content) % 2) == 1:
                        result.append(0)
                    return bytes(result)

                content = bytearray()
                content.extend(b"WAVE")
                content.extend(extend_wav_chunk(b"fmt ", obj.data[: obj.chunk_sizes[0]]))
                content.extend(extend_wav_chunk(b"data", obj.data[obj.chunk_sizes[0] :]))
                wav = bytearray()
                wav.extend(extend_wav_chunk(b"RIFF", content))

                ffmpeg(["-ac", "1", "-ar", "11025", "-c:a", "libfdk_aac", "-profile:a", "aac_low", "-vbr", "3", "-movflags", "+faststart", f"extract/{filename}/{obj.id}.m4a"], wav)

                return 1
            case SI.FileType.STL:
                write_bitmap(f"extract/{filename}/{obj.id}.png", obj)
                return 1
            case SI.FileType.FLC:
                mem_file = io.BytesIO()
                write_flc(mem_file, obj)
                # Manually patch the corrupted flic #65 in jukebox
                # The size of the first chunk's subchunk is 5196 even though it should be 5198 (w*h + header).
                if filename == "jukebox" and obj.id == 65:
                    original_data = mem_file.getbuffer()
                    assert original_data[0x39A] == 76
                    original_data[0x39A] += 2
                    assert original_data[0x39A] == 78
                mem_file.seek(0)
                try:
                    flc = FLC(mem_file)
                    mem_file.seek(0)
                    # face animations don't need to be filtered
                    if flc.width == 128:
                        # h264 needs width and height to be divisible by 2
                        ffmpeg(["-vf", f"scale=w={flc.width if flc.width % 2 == 0 else flc.width * 2}:h={flc.height if flc.height % 2 == 0 else flc.height * 2}", "-c:v", "libx264", "-crf", "18", "-preset", "veryslow", "-pix_fmt", "yuv420p", "-avoid_negative_ts", "make_zero", "-fflags", "+genpts", "-movflags", "+faststart", f"extract/{filename}/{obj.id}.mp4"], mem_file.getvalue())
                    else:
                        write_video(filename, obj, mem_file, flc.fps, flc.width, flc.height)
                except Exception as e:
                    logger.error(f"Error writing {filename}_{obj.id}.flc: {e}")
                return 1
            case SI.FileType.SMK:
                mem_file = io.BytesIO()
                mem_file.write(obj.data)
                mem_file.seek(0)
                smk = SMK(mem_file)
                mem_file.seek(0)
                write_video(filename, obj, mem_file, smk.fps, smk.width, smk.height)
                return 1
            case _:
                return 0

    def process_file(file: File) -> int:
        logger.info(f"Extracting {file.name} ..")
        result = sum(write_si(os.path.basename(file.name), obj) for obj in file.si.object_list.values())
        logger.info(f"Extracting {file.name} .. [done]")
        return result

    si_files: list[File] = []
    wdb_files: list[io.BytesIO] = []
    with ISO9660(get_iso_path(args.iso)) as iso:
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

    if not args.no_extract:
        exported_files = 0

        logger.info("Exporting WDB textures ..")
        os.makedirs("extract/world", exist_ok=True)
        os.makedirs("extract/world/images", exist_ok=True)
        os.makedirs("extract/world/part_textures", exist_ok=True)
        os.makedirs("extract/world/model_textures", exist_ok=True)
        for wdb_file in wdb_files:
            with open("extract/world.wdb", "wb") as f:
                f.write(wdb_file.getvalue())

            wdb = WDB(io.BytesIO(open("extract/world.wdb", "rb").read()))
            for image in wdb.images:
                write_gif(image, f"extract/world/images/{image.title.lower()}.png")
            for texture in wdb.part_textures:
                write_gif(texture, f"extract/world/part_textures/{texture.title.lower()}.png")
            for model_texture in wdb.model_textures:
                write_gif(model_texture, f"extract/world/model_textures/{model_texture.title.lower()}.png")
            exported_files += len(wdb.images) + len(wdb.part_textures) + len(wdb.model_textures)
        logger.info("Exporting WDB textures .. [done]")

        for si_file in si_files:
            process_file(si_file)

        logger.info(f"Exported {exported_files} files")

    if args.isle:
        isle_path_str = args.isle
    else:
        isle_path_str = os.getenv("LEGO_ISLAND_DECOMP_FOLDER")

    if isle_path_str:
        isle_path = pathlib.Path(isle_path_str)
        if not isle_path.is_dir():
            logger.error("Isle path is not a valid directory")
            sys.exit(1)

        if "LEGO1" in (p.name for p in isle_path.iterdir()):
            isle_path = isle_path / "LEGO1" / "lego" / "legoomni" / "include" / "actions"

        objects: dict[str, dict[int, SI.Object]] = defaultdict(dict)

        def walk(objs: list[SI.Object]) -> list[SI.Object]:
            result: list[SI.Object] = []
            for obj in objs:
                result.append(obj)
            return result

        for si_file in si_files:
            for obj in si_file.si.object_list.values():
                objects[si_file.name][obj.id] = obj

        def filter_none_deep(data: Any) -> Any:
            if isinstance(data, SI.Object):
                data = data.to_dict()

            if isinstance(data, list):
                return [filter_none_deep(item) for item in data]
            elif isinstance(data, dict):
                return {key: filter_none_deep(value) for key, value in data.items() if value is not None}
            elif isinstance(data, str):
                return data if data else None
            else:
                return data

        os.makedirs("actions", exist_ok=True)
        with open("actions/types.ts", "w") as tfile:
            tfile.write("export namespace Action {\n")

            tfile.write("export enum Type {\n")
            for name, tvalue in SI.Type.__members__.items():
                tfile.write(f"    {name} = {tvalue},\n")
            tfile.write("}\n")

            tfile.write("export enum FileType {\n")
            for name, fvalue in SI.FileType.__members__.items():
                tfile.write(f"    {name} = {fvalue},\n")
            tfile.write("}\n")

            tfile.write("export enum Flags {\n")
            for name, flvalue in SI.Flags.__members__.items():
                tfile.write(f"    {name} = {flvalue},\n")
            tfile.write("}\n")

            tfile.write("}\n")

        def set_filename(obj: SI.Object, filename: str) -> None:
            obj.si_file = filename
            for child in obj.children:
                set_filename(child, filename)

        for filename, obj_dict in objects.items():
            filename = os.path.basename(filename).lower().replace(".si", "")
            print(f"Processing {filename} ..")

            for obj in obj_dict.values():
                if "Map;" in obj.extra_data or "Filler_index" in obj.extra_data:
                    obj.should_export_palette = True
                    for child in obj.children:
                        child.should_export_palette = True

            for obj in obj_dict.values():
                if obj.file_type == SI.FileType.STL:
                    still = get_image(obj)
                    obj.dimensions = SI.Dimensions(still.width, still.height)
                    palette = still.getpalette()
                    if palette and obj.should_export_palette:
                        assert len(palette) % 3 == 0
                        obj.color_palette = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in itertools.batched(palette, 3)]
                elif obj.file_type == SI.FileType.FLC:
                    try:
                        mem_file = io.BytesIO()
                        write_flc(mem_file, obj)
                        mem_file.seek(0)
                        flc = FLC(mem_file)
                        obj.dimensions = SI.Dimensions(flc.width, flc.height)
                    except Exception as e:
                        logger.error(f"Error reading {filename}_{obj.id}.flc: {e}")
                elif obj.file_type == SI.FileType.SMK:
                    smk = SMK(io.BytesIO(obj.data))
                    obj.dimensions = SI.Dimensions(smk.width, smk.height)

            with open(isle_path / f"{filename}_actions.h", "r") as hfile:
                matches = re.findall(r"c_([A-Z0-9_]+)\s*=\s*(\d+)", hfile.read(), re.IGNORECASE | re.MULTILINE)
                with open(f"actions/{filename}.ts", "w") as tfile:
                    tfile.write('import { Action } from "./types"\n')

                    for match in matches:
                        action = obj_dict[int(match[1])]
                        set_filename(action, filename)
                        obj_str = json.dumps(filter_none_deep(action.to_dict()), indent=2)
                        obj_str = re.sub(r"\"type\": (\d+)", lambda match: f'"type": Action.Type.{SI.Type(int(match.group(1))).name}', obj_str)
                        obj_str = re.sub(r"\"file_type\": (\d+)", lambda match: f'"file_type": Action.FileType.{SI.FileType(int(match.group(1))).name}', obj_str)
                        tfile.write(f"export const {match[0]} = {obj_str} as const\n")
