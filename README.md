# Asset Ripper for LEGO Island (1997)

This is an asset ripper for the classic 1997 game LEGO Island. It extracts all assets, including music, dialogs, cutscenes, 3d models, and textures.

![lego_island_blender](https://github.com/user-attachments/assets/a0587ce7-7241-4262-ad6d-78dfae8bea8f)

It'll extract all generated files into the `extract` directory for each SI-file. If also a path to the isle
decompliation directory is provided, it'll generate the corresponding typescript files in `actions` for each SI-file.
The directory either needs to contain all header files directly or it must be the root directory of the decompliation
repository.

---

## Supported Formats

| Original Format  | Content                             | Converted Formats          |
|------------------|-------------------------------------|----------------------------|
| **Wave**         | Music, Dialogs                      | WAV                        |
| **Flic**         | Facial Expressions, 2D Animations   | Sprite Sheet (BMP), AVI, FLC |
| **Smacker**      | Cutscenes                           | AVI, SMK                   |
| **Images**       | Backgrounds, Textures               | BMP                        |
| **3D Models**    |                                     | GLB                        |

---

## Requirements

- Python 3.13+
- pillow 11.2.1+
- typed-ffmpeg 3.5.1+
- numpy 2.3.1+
- python-dotenv 1.1.1+

Note that `ffmpeg` needs to be available for `typed-ffmpeg`. Placing the binary (`ffmpeg.exe` or `ffmpeg`) itself in
this directory is sufficent.

---

## Usage

```bash
python extract.py <ISO_FILE> [-E] [--isle ISLEDECOMP]
```

Replace `<ISO_FILE>` with the path to your LEGO Island ISO file and if none is provided it'll query via a file open
dialog. Full extract takes around 5-10 minutes.

With `-E`/`--no-extract` no files are extracted and can be used to generate the typescript files. When `-I`/`--isle` or
`LEGO_ISLAND_DECOMP_FOLDER` point to the isle decompliation directory, it'll generate the typescript files. The
parameter takes precedence.

#### Example:

```bash
python extract.py lego_island.iso
```

#### Example with poetry:

```bash
poetry install
poetry run python extract.py lego_island.iso
```
