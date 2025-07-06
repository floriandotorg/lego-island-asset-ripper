# Asset Ripper for LEGO Island (1997)

This is an asset ripper for the classic 1997 game LEGO Island. It extracts all assets, including music, dialogs, cutscenes, 3d models, and textures.

![lego_island_blender](https://github.com/user-attachments/assets/a0587ce7-7241-4262-ad6d-78dfae8bea8f)

It'll extract all generated files into the `extract` directory for each SI-file.

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

---

## Usage

```bash
python extract.py <ISO_FILE>
```

Replace `<ISO_FILE>` with the path to your LEGO Island ISO file and if none is provided it'll query via a file open
dialog. Full extract takes around 5-10 minutes.

#### Example:

```bash
python extract.py lego_island.iso
```

#### Example with poetry:

```bash
poetry install
poetry run python extract.py lego_island.iso
```
