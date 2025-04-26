# Asset Ripper for LEGO Island (1997)

This project is an open-source Python tool designed specifically to extract and convert all game assets from the classic 1997 LEGO Island game into modern, accessible formats.

---

## Supported Formats

| Original Format  | Content                             | Converted Formats          |
|------------------|-------------------------------------|----------------------------|
| **Wave**         | Music, Dialogs                      | WAV                        |
| **Flic**         | Facial Expressions, 2D Animations   | Sprite Sheet (BMP), AVI, FLC |
| **Smacker**      | Cutscenes                           | AVI, SMK                   |
| **Images**       | Backgrounds, Textures               | BMP                        |
| **3D Models**    |                                     | GLB                        |
| **Animations**   |                                     | WiP                        | 

---

## Requirements

- Python 3.13+

---

## Usage

```bash
python extract.py <ISO_FILE>
```

Replace `<ISO_FILE>` with the path to your LEGO Island ISO file. Full extract takes around 5-10 minutes.

Example:

```bash
python extract.py lego_island.iso
```
