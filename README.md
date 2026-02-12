# Webcomic Panel Cropper

Small Python tool that automatically detects and crops panels from a webcomic page image.

## Install
```bash
pip install -r requirements.txt
```

## Usage
Single image:
```bash
python auto_crop_panels.py page.png
```

Multiple images:
```bash
python auto_crop_panels.py *.png
```

Debug:
```bash
python auto_crop_panels.py page.png --debug --keep-debug
```

## Output
Cropped panels are written to ./cropped/ next to the input image.
