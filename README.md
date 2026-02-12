# Webcomic Panel Cropper

Small Python tool that automatically detects and crops panels from a webcomic page image.

## Quick start (all OS)
```bash
pip install -r requirements.txt
python run.py
```
And for specific files:
```bash
python run.py page1.png page2.jpg
python run.py ./folder_with_pages
python run.py *.png
```

## Output
Cropped panels are written to ./cropped/ next to the input image.
