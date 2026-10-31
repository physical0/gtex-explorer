"""Compose existing screenshots for dictionary visual QA, without changing them."""
from pathlib import Path
from PIL import Image

folder = Path(__file__).resolve().parent
source = Image.open(folder.parent / "biology-dictionary-target-20261020.png").convert("RGB")
for name in ["01-initial.jpg", "02-final.jpg"]:
    path = folder / name
    if not path.exists():
        continue
    screen = Image.open(path).convert("RGB")
    assert screen.size == source.size, (screen.size, source.size)
    comparison = Image.new("RGB", (source.width * 2, source.height), "white")
    comparison.paste(source, (0, 0))
    comparison.paste(screen, (source.width, 0))
    comparison.save(folder / f"{path.stem}-comparison.png")
    # Readable side-by-side detail of term selection and the reading panel.
    crop = (290, 205, 1465, 1000)
    left, right = source.crop(crop), screen.crop(crop)
    detail = Image.new("RGB", (left.width * 2, left.height), "white")
    detail.paste(left, (0, 0))
    detail.paste(right, (left.width, 0))
    detail.save(folder / f"{path.stem}-detail.png")
