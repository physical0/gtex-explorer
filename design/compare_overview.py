"""Normalize existing QA captures only. Does not generate or change application UI."""
from pathlib import Path
from PIL import Image

directory = Path(__file__).resolve().parent
reference = Image.open(directory / "overview-reference.png").convert("RGB")
capture = Image.open(directory / "overview-initial.png").convert("RGB")

# Remove native title-bar chrome and compare at the same visible content width.
implementation = capture.crop((0, 31, capture.width, capture.height))
reference = reference.resize(
    (implementation.width, round(reference.height * implementation.width / reference.width)),
    Image.Resampling.LANCZOS,
)
reference = reference.crop((0, 0, implementation.width, implementation.height))
comparison = Image.new("RGB", (implementation.width * 2, implementation.height), "white")
comparison.paste(reference, (0, 0))
comparison.paste(implementation, (implementation.width, 0))
comparison.save(directory / "overview-initial-comparison.png")

# A separate close-up preserves legibility of configuration and result labels.
reference_detail = reference.crop((280, 275, 1410, 540))
implementation_detail = implementation.crop((268, 310, 1388, 600))
detail = Image.new("RGB", (2250, 290), "white")
detail.paste(reference_detail, (0, 0))
detail.paste(implementation_detail, (1130, 0))
detail.save(directory / "overview-initial-detail.png")
