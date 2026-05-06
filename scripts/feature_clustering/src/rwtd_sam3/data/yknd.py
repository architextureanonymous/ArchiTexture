"""YKND dataset adapter for SAM auditing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

YKND_DATASET_ID = "YKND"

@dataclass(frozen=True)
class YkndSample:
    crop_name: str
    image: Image.Image
    width: int
    height: int
    label_mask: np.ndarray  # Categorical: [0, 1, 2, 3, 4, 5]

# Unique labels in YKND: [0, 1, 2, 4, 5, 6]
# remap to consecutive integers for CrossEntropy
YKND_LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
    5: 4,
    6: 5
}

def remap_yknd_labels(label_img: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(label_img)
    for src, dst in YKND_LABEL_MAP.items():
        remapped[label_img == src] = dst
    return remapped

def resolve_yknd_pairs(root: Path) -> list[tuple[Path, Path]]:
    img_dir = root / "images"
    label_dir = root / "labels"
    
    if not img_dir.exists() or not label_dir.exists():
        return []
        
    img_paths = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    pairs = []
    for img_path in img_paths:
        label_path = label_dir / f"{img_path.stem}.png"
        if label_path.exists():
            pairs.append((img_path, label_path))
            
    return pairs

def iter_yknd_samples(
    root: str | Path,
    *,
    split: str = "train",
    test_stems: Sequence[str] | None = None,
    limit: int | None = None,
) -> Iterator[YkndSample]:
    root = Path(root)
    pairs = resolve_yknd_pairs(root)
    
    if not pairs:
        LOGGER.warning(f"No image-label pairs found in {root}")
        return

    # Deterministic split
    test_stems = set(test_stems or [])
    
    count = 0
    for img_path, label_path in pairs:
        is_test = img_path.stem in test_stems
        if split == "train" and is_test:
            continue
        if split == "test" and not is_test:
            continue
            
        img = Image.open(img_path)
        label_img = np.array(Image.open(label_path))
        
        # Categorical remapping
        label_mask = remap_yknd_labels(label_img)
        
        yield YkndSample(
            crop_name=img_path.stem,
            image=img,
            width=img.width,
            height=img.height,
            label_mask=label_mask,
        )
        
        count += 1
        if limit is not None and count >= limit:
            break
