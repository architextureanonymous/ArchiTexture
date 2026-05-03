from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io_utils import ensure_binary


PROMPT_MASK_RE = re.compile(r"^rwtd_(?P<image_id>\d+)_p\d+_m\d+\.png$")


@dataclass(frozen=True)
class ProposalLoadConfig:
    min_area: int = 16
    dedupe_exact: bool = True


class PromptMaskProposalStore:
    """
    Loads precomputed SAM prompt masks from a flat directory.

    Expected file pattern: rwtd_<image_id>_pXX_mYY.png
    """

    def __init__(self, prompt_masks_root: Path, cfg: ProposalLoadConfig | None = None):
        self.prompt_masks_root = prompt_masks_root
        self.cfg = cfg or ProposalLoadConfig()
        self._index: dict[int, list[Path]] = self._build_index()

    def _build_index(self) -> dict[int, list[Path]]:
        if not self.prompt_masks_root.exists():
            raise FileNotFoundError(f"Prompt mask directory not found: {self.prompt_masks_root}")

        by_id: dict[int, list[Path]] = {}
        for p in sorted(self.prompt_masks_root.glob("*.png")):
            m = PROMPT_MASK_RE.match(p.name)
            if m is None:
                continue
            image_id = int(m.group("image_id"))
            by_id.setdefault(image_id, []).append(p)
        return by_id

    @staticmethod
    def _resolve_readable_path(path: Path) -> Path | None:
        if path.exists():
            return path
        if not path.is_symlink():
            return None

        target_raw = Path(os.readlink(path))
        if target_raw.is_absolute() and target_raw.exists():
            return target_raw

        target_from_link = (path.parent / target_raw).resolve(strict=False)
        if target_from_link.exists():
            return target_from_link

        # ArchiTexture contains copied report trees whose relative symlinks
        # were authored from a shallower workspace. Recover the original path
        # under $HOME when the target suffix begins at `repo/...`.
        if "repo" in target_raw.parts:
            repo_idx = target_raw.parts.index("repo")
            home_candidate = Path.home().joinpath(*target_raw.parts[repo_idx:])
            if home_candidate.exists():
                return home_candidate

        return None

    @property
    def image_ids(self) -> set[int]:
        return set(self._index.keys())

    def load(self, image_id: int, expected_shape: tuple[int, int] | None = None) -> list[np.ndarray]:
        files = self._index.get(image_id, [])
        if not files:
            return []

        masks: list[np.ndarray] = []
        seen: set[bytes] = set()

        for path in files:
            readable_path = self._resolve_readable_path(path)
            if readable_path is None:
                continue
            arr = cv2.imread(str(readable_path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            mask = ensure_binary(arr)
            if expected_shape is not None and mask.shape != expected_shape:
                continue
            if int(mask.sum()) < self.cfg.min_area:
                continue

            if self.cfg.dedupe_exact:
                key = np.packbits(mask.astype(np.uint8), axis=None).tobytes()
                if key in seen:
                    continue
                seen.add(key)

            masks.append(mask.astype(np.uint8))

        return masks


def union_proposals(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    out = np.zeros(shape, dtype=np.uint8)
    for m in masks:
        out = np.logical_or(out, m > 0)
    return out.astype(np.uint8)
