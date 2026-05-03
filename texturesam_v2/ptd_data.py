from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class PTDEntry:
    rel_path: str
    class_name: str
    class_id: int


@dataclass(frozen=True)
class PTDSplit:
    train: list[PTDEntry]
    val: list[PTDEntry]


class PTDImageBackend:
    """
    PTD reader that supports:
    - extracted layout: <root>/ptd/images/<class>/<file>.png
    - zip layout: <root>/ptd.zip
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.zip_path = self.root / "ptd.zip"
        self.images_root = self.root / "ptd" / "images"
        self.classes_path = self.root / "ptd" / "classes.txt"
        self.meta_path = self.root / "ptd" / "metafile.txt"

        self._zip: zipfile.ZipFile | None = None

        if not self.images_root.exists():
            if not self.zip_path.exists():
                raise FileNotFoundError(
                    f"PTD not found under {self.root}. Expected either extracted "
                    f"'{self.images_root}' or archive '{self.zip_path}'."
                )

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __del__(self) -> None:
        self.close()

    def _get_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
        return self._zip

    def _read_text(self, rel_file: str) -> str:
        p = self.root / "ptd" / rel_file
        if p.exists():
            return p.read_text(encoding="utf-8")
        z = self._get_zip()
        return z.read(f"ptd/{rel_file}").decode("utf-8")

    def read_optional_text(self, rel_file: str) -> str | None:
        p = self.root / "ptd" / rel_file
        if p.exists():
            return p.read_text(encoding="utf-8")
        if self.zip_path.exists():
            z = self._get_zip()
            try:
                return z.read(f"ptd/{rel_file}").decode("utf-8")
            except KeyError:
                return None
        return None

    def class_names(self) -> list[str]:
        lines = self._read_text("classes.txt").splitlines()
        out = [x.strip() for x in lines if x.strip()]
        if not out:
            raise RuntimeError("PTD classes.txt is empty.")
        return out

    def manifest_lines(self) -> list[str]:
        lines = self._read_text("metafile.txt").splitlines()
        out = [x.strip() for x in lines if x.strip()]
        if not out:
            raise RuntimeError("PTD metafile.txt is empty.")
        return out

    def read_rgb(self, rel_path: str) -> np.ndarray:
        rel_path = rel_path.strip().lstrip("/")
        p = self.images_root / rel_path
        if p.exists():
            arr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if arr is None:
                raise FileNotFoundError(f"Failed to read PTD image: {p}")
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        z = self._get_zip()
        try:
            b = z.read(f"ptd/images/{rel_path}")
        except KeyError as exc:
            raise FileNotFoundError(f"Missing PTD zip entry: ptd/images/{rel_path}") from exc
        buf = np.frombuffer(b, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Failed to decode PTD image from zip: {rel_path}")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def load_ptd_entries(root: Path) -> tuple[list[str], list[PTDEntry]]:
    backend = PTDImageBackend(root)
    class_names = backend.class_names()
    cls_to_id = {c: i for i, c in enumerate(class_names)}
    items: list[PTDEntry] = []

    for ln in backend.manifest_lines():
        # format: "<class_name>/<filename>.png"
        parts = ln.split("/", 1)
        if len(parts) != 2:
            continue
        cname, _ = parts
        if cname not in cls_to_id:
            continue
        items.append(PTDEntry(rel_path=ln, class_name=cname, class_id=cls_to_id[cname]))

    if not items:
        raise RuntimeError("PTD manifest produced zero usable entries.")
    return class_names, items


def split_ptd_entries(
    entries: Iterable[PTDEntry],
    *,
    val_fraction: float = 0.10,
    split_seed: int = 1337,
    root: Path | None = None,
) -> PTDSplit:
    arr = list(entries)
    if root is not None:
        backend = PTDImageBackend(root)
        train_txt = backend.read_optional_text("train_metafile.txt")
        val_txt = backend.read_optional_text("val_metafile.txt")
        if train_txt is not None or val_txt is not None:
            if train_txt is None or val_txt is None:
                raise RuntimeError(
                    f"PTD split under {root} is incomplete; expected both train_metafile.txt and val_metafile.txt."
                )
            by_rel = {e.rel_path: e for e in arr}
            train = [by_rel[ln] for ln in train_txt.splitlines() if ln.strip() and ln.strip() in by_rel]
            val = [by_rel[ln] for ln in val_txt.splitlines() if ln.strip() and ln.strip() in by_rel]
            if not train or not val:
                raise RuntimeError(
                    f"Explicit PTD split under {root} produced an empty partition "
                    f"(train={len(train)} val={len(val)})."
                )
            return PTDSplit(train=train, val=val)

    train: list[PTDEntry] = []
    val: list[PTDEntry] = []
    val_fraction = float(np.clip(val_fraction, 0.01, 0.50))
    mod = 10_000
    thr = int(round(val_fraction * mod))

    for e in arr:
        key = f"{split_seed}|{e.rel_path}".encode("utf-8")
        h = int(hashlib.sha1(key).hexdigest()[:12], 16) % mod
        if h < thr:
            val.append(e)
        else:
            train.append(e)

    if not train or not val:
        # deterministic fallback if split is degenerate
        arr.sort(key=lambda x: x.rel_path)
        n_val = max(1, int(round(len(arr) * val_fraction)))
        val = arr[:n_val]
        train = arr[n_val:]
    return PTDSplit(train=train, val=val)


def group_entries_by_class(entries: Iterable[PTDEntry]) -> dict[int, list[PTDEntry]]:
    out: dict[int, list[PTDEntry]] = {}
    for e in entries:
        out.setdefault(int(e.class_id), []).append(e)
    return out


class PTDTextureDataset(Dataset):
    def __init__(
        self,
        *,
        backend: PTDImageBackend,
        entries: list[PTDEntry],
        transform: transforms.Compose | None = None,
        max_images: int | None = None,
    ):
        self.backend = backend
        self.entries = list(entries)
        if max_images is not None:
            self.entries = self.entries[: int(max_images)]
        self.transform = transform
        if not self.entries:
            raise ValueError("PTDTextureDataset has zero entries.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        rgb = self.backend.read_rgb(e.rel_path)
        img = Image.fromarray(rgb)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(e.class_id)


def build_ptd_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.08)],
                    p=0.75,
                ),
                transforms.RandomGrayscale(p=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.10)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
