from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DTD


class SmallTextureCNN(nn.Module):
    def __init__(self, num_classes: int = 47, emb_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        emb = self.embedding(h)
        emb = F.normalize(emb, p=2, dim=1)
        if return_embedding:
            return emb
        return self.classifier(emb)


@dataclass(frozen=True)
class DTDTrainConfig:
    data_root: Path
    out_ckpt: Path
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 8
    image_size: int = 96
    seed: int = 1337
    device: str = "cuda"


@dataclass(frozen=True)
class DTDEncoderConfig:
    checkpoint: Path
    image_size: int = 96
    device: str = "cuda"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dtd_encoder(cfg: DTDTrainConfig) -> dict[str, float]:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    t_train = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    t_eval = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = DTD(root=str(cfg.data_root), split="train", download=True, transform=t_train)
    val_ds = DTD(root=str(cfg.data_root), split="val", download=True, transform=t_eval)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SmallTextureCNN(num_classes=47, emb_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0
        correct = 0
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(xb.size(0))

        sched.step()

        model.eval()
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                val_correct += int((pred == yb).sum().item())
                val_total += int(xb.size(0))

        val_acc = float(val_correct / max(val_total, 1))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"[DTD] epoch={epoch:02d}/{cfg.epochs} "
            f"train_loss={train_loss/max(total,1):.4f} train_acc={correct/max(total,1):.4f} "
            f"val_acc={val_acc:.4f}"
        )

    cfg.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "image_size": cfg.image_size,
        "emb_dim": 128,
        "num_classes": 47,
        "best_val_acc": best_val_acc,
    }
    torch.save(ckpt, cfg.out_ckpt)

    return {"best_val_acc": float(best_val_acc)}


class DTDTextureEncoder:
    def __init__(self, cfg: DTDEncoderConfig):
        self.cfg = cfg
        device = torch.device(cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self.device = device
        ckpt = torch.load(cfg.checkpoint, map_location="cpu")

        emb_dim = int(ckpt.get("emb_dim", 128))
        num_classes = int(ckpt.get("num_classes", 47))
        self.model = SmallTextureCNN(num_classes=num_classes, emb_dim=emb_dim)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.image_size = int(ckpt.get("image_size", cfg.image_size))
        self.t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _masked_crop(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        x0 = max(0, int(xs.min()) - 2)
        x1 = min(image_rgb.shape[1], int(xs.max()) + 3)
        y0 = max(0, int(ys.min()) - 2)
        y1 = min(image_rgb.shape[0], int(ys.max()) + 3)

        crop = image_rgb[y0:y1, x0:x1].copy()
        cm = m[y0:y1, x0:x1]
        crop[cm == 0] = 0
        crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def encode_regions(self, image_rgb: np.ndarray, masks: Iterable[np.ndarray]) -> list[np.ndarray]:
        tensors = []
        for m in masks:
            crop = self._masked_crop(image_rgb, m)
            tensors.append(self.t(crop))

        if not tensors:
            return []

        xb = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            emb = self.model(xb, return_embedding=True).detach().cpu().numpy().astype(np.float32)
        return [emb[i] for i in range(emb.shape[0])]
