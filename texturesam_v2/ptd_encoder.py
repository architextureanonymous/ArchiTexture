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
from torchvision.models import ConvNeXt_Base_Weights, ConvNeXt_Small_Weights, ConvNeXt_Tiny_Weights
from torchvision.models import convnext_base, convnext_small, convnext_tiny

from .ptd_data import PTDImageBackend, PTDTextureDataset, build_ptd_transforms, load_ptd_entries, split_ptd_entries


def _resolve_device(device_name: str) -> torch.device:
    use_cuda = str(device_name).startswith("cuda") and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    p = logits.argmax(dim=1)
    return float((p == targets).float().mean().item())


def _is_timm_backbone(backbone: str) -> bool:
    return str(backbone).startswith("timm:")


def _build_backbone(backbone: str, pretrained: bool) -> tuple[str, nn.Module, nn.Module | None, int]:
    name = str(backbone).strip()
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = convnext_tiny(weights=weights)
        in_dim = int(base.classifier[-1].in_features) if hasattr(base.classifier[-1], "in_features") else 768
        return "convnext", base.features, base.avgpool, in_dim
    if name == "convnext_small":
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        base = convnext_small(weights=weights)
        in_dim = int(base.classifier[-1].in_features) if hasattr(base.classifier[-1], "in_features") else 768
        return "convnext", base.features, base.avgpool, in_dim
    if name == "convnext_base":
        weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        base = convnext_base(weights=weights)
        in_dim = int(base.classifier[-1].in_features) if hasattr(base.classifier[-1], "in_features") else 1024
        return "convnext", base.features, base.avgpool, in_dim
    if _is_timm_backbone(name):
        timm_name = name.split(":", 1)[1]
        try:
            import timm
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError("Backbone requires timm, but timm is not available.") from exc
        # Keep compatibility across timm versions: some models/versions do not expose `global_pool`.
        try:
            base = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        except TypeError:
            base = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        in_dim = int(getattr(base, "num_features", 768))
        return "timm", base, None, in_dim
    raise ValueError(f"Unsupported PTD backbone: {name}")


class PTDBackboneEncoder(nn.Module):
    def __init__(self, *, num_classes: int, emb_dim: int = 384, backbone: str = "convnext_tiny", pretrained: bool = False):
        super().__init__()
        self.backbone_name = str(backbone)
        kind, feat_or_backbone, avgpool, in_dim = _build_backbone(backbone=self.backbone_name, pretrained=bool(pretrained))
        self.backbone_kind = kind
        if kind == "convnext":
            # Keep field names (`features`, `avgpool`) for checkpoint backward compatibility.
            self.features = feat_or_backbone
            assert avgpool is not None
            self.avgpool = avgpool
            self.backbone = None
        else:
            self.backbone = feat_or_backbone
            self.features = None
            self.avgpool = None
        self.norm = nn.LayerNorm(in_dim)
        self.embedding = nn.Linear(in_dim, int(emb_dim))
        self.classifier = nn.Linear(int(emb_dim), int(num_classes))

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_kind == "convnext":
            assert self.features is not None
            assert self.avgpool is not None
            h = self.features(x)
            h = self.avgpool(h)
        else:
            assert self.backbone is not None
            h = self.backbone(x)
        if h.ndim > 2:
            h = torch.flatten(h, 1)
        return h

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self._forward_features(x)
        h = self.norm(h)
        emb = self.embedding(h)
        emb = F.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb)

        if return_embedding and return_logits:
            return logits, emb
        if return_embedding:
            return emb
        return logits


@dataclass(frozen=True)
class PTDTrainConfig:
    data_root: Path
    out_ckpt: Path
    epochs: int = 2
    batch_size: int = 48
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    image_size: int = 192
    emb_dim: int = 384
    backbone: str = "convnext_tiny"
    pretrained: bool = False
    supcon_weight: float = 0.0
    supcon_temperature: float = 0.07
    seed: int = 1337
    device: str = "cuda"
    val_fraction: float = 0.10
    max_train_images: int | None = None
    max_val_images: int | None = 20000


@dataclass(frozen=True)
class PTDEncoderConfig:
    checkpoint: Path
    image_size: int = 192
    device: str = "cuda"
    use_ring_context: bool = False
    ring_dilation: int = 9
    ring_min_pixels: int = 24


def _supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if emb.ndim != 2:
        raise ValueError("Expected emb shape [N, D]")
    n = int(emb.shape[0])
    if n <= 1:
        return emb.new_tensor(0.0)

    t = max(float(temperature), 1e-6)
    sim = torch.matmul(emb, emb.t()) / t
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(n, device=emb.device, dtype=torch.bool)
    same = labels.view(-1, 1).eq(labels.view(1, -1))
    pos = torch.logical_and(same, ~eye)

    exp_sim = torch.exp(sim) * (~eye).float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return emb.new_tensor(0.0)

    loss_i = -(log_prob * pos.float()).sum(dim=1) / torch.clamp(pos_count.float(), min=1.0)
    return loss_i[valid].mean()


def train_ptd_encoder(cfg: PTDTrainConfig) -> dict[str, float | int]:
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    backend = PTDImageBackend(cfg.data_root)

    class_names, entries = load_ptd_entries(cfg.data_root)
    split = split_ptd_entries(entries, val_fraction=cfg.val_fraction, split_seed=cfg.seed, root=cfg.data_root)

    train_ds = PTDTextureDataset(
        backend=backend,
        entries=split.train,
        transform=build_ptd_transforms(cfg.image_size, train=True),
        max_images=cfg.max_train_images,
    )
    val_ds = PTDTextureDataset(
        backend=backend,
        entries=split.val,
        transform=build_ptd_transforms(cfg.image_size, train=False),
        max_images=cfg.max_val_images,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = PTDBackboneEncoder(
        num_classes=len(class_names),
        emb_dim=cfg.emb_dim,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(cfg.epochs), 1))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        tr_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, emb = model(xb, return_embedding=True, return_logits=True)
                ce = F.cross_entropy(logits, yb)
                if float(cfg.supcon_weight) > 0.0:
                    scl = _supervised_contrastive_loss(emb, yb, temperature=cfg.supcon_temperature)
                    loss = ce + float(cfg.supcon_weight) * scl
                else:
                    loss = ce

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            b = xb.size(0)
            tr_loss += float(loss.item()) * b
            tr_acc += _top1(logits.detach(), yb) * b
            tr_n += int(b)

        sched.step()

        model.eval()
        va_acc = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(xb)
                b = xb.size(0)
                va_acc += _top1(logits, yb) * b
                va_n += int(b)

        tr_loss_m = tr_loss / max(tr_n, 1)
        tr_acc_m = tr_acc / max(tr_n, 1)
        va_acc_m = va_acc / max(va_n, 1)

        if va_acc_m > best_val:
            best_val = float(va_acc_m)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"[PTD] backbone={cfg.backbone} epoch={ep:02d}/{cfg.epochs} "
            f"train_loss={tr_loss_m:.4f} train_acc={tr_acc_m:.4f} val_acc={va_acc_m:.4f}"
        )

    cfg.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "arch": str(cfg.backbone),
        "pretrained": bool(cfg.pretrained),
        "emb_dim": int(cfg.emb_dim),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "image_size": int(cfg.image_size),
        "best_val_acc": float(best_val),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "supcon_weight": float(cfg.supcon_weight),
        "supcon_temperature": float(cfg.supcon_temperature),
    }
    torch.save(ckpt, cfg.out_ckpt)

    return {
        "best_val_acc": float(best_val),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "num_classes": int(len(class_names)),
        "backbone": str(cfg.backbone),
        "pretrained": int(bool(cfg.pretrained)),
    }


class PTDTextureEncoder:
    def __init__(self, cfg: PTDEncoderConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.use_ring_context = bool(cfg.use_ring_context)
        self.ring_dilation = int(max(3, cfg.ring_dilation))
        if self.ring_dilation % 2 == 0:
            self.ring_dilation += 1
        self.ring_min_pixels = int(max(1, cfg.ring_min_pixels))

        ckpt = torch.load(cfg.checkpoint, map_location="cpu")
        emb_dim = int(ckpt.get("emb_dim", 384))
        num_classes = int(ckpt.get("num_classes", 56))
        self.image_size = int(ckpt.get("image_size", cfg.image_size))
        backbone = str(ckpt.get("arch", "convnext_tiny"))

        self.model = PTDBackboneEncoder(
            num_classes=num_classes,
            emb_dim=emb_dim,
            backbone=backbone,
            pretrained=False,
        )
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray, margin: int = 4) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x0 = max(0, int(xs.min()) - margin)
        x1 = int(xs.max()) + margin + 1
        y0 = max(0, int(ys.min()) - margin)
        y1 = int(ys.max()) + margin + 1
        return x0, y0, x1, y1

    @staticmethod
    def _render_focus_crop(crop_rgb: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        if crop_rgb.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        out = crop_rgb.astype(np.float32)
        outside = keep_mask == 0
        if np.any(outside):
            out[outside] = 0.20 * out[outside] + 0.80 * 127.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def _masked_crops(self, image_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        m = (mask > 0).astype(np.uint8)
        bbox = self._bbox_from_mask(m)
        if bbox is None:
            z = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return z, z if self.use_ring_context else None

        x0, y0, x1, y1 = bbox
        x1 = min(x1, image_rgb.shape[1])
        y1 = min(y1, image_rgb.shape[0])
        crop = image_rgb[y0:y1, x0:x1].copy()
        cm = m[y0:y1, x0:x1]
        if crop.size == 0:
            z = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return z, z if self.use_ring_context else None

        region = self._render_focus_crop(crop, cm)
        region = cv2.resize(region, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        if not self.use_ring_context:
            return region, None

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ring_dilation, self.ring_dilation))
        dil = cv2.dilate(cm, k, iterations=1)
        ring = np.logical_and(dil > 0, cm == 0).astype(np.uint8)
        if int(ring.sum()) < self.ring_min_pixels:
            ring = (cm == 0).astype(np.uint8)
        ring_crop = self._render_focus_crop(crop, ring)
        ring_crop = cv2.resize(ring_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return region, ring_crop

    def _encode_batch(self, tensors: list[torch.Tensor]) -> np.ndarray:
        if not tensors:
            return np.zeros((0, 0), dtype=np.float32)
        xb = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                emb = self.model(xb, return_embedding=True).detach().cpu().numpy().astype(np.float32)
        return emb

    def encode_regions(self, image_rgb: np.ndarray, masks: Iterable[np.ndarray]) -> list[np.ndarray]:
        region_tensors: list[torch.Tensor] = []
        ring_tensors: list[torch.Tensor] = []
        use_ring = self.use_ring_context

        for m in masks:
            region, ring = self._masked_crops(image_rgb, m)
            region_tensors.append(self.t(region))
            if use_ring:
                assert ring is not None
                ring_tensors.append(self.t(ring))

        if not region_tensors:
            return []

        emb_region = self._encode_batch(region_tensors)
        if not use_ring:
            return [emb_region[i] for i in range(emb_region.shape[0])]

        emb_ring = self._encode_batch(ring_tensors)
        # Concatenate region identity with context contrast for stronger compatibility cues.
        desc = np.concatenate([emb_region, emb_region - emb_ring], axis=1).astype(np.float32)
        norm = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8
        desc = desc / norm
        return [desc[i] for i in range(desc.shape[0])]
