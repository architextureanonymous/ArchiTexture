#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.features import compute_texture_feature_map, mean_feature, region_descriptor, region_variance
from texturesam_v2.io_utils import (
    ensure_binary,
    ensure_binary_gt,
    infer_rwtd_dirs,
    list_rwtd_images,
    read_image_rgb,
    read_mask_raw,
    write_binary_mask,
)
from texturesam_v2.metrics import rwtd_invariant_metrics
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig
from texturesam_v2.ptd_data import PTDImageBackend, group_entries_by_class, load_ptd_entries, split_ptd_entries
from texturesam_v2.ptd_encoder import PTDEncoderConfig, PTDTextureEncoder
from texturesam_v2.ptd_v4_set import PTDV4SetScorer
from texturesam_v2.ptd_v6_coverage import PTDV6CoverageScorer
from texturesam_v2.ptd_v8_partition import (
    PTDV8PartitionScorer,
    PTDV8PartitionTrainConfig,
    _compose_synthetic_target_union,
    _fragment_mask,
    _make_synthetic_proposals_union,
    _proposal_consensus_masks,
    _proposal_union,
)

EPS = 1e-8


@dataclass(frozen=True)
class _ValGroup:
    start: int
    end: int
    core_iou: float
    oracle_iou: float


@dataclass(frozen=True)
class _ChoiceGroup:
    start: int
    end: int
    target: int
    option_ious: tuple[float, ...]
    option_utils: tuple[float, ...]


class _TorchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden1: int = 256, hidden2: int = 128, dropout: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x).squeeze(-1)


class TorchMLPBinaryClassifier:
    def __init__(
        self,
        *,
        device: str = "cuda",
        seed: int = 1337,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.10,
        batch_size: int = 256,
        max_epochs: int = 80,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
    ):
        self.device = device
        self.seed = int(seed)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.model: _TorchMLP | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.classes_ = np.array([0, 1], dtype=np.int32)

    def _to_device(self) -> torch.device:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.device)
        return torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = X.mean(axis=0, keepdims=True).astype(np.float32)
        self.std_ = X.std(axis=0, keepdims=True).astype(np.float32)
        self.std_[self.std_ < 1e-6] = 1.0
        Xn = (X - self.mean_) / self.std_

        rs = np.random.RandomState(self.seed)
        n = Xn.shape[0]
        idx = np.arange(n)
        rs.shuffle(idx)
        n_val = max(16, int(round(0.15 * n))) if n >= 64 else max(4, int(round(0.20 * n)))
        n_val = min(max(1, n_val), max(1, n - 1))
        va_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        if tr_idx.size == 0:
            tr_idx = va_idx
            va_idx = idx[: min(1, len(idx))]

        Xtr = torch.from_numpy(Xn[tr_idx])
        ytr = torch.from_numpy(y[tr_idx])
        Xva = torch.from_numpy(Xn[va_idx])
        yva = torch.from_numpy(y[va_idx])

        device = self._to_device()
        torch.manual_seed(self.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        model = _TorchMLP(X.shape[1], self.hidden1, self.hidden2, self.dropout).to(device)
        ds = TensorDataset(Xtr, ytr)
        loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True, drop_last=False)

        pos = float(ytr.sum().item())
        neg = float(len(ytr) - pos)
        pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], device=device, dtype=torch.float32)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state = None
        best_score = -1.0
        patience_left = self.patience
        Xva_d = Xva.to(device)
        yva_np = yva.numpy().astype(np.int32)

        for _ in range(self.max_epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logit = model(xb)
                loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                prob_va = torch.sigmoid(model(Xva_d)).detach().cpu().numpy().astype(np.float32)
            try:
                auc = float(roc_auc_score(yva_np, prob_va))
            except Exception:
                auc = 0.0
            pred_va = (prob_va >= 0.5).astype(np.int32)
            try:
                f1 = float(f1_score(yva_np, pred_va))
            except Exception:
                f1 = 0.0
            score = auc + 0.1 * f1
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model = model.cpu()
        self.model = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("TorchMLPBinaryClassifier must be fit before predict_proba.")
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.mean_) / self.std_
        with torch.no_grad():
            prob = torch.sigmoid(self.model(torch.from_numpy(Xn))).cpu().numpy().astype(np.float32)
        return np.stack([1.0 - prob, prob], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(np.int32)


class TorchMLPGroupChooser:
    def __init__(
        self,
        *,
        device: str = "cuda",
        seed: int = 1337,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.10,
        batch_size: int = 32,
        max_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 12,
        utility_temperature: float = 0.10,
        hard_target_weight: float = 0.35,
    ):
        self.device = device
        self.seed = int(seed)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.utility_temperature = float(utility_temperature)
        self.hard_target_weight = float(hard_target_weight)
        self.model: _TorchMLP | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _to_device(self) -> torch.device:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.device)
        return torch.device("cpu")

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TorchMLPGroupChooser must be fit before normalization.")
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.std_

    def fit(
        self,
        X_tr: np.ndarray,
        groups_tr: list[_ChoiceGroup],
        X_va: np.ndarray,
        groups_va: list[_ChoiceGroup],
    ) -> dict[str, float]:
        if not groups_tr or not groups_va:
            raise RuntimeError("TorchMLPGroupChooser requires non-empty train and val groups.")
        X_tr = np.asarray(X_tr, dtype=np.float32)
        X_va = np.asarray(X_va, dtype=np.float32)
        self.mean_ = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
        self.std_ = X_tr.std(axis=0, keepdims=True).astype(np.float32)
        self.std_[self.std_ < 1e-6] = 1.0
        Xtr_n = self._normalize(X_tr)
        Xva_n = self._normalize(X_va)

        device = self._to_device()
        torch.manual_seed(self.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        model = _TorchMLP(X_tr.shape[1], self.hidden1, self.hidden2, self.dropout).to(device)
        Xtr_d = torch.from_numpy(Xtr_n).to(device)
        Xva_d = torch.from_numpy(Xva_n).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        rs = np.random.RandomState(self.seed)
        best_state = None
        best_score = float("-inf")
        best_metrics = {
            "choice_val_acc": 0.0,
            "choice_val_selected_iou_mean": 0.0,
            "choice_val_selected_utility_mean": 0.0,
            "choice_val_switch_rate": 0.0,
            "choice_val_oracle_utility_mean": 0.0,
            "choice_target_switch_rate_train": 0.0,
            "choice_target_switch_rate_val": 0.0,
        }
        patience_left = self.patience
        target_train = np.asarray([int(g.target > 0) for g in groups_tr], dtype=np.float32)
        keep_rate = float(np.mean(1.0 - target_train)) if len(target_train) else 0.5
        switch_rate = float(np.mean(target_train)) if len(target_train) else 0.5
        keep_weight = 0.5 / max(keep_rate, 1e-3)
        switch_weight = 0.5 / max(switch_rate, 1e-3)
        target_val = np.asarray([int(g.target > 0) for g in groups_va], dtype=np.float32)

        def _group_target_probs(g: _ChoiceGroup, dev: torch.device) -> torch.Tensor:
            utils = torch.tensor(g.option_utils, dtype=torch.float32, device=dev)
            temp = max(self.utility_temperature, 1e-4)
            utils = (utils - utils.max()) / temp
            return torch.softmax(utils, dim=0)

        def _eval(groups: list[_ChoiceGroup], Xd: torch.Tensor) -> tuple[float, dict[str, float]]:
            model.eval()
            selected_ious: list[float] = []
            selected_utils: list[float] = []
            oracle_utils: list[float] = []
            switches = 0
            correct = 0
            with torch.no_grad():
                for g in groups:
                    logits = model(Xd[g.start:g.end])
                    pred = int(torch.argmax(logits).item())
                    correct += int(pred == int(g.target))
                    selected_ious.append(float(g.option_ious[pred]))
                    selected_utils.append(float(g.option_utils[pred]))
                    oracle_utils.append(float(max(g.option_utils)))
                    switches += int(pred > 0)
            sel_util = float(np.mean(selected_utils)) if selected_utils else 0.0
            metrics = {
                "choice_val_acc": float(correct / max(len(groups), 1)),
                "choice_val_selected_iou_mean": float(np.mean(selected_ious)) if selected_ious else 0.0,
                "choice_val_selected_utility_mean": sel_util,
                "choice_val_switch_rate": float(switches / max(len(groups), 1)),
                "choice_val_oracle_utility_mean": float(np.mean(oracle_utils)) if oracle_utils else 0.0,
                "choice_target_switch_rate_train": float(switch_rate),
                "choice_target_switch_rate_val": float(np.mean(target_val)) if len(target_val) else 0.0,
            }
            return sel_util, metrics

        train_order = np.arange(len(groups_tr))
        for _ in range(self.max_epochs):
            rs.shuffle(train_order)
            model.train()
            for start in range(0, len(train_order), self.batch_size):
                batch_idx = train_order[start : start + self.batch_size]
                loss = torch.zeros((), dtype=torch.float32, device=device)
                for idx in batch_idx:
                    g = groups_tr[int(idx)]
                    logits = model(Xtr_d[g.start:g.end])
                    target_probs = _group_target_probs(g, device)
                    loss_soft = -(target_probs * torch.log_softmax(logits, dim=0)).sum()
                    target_idx = torch.tensor([int(g.target)], dtype=torch.long, device=device)
                    loss_hard = F.cross_entropy(logits[None, :], target_idx)
                    group_weight = keep_weight if int(g.target) == 0 else switch_weight
                    loss = loss + float(group_weight) * (loss_soft + self.hard_target_weight * loss_hard)
                loss = loss / max(len(batch_idx), 1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            val_score, val_metrics = _eval(groups_va, Xva_d)
            if val_score > best_score:
                best_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = dict(val_metrics)
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model.cpu()
        return best_metrics

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TorchMLPGroupChooser must be fit before predict_scores.")
        Xn = self._normalize(X)
        with torch.no_grad():
            scores = self.model(torch.from_numpy(Xn)).cpu().numpy().astype(np.float32)
        return scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TextureSAM-2 acute learned rescue: PTD-only model that predicts whether a dense "
            "rescue candidate will improve the conservative core mask."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--dense-prompt-masks-root", type=Path, required=True)
    p.add_argument("--v9-masks-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--image-ids-file", type=Path, default=None)

    p.add_argument("--ptd-root", type=Path, default=Path("external/data/ptd"))
    p.add_argument("--ptd-checkpoint", type=Path, default=ROOT / "artifacts/ptd_convnext_tiny.pt")
    p.add_argument("--ptd-v4-bundle", type=Path, default=ROOT / "artifacts/ptd_v4_set_bundle_sanitized.pkl")
    p.add_argument("--ptd-v6-bundle", type=Path, default=ROOT / "artifacts/ptd_v6_coverage_bundle_sanitized.pkl")
    p.add_argument("--ptd-v8-bundle", type=Path, default=ROOT / "artifacts/ptd_v8_partition_bundle_sanitized_cuda.pkl")
    p.add_argument("--ptd-device", type=str, default="cuda")
    p.add_argument("--core-surrogate", type=str, default="v9stack", choices=["v8", "v9stack"])

    p.add_argument("--bundle-path", type=Path, default=ROOT / "artifacts/ptd_acute_rescue_bundle.pkl")
    p.add_argument("--bundle-metrics-json", type=Path, default=ROOT / "artifacts/ptd_acute_rescue_metrics.json")
    p.add_argument("--train-if-missing", action="store_true")
    p.add_argument("--retrain", action="store_true")

    p.add_argument("--synthetic-samples", type=int, default=960)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gain-positive-margin", type=float, default=0.03)
    p.add_argument(
        "--cls-model",
        type=str,
        default="rf_balanced",
        choices=["hgb", "rf_balanced", "logreg_balanced", "torch_mlp"],
        help="Classifier used for learned switch decisions.",
    )
    p.add_argument(
        "--decision-architecture",
        type=str,
        default="scene_gate",
        choices=["candidate_cls", "scene_gate", "candidate_veto_gate", "group_argmax", "triage_top2"],
        help=(
            "candidate_cls: mark any dense candidate predicted positive and pick the highest-gain one. "
            "scene_gate: pick the top regressed dense candidate, then run a learned switch/no-switch gate on that scene. "
            "candidate_veto_gate: use candidate_cls to select a candidate, then apply a learned veto gate to that selected candidate. "
            "group_argmax: score keep-core and all dense rescue options jointly with a neural chooser and pick the highest-scoring option. "
            "triage_top2: PTD-only 3-way policy that chooses among {keep core, dense rank-1, dense rank-2}."
        ),
    )
    p.add_argument(
        "--supervision-source",
        type=str,
        default="gt",
        choices=["gt", "v11_teacher"],
        help="How PTD synthetic positives and ranking targets are defined.",
    )
    p.add_argument("--teacher-cov9-low", type=float, default=0.30)
    p.add_argument("--teacher-cov-gain-min", type=float, default=0.10)
    p.add_argument("--teacher-prec-best-min", type=float, default=0.25)
    p.add_argument("--teacher-score-margin", type=float, default=0.08)
    p.add_argument("--teacher-area-ratio-min", type=float, default=0.05)
    p.add_argument("--teacher-area-ratio-max", type=float, default=2.2)
    p.add_argument("--teacher-min-iou-gain", type=float, default=0.0)
    p.add_argument("--teacher-min-ari-gain", type=float, default=0.0)
    p.add_argument(
        "--train-core-mode",
        type=str,
        default="v9stack",
        choices=["v9stack", "repair_mix"],
        help="How PTD training cores are constructed. repair_mix injects explicit under-coverage cores and hard rescue decoys.",
    )
    p.add_argument("--repair-core-prob", type=float, default=0.75)
    p.add_argument(
        "--label-mode",
        type=str,
        default="oracle_winner",
        choices=["gain_margin", "oracle_winner", "oracle_band", "oracle_safe_winner"],
        help=(
            "How PTD synthetic rescue positives are labeled. "
            "'gain_margin' marks any candidate above the gain margin. "
            "'oracle_winner' marks only the single best dense candidate if it clears the margin. "
            "'oracle_band' marks candidates within oracle slack of the best dense candidate if the best clears the margin. "
            "'oracle_safe_winner' marks only the best dense candidate when it improves both IoU and ARI over core."
        ),
    )
    p.add_argument(
        "--oracle-slack",
        type=float,
        default=0.01,
        help="Only used by --label-mode oracle_band.",
    )
    p.add_argument("--safe-min-ari-gain", type=float, default=0.0)
    p.add_argument("--ari-gain-weight", type=float, default=0.25)
    p.add_argument("--choice-risk-penalty-ari", type=float, default=0.0)
    p.add_argument("--choice-risk-penalty-iou", type=float, default=0.0)
    p.add_argument("--risk-margin-ari", type=float, default=0.0)
    p.add_argument("--risk-margin-iou", type=float, default=0.0)

    p.add_argument("--synthetic-image-size", type=int, default=256)
    p.add_argument("--min-regions", type=int, default=3)
    p.add_argument("--max-regions", type=int, default=6)
    p.add_argument("--min-fg-frags", type=int, default=4)
    p.add_argument("--max-fg-frags", type=int, default=10)
    p.add_argument("--min-area", type=int, default=24)
    p.add_argument("--max-class-pool", type=int, default=4)
    p.add_argument("--multi-target-prob", type=float, default=0.70)
    p.add_argument("--max-proposals-per-sample", type=int, default=18)
    p.add_argument(
        "--wrong-side-negatives-per-sample",
        type=int,
        default=3,
        help="How many coherent wrong-side negatives to inject per synthetic sample in repair_mix mode.",
    )
    p.add_argument(
        "--wrong-side-area-ratio-min",
        type=float,
        default=0.35,
        help="Minimum candidate/core area ratio for coherent wrong-side negatives.",
    )
    p.add_argument(
        "--wrong-side-area-ratio-max",
        type=float,
        default=2.8,
        help="Maximum candidate/core area ratio for coherent wrong-side negatives.",
    )
    p.add_argument(
        "--wrong-side-dilate-ksize",
        type=int,
        default=9,
        help="Dilation kernel size used when constructing coherent wrong-side negatives.",
    )
    p.add_argument(
        "--decision-margin-min",
        type=float,
        default=0.0,
        help="Minimum top-1 minus top-2 decision margin required before switching to a dense candidate.",
    )
    p.add_argument(
        "--abstain-margin",
        type=float,
        default=-1.0,
        help="If >= 0 and decision margin is below this threshold, mark case as ambiguous and keep core.",
    )
    p.add_argument(
        "--plausible-gap",
        type=float,
        default=0.02,
        help="Dense candidates within this score gap of the top dense score are treated as plausible alternatives.",
    )
    p.add_argument(
        "--support-prior-weight",
        type=float,
        default=0.0,
        help="Weight for proposal-support prior added to predicted dense gain before ranking.",
    )
    p.add_argument(
        "--topk-output",
        type=int,
        default=0,
        help="If > 0, export top-k dense candidate masks per image under diagnostics/topk.",
    )
    p.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Write per-image JSON diagnostics with candidate-level scores and metrics.",
    )
    p.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance for binary boundary precision/recall/F1/IoU diagnostics.",
    )
    p.add_argument(
        "--save-audit-artifacts",
        action="store_true",
        help=(
            "Export per-image audit folders with exact scored masks, candidate masks, "
            "proposal support maps, and binary diff overlays."
        ),
    )
    p.add_argument(
        "--audit-max-candidates",
        type=int,
        default=0,
        help="Maximum dense candidates to export per image when --save-audit-artifacts is enabled (0 = all).",
    )
    return p.parse_args()


def _load_image_id_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keep: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        keep.add(str(int(s)))
    return keep or None


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    return ensure_binary(read_mask_raw(path))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    uni = float(np.logical_or(aa, bb).sum())
    return 1.0 if uni <= 0 else inter / uni


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.size == 0:
        return m.astype(bool)
    k = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    return grad > 0


def boundary_metrics(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> tuple[float, float, float]:
    ep = _boundary_mask(pred)
    eg = _boundary_mask(gt)
    npix = int(ep.sum())
    ngt = int(eg.sum())
    if npix == 0 and ngt == 0:
        return 1.0, 1.0, 1.0
    if npix == 0 or ngt == 0:
        return 0.0, 0.0, 0.0
    r = max(int(tol), 0)
    if r > 0:
        k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        ep_d = cv2.dilate(ep.astype(np.uint8), k, iterations=1) > 0
        eg_d = cv2.dilate(eg.astype(np.uint8), k, iterations=1) > 0
    else:
        ep_d = ep
        eg_d = eg
    tp_p = int(np.logical_and(ep, eg_d).sum())
    tp_g = int(np.logical_and(eg, ep_d).sum())
    prec = float(tp_p / max(npix, 1))
    rec = float(tp_g / max(ngt, 1))
    if (prec + rec) <= EPS:
        f1 = 0.0
    else:
        f1 = float((2.0 * prec * rec) / (prec + rec))
    b_inter = float(np.logical_and(ep_d, eg_d).sum())
    b_union = float(np.logical_or(ep_d, eg_d).sum())
    biou = 1.0 if b_union <= 0 else float(b_inter / b_union)
    return f1, biou, prec


def _safe_margin(top: float, second: float) -> float:
    return float(top - second)


def _top2_indices(values: np.ndarray) -> tuple[int, int]:
    if values.size == 0:
        return -1, -1
    if values.size == 1:
        return 0, 0
    order = np.argsort(values)
    return int(order[-1]), int(order[-2])


def _normalized_support_prior(proposals: list[np.ndarray], dense_cands: list[np.ndarray]) -> np.ndarray:
    if not proposals or not dense_cands:
        return np.zeros((len(dense_cands),), dtype=np.float32)
    stack = np.stack([(p > 0).astype(np.float32) for p in proposals], axis=0)
    freq = stack.mean(axis=0)
    vals: list[float] = []
    for dm in dense_cands:
        m = dm > 0
        if int(m.sum()) <= 0:
            vals.append(0.0)
        else:
            vals.append(float(freq[m].mean()))
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return arr
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd < 1e-6:
        return arr * 0.0
    return ((arr - mu) / sd).astype(np.float32)


def _mask_change_ratio(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    return float(np.logical_xor(aa, bb).sum() / max(aa.size, 1))


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _write_gray_float(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        out = np.zeros((1, 1), dtype=np.uint8)
    else:
        lo = float(np.min(a))
        hi = float(np.max(a))
        if (hi - lo) < 1e-8:
            out = np.zeros_like(a, dtype=np.uint8)
        else:
            out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
            out = (255.0 * out).astype(np.uint8)
    cv2.imwrite(str(path), out)


def _write_support_heatmap(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        heat = np.zeros((1, 1, 3), dtype=np.uint8)
    else:
        scaled = np.clip(255.0 * a, 0.0, 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(path), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))


def _candidate_stack_density(dense_cands: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not dense_cands:
        return np.zeros(shape, dtype=np.float32)
    stack = np.stack([(m > 0).astype(np.float32) for m in dense_cands], axis=0)
    return np.clip(stack.mean(axis=0), 0.0, 1.0).astype(np.float32)


def _write_candidate_stack_overlay(path: Path, image_rgb: np.ndarray, density: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if density.size == 0:
        out = image_rgb.copy()
        cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return
    scaled = np.clip(255.0 * density, 0.0, 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    out = image_rgb.astype(np.float32).copy()
    alpha_map = (density[..., None] * 0.72).astype(np.float32)
    out = (1.0 - alpha_map) * out + alpha_map * heat.astype(np.float32)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def _binary_diff_overlay_rgb(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = reference > 0
    cmp = candidate > 0
    out = np.zeros((ref.shape[0], ref.shape[1], 3), dtype=np.uint8)
    both = np.logical_and(ref, cmp)
    added = np.logical_and(~ref, cmp)
    removed = np.logical_and(ref, ~cmp)
    out[both] = np.array([180, 180, 180], dtype=np.uint8)
    out[added] = np.array([46, 170, 60], dtype=np.uint8)
    out[removed] = np.array([220, 60, 60], dtype=np.uint8)
    return out


def _write_binary_diff_overlay(path: Path, reference: np.ndarray, candidate: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _binary_diff_overlay_rgb(reference, candidate)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _cov_prec_area(mask: np.ndarray, union: np.ndarray) -> tuple[float, float, float]:
    m = mask > 0
    u = union > 0
    ma = int(m.sum())
    ua = int(u.sum())
    if ua <= 0 or ma <= 0:
        return 0.0, 0.0, 0.0
    inter = float(np.logical_and(m, u).sum())
    cov = float(inter / max(ua, 1))
    prec = float(inter / max(ma, 1))
    area_u = float(ma / max(ua, 1))
    return cov, prec, area_u


def _frag(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    n_cc, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cc_pen = max(0, n_cc - 2)
    inv = 1 - m
    n_h, lbl_h, stats_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    holes = 0
    if n_h > 1:
        h, w = m.shape
        border = set(np.unique(lbl_h[0, :]).tolist())
        border.update(np.unique(lbl_h[h - 1, :]).tolist())
        border.update(np.unique(lbl_h[:, 0]).tolist())
        border.update(np.unique(lbl_h[:, w - 1]).tolist())
        for c in range(1, n_h):
            if c in border:
                continue
            if int(stats_h[c, cv2.CC_STAT_AREA]) > 0:
                holes += 1
    return float(cc_pen + 0.5 * holes)


def _teacher_score_candidate(mask: np.ndarray, proposal_union: np.ndarray, feat_map: np.ndarray) -> tuple[float, float, float, float]:
    cov, prec, area_u = _cov_prec_area(mask, proposal_union)
    inside = mean_feature(feat_map, mask)
    outside = mean_feature(feat_map, 1 - (mask > 0).astype(np.uint8))
    delta = float(np.linalg.norm(inside - outside))
    var = float(region_variance(feat_map, mask))
    frag = float(_frag(mask))
    score = 0.60 * cov + 0.18 * prec + 0.10 * delta - 0.08 * var - 0.10 * frag
    return float(score), float(cov), float(prec), float(area_u)


def _logit(x: np.ndarray) -> np.ndarray:
    q = np.clip(x.astype(np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(q / (1.0 - q))


def _dedupe(masks: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for m in masks:
        b = (m > 0).astype(np.uint8)
        if int(b.sum()) < min_area:
            continue
        k = np.packbits(b, axis=None).tobytes()
        if k in seen:
            continue
        seen.add(k)
        out.append(b)
    return out


def _freq_candidates(proposals: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    if not proposals:
        return []
    stack = np.stack([(p > 0).astype(np.float32) for p in proposals], axis=0)
    freq = stack.mean(axis=0)
    out: list[np.ndarray] = []
    for t in (0.10, 0.16, 0.24, 0.32, 0.42, 0.56, 0.70):
        m = (freq >= t).astype(np.uint8)
        if int(m.sum()) >= min_area:
            out.append(m)
    return _dedupe(out, min_area)


def _topk_unions(proposals: list[np.ndarray], min_area: int, max_k: int = 6) -> list[np.ndarray]:
    if not proposals:
        return []
    n = len(proposals)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a = proposals[i] > 0
        for j in range(i, n):
            b = proposals[j] > 0
            inter = float(np.logical_and(a, b).sum())
            uni = float(np.logical_or(a, b).sum())
            v = 1.0 if uni <= 0 else inter / uni
            mat[i, j] = v
            mat[j, i] = v
    central = mat.mean(axis=1)
    order = np.argsort(central)[::-1].tolist()

    out: list[np.ndarray] = []
    for k in range(2, min(max_k, n) + 1):
        u = np.zeros_like(proposals[0], dtype=np.uint8)
        for idx in order[:k]:
            u = np.logical_or(u > 0, proposals[idx] > 0)
        if int(u.sum()) >= min_area:
            out.append(u.astype(np.uint8))
    return _dedupe(out, min_area)


def _dense_candidate_bank(proposals: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    cands: list[np.ndarray] = []
    cands.extend(_freq_candidates(proposals, min_area=min_area))
    cands.extend(_topk_unions(proposals, min_area=min_area, max_k=6))
    if proposals:
        union = _proposal_union(proposals)
        if int(union.sum()) >= min_area:
            cands.append(union.astype(np.uint8))
    return _dedupe(cands, min_area=min_area)


def _pair_feature(
    *,
    core_mask: np.ndarray,
    dense_mask: np.ndarray,
    core_feat: np.ndarray,
    dense_feat: np.ndarray,
    core_score: float,
    dense_score: float,
    proposal_count: int,
    proposal_union: np.ndarray,
    feat_map: np.ndarray,
) -> np.ndarray:
    c = core_mask > 0
    d = dense_mask > 0
    ca = float(c.sum())
    da = float(d.sum())
    inter = float(np.logical_and(c, d).sum())
    uni = float(np.logical_or(c, d).sum())
    iou_cd = inter / max(uni, 1.0)
    cover_core = inter / max(ca, 1.0)
    cover_dense = inter / max(da, 1.0)
    area_ratio = da / max(ca, 1.0)
    log_area_ratio = math.log(max(area_ratio, 1e-5))
    core_cov_u, core_prec_u, core_area_u = _cov_prec_area(core_mask, proposal_union)
    dense_cov_u, dense_prec_u, dense_area_u = _cov_prec_area(dense_mask, proposal_union)
    core_inside = mean_feature(feat_map, core_mask)
    core_outside = mean_feature(feat_map, 1 - (core_mask > 0).astype(np.uint8))
    dense_inside = mean_feature(feat_map, dense_mask)
    dense_outside = mean_feature(feat_map, 1 - (dense_mask > 0).astype(np.uint8))
    core_delta = float(np.linalg.norm(core_inside - core_outside))
    dense_delta = float(np.linalg.norm(dense_inside - dense_outside))
    core_var = float(region_variance(feat_map, core_mask))
    dense_var = float(region_variance(feat_map, dense_mask))
    core_frag = float(_frag(core_mask))
    dense_frag = float(_frag(dense_mask))
    extra = np.array(
        [
            float(core_score),
            float(dense_score),
            float(dense_score - core_score),
            float(ca / max(c.size, 1)),
            float(da / max(d.size, 1)),
            float(area_ratio),
            float(log_area_ratio),
            float(iou_cd),
            float(cover_core),
            float(cover_dense),
            float(min(proposal_count, 160) / 160.0),
            float(core_cov_u),
            float(core_prec_u),
            float(core_area_u),
            float(dense_cov_u),
            float(dense_prec_u),
            float(dense_area_u),
            float(dense_cov_u - core_cov_u),
            float(dense_prec_u - core_prec_u),
            float(dense_area_u - core_area_u),
            float(core_delta),
            float(dense_delta),
            float(dense_delta - core_delta),
            float(core_var),
            float(dense_var),
            float(dense_var - core_var),
            float(core_frag),
            float(dense_frag),
            float(dense_frag - core_frag),
        ],
        dtype=np.float32,
    )
    return np.concatenate([core_feat, dense_feat, dense_feat - core_feat, extra], axis=0).astype(np.float32)


def _choice_option_feature(
    *,
    core_mask: np.ndarray,
    option_mask: np.ndarray,
    core_feat: np.ndarray,
    option_feat: np.ndarray,
    core_score: float,
    option_score: float,
    proposal_count: int,
    proposal_union: np.ndarray,
    feat_map: np.ndarray,
    is_keep: bool,
) -> np.ndarray:
    base = _pair_feature(
        core_mask=core_mask,
        dense_mask=option_mask,
        core_feat=core_feat,
        dense_feat=option_feat,
        core_score=core_score,
        dense_score=option_score,
        proposal_count=proposal_count,
        proposal_union=proposal_union,
        feat_map=feat_map,
    )
    return np.concatenate(
        [base, np.array([1.0 if is_keep else 0.0], dtype=np.float32)],
        axis=0,
    ).astype(np.float32)


def _augment_choice_option_block(
    base_block: np.ndarray,
    pred_gain: np.ndarray,
    pred_prob: np.ndarray,
    risk_prob: np.ndarray,
) -> np.ndarray:
    n_opt = base_block.shape[0]
    extras = np.zeros((n_opt, 6), dtype=np.float32)
    best_gain = float(np.max(pred_gain)) if pred_gain.size else 0.0
    best_prob = float(np.max(pred_prob)) if pred_prob.size else 0.0
    worst_risk = float(np.max(risk_prob)) if risk_prob.size else 0.0
    extras[0] = np.array([0.0, 0.0, 0.0, best_gain, best_prob, worst_risk], dtype=np.float32)
    for j in range(pred_gain.shape[0]):
        extras[j + 1] = np.array(
            [float(pred_gain[j]), float(pred_prob[j]), float(risk_prob[j]), best_gain, best_prob, worst_risk],
            dtype=np.float32,
        )
    return np.concatenate([base_block.astype(np.float32), extras], axis=1).astype(np.float32)


def _augment_choice_features_with_pair_preds(
    *,
    choice_X: np.ndarray,
    choice_groups: list[_ChoiceGroup],
    pair_groups: list[_ValGroup],
    pred_gain: np.ndarray,
    pred_prob: np.ndarray,
    risk_prob: np.ndarray,
) -> np.ndarray:
    if len(choice_groups) != len(pair_groups):
        raise ValueError("choice_groups and pair_groups must align one-to-one.")
    blocks: list[np.ndarray] = []
    for cg, pg in zip(choice_groups, pair_groups):
        base = choice_X[cg.start:cg.end]
        local_gain = pred_gain[pg.start:pg.end]
        local_prob = pred_prob[pg.start:pg.end]
        local_risk = risk_prob[pg.start:pg.end]
        if base.shape[0] != local_gain.shape[0] + 1:
            raise ValueError("Choice block must have exactly one keep option plus all pair candidates.")
        blocks.append(_augment_choice_option_block(base, local_gain, local_prob, local_risk))
    return np.concatenate(blocks, axis=0).astype(np.float32)


def _choice_utility(
    *,
    cand_iou: float,
    cand_ari: float,
    core_iou: float,
    core_ari: float,
    ari_weight: float,
    risk_penalty_ari: float,
    risk_penalty_iou: float,
) -> float:
    util = float(cand_iou + ari_weight * cand_ari)
    util -= float(risk_penalty_ari) * max(0.0, float(core_ari - cand_ari))
    util -= float(risk_penalty_iou) * max(0.0, float(core_iou - cand_iou))
    return util


def _label_dense_candidates(
    dense_ious: list[float],
    core_iou: float,
    gain_positive_margin: float,
    label_mode: str,
    oracle_slack: float,
) -> list[int]:
    if not dense_ious:
        return []
    gains = [float(v - core_iou) for v in dense_ious]
    if label_mode == "gain_margin":
        return [int(g > gain_positive_margin) for g in gains]

    best_idx = int(np.argmax(np.asarray(dense_ious, dtype=np.float32)))
    best_iou = float(dense_ious[best_idx])
    best_gain = float(best_iou - core_iou)
    if best_gain <= gain_positive_margin:
        return [0 for _ in dense_ious]

    if label_mode == "oracle_winner":
        labels = [0 for _ in dense_ious]
        labels[best_idx] = 1
        return labels

    if label_mode == "oracle_band":
        return [int((best_iou - float(v)) <= oracle_slack) for v in dense_ious]

    raise ValueError(f"Unsupported label_mode: {label_mode}")


def _synthetic_repair_core_and_candidates(
    *,
    gt: np.ndarray,
    labels: np.ndarray,
    region_classes: list[int],
    target_class: int,
    rng: np.random.Generator,
    cfg: PTDV8PartitionTrainConfig,
    min_area: int,
    wrong_side_negatives_per_sample: int = 0,
    wrong_side_area_ratio_min: float = 0.35,
    wrong_side_area_ratio_max: float = 2.8,
    wrong_side_dilate_ksize: int = 9,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    target_masks: list[np.ndarray] = []
    other_masks: list[np.ndarray] = []
    for ridx, cid in enumerate(region_classes, start=1):
        m = (labels == ridx).astype(np.uint8)
        if int(m.sum()) < min_area:
            continue
        if int(cid) == int(target_class):
            target_masks.append(m)
        else:
            other_masks.append(m)
    if not target_masks:
        return None, []

    def _u(ms: list[np.ndarray]) -> np.ndarray:
        out = np.zeros_like(gt, dtype=np.uint8)
        for m in ms:
            out = np.logical_or(out > 0, m > 0).astype(np.uint8)
        return out

    positives: list[np.ndarray] = []
    core: np.ndarray | None = None

    if len(target_masks) >= 2 and rng.random() < 0.7:
        keep_n = int(rng.integers(1, len(target_masks)))
        perm = rng.permutation(len(target_masks)).tolist()
        keep_idx = set(perm[:keep_n])
        kept = [target_masks[i] for i in range(len(target_masks)) if i in keep_idx]
        omitted = [target_masks[i] for i in range(len(target_masks)) if i not in keep_idx]
        core = _u(kept)
        run = core.copy()
        for m in omitted:
            run = np.logical_or(run > 0, m > 0).astype(np.uint8)
            positives.append(run.astype(np.uint8))
    else:
        frags = _fragment_mask(gt.astype(np.uint8), rng, max(3, cfg.min_fg_frags), max(cfg.min_fg_frags + 1, cfg.max_fg_frags))
        frags = [f.astype(np.uint8) for f in frags if int(f.sum()) >= min_area]
        if len(frags) >= 2:
            keep_n = int(rng.integers(1, len(frags)))
            perm = rng.permutation(len(frags)).tolist()
            keep_idx = set(perm[:keep_n])
            kept = [frags[i] for i in range(len(frags)) if i in keep_idx]
            omitted = [frags[i] for i in range(len(frags)) if i not in keep_idx]
            core = _u(kept)
            run = core.copy()
            for m in omitted:
                run = np.logical_or(run > 0, m > 0).astype(np.uint8)
                positives.append(run.astype(np.uint8))
        else:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            core_er = cv2.erode(gt.astype(np.uint8), k, iterations=1)
            if int(core_er.sum()) >= min_area:
                core = core_er.astype(np.uint8)
                positives.append(gt.astype(np.uint8))

    if core is None or int(core.sum()) < min_area or np.array_equal(core, gt):
        return None, []

    negatives: list[np.ndarray] = []
    rng.shuffle(other_masks)
    for om in other_masks[: min(3, len(other_masks))]:
        negatives.append(np.logical_or(core > 0, om > 0).astype(np.uint8))
        negatives.append(np.logical_or(gt > 0, om > 0).astype(np.uint8))
        if rng.random() < 0.5:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            negatives.append(cv2.dilate(np.logical_or(core > 0, om > 0).astype(np.uint8), k, iterations=1))

    negatives.extend(
        _coherent_wrong_side_candidates(
            core=core.astype(np.uint8),
            other_masks=other_masks,
            rng=rng,
            min_area=min_area,
            max_candidates=int(max(0, wrong_side_negatives_per_sample)),
            area_ratio_min=float(wrong_side_area_ratio_min),
            area_ratio_max=float(wrong_side_area_ratio_max),
            dilate_ksize=int(wrong_side_dilate_ksize),
        )
    )

    cands = positives + negatives + [gt.astype(np.uint8)]
    cands = [c.astype(np.uint8) for c in cands if int(c.sum()) >= min_area and not np.array_equal(c, core)]
    return core.astype(np.uint8), _dedupe(cands, min_area=min_area)


def _coherent_wrong_side_candidates(
    *,
    core: np.ndarray,
    other_masks: list[np.ndarray],
    rng: np.random.Generator,
    min_area: int,
    max_candidates: int,
    area_ratio_min: float,
    area_ratio_max: float,
    dilate_ksize: int,
) -> list[np.ndarray]:
    if max_candidates <= 0 or not other_masks:
        return []

    core_bin = (core > 0).astype(np.uint8)
    core_area = int(core_bin.sum())
    if core_area < min_area:
        return []

    union_other = np.zeros_like(core_bin, dtype=np.uint8)
    for om in other_masks:
        union_other = np.logical_or(union_other > 0, om > 0).astype(np.uint8)

    cands: list[np.ndarray] = []
    if int(union_other.sum()) >= min_area:
        cands.append(union_other.astype(np.uint8))
        ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(union_other.astype(np.uint8), connectivity=8)
        if ncc > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            j = int(np.argmax(areas)) + 1
            largest = (lbl == j).astype(np.uint8)
            if int(largest.sum()) >= min_area:
                cands.append(largest)

    scored: list[tuple[float, np.ndarray]] = []
    for om in other_masks:
        m = (om > 0).astype(np.uint8)
        a = int(m.sum())
        if a < min_area:
            continue
        ratio = float(a / max(core_area, 1))
        if ratio < float(area_ratio_min) or ratio > float(area_ratio_max):
            continue
        score = abs(math.log(max(ratio, 1e-6)))
        scored.append((score, m))

    scored.sort(key=lambda t: t[0])
    for _, m in scored[: max_candidates]:
        cands.append(m.astype(np.uint8))
        if dilate_ksize >= 3 and dilate_ksize % 2 == 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate_ksize), int(dilate_ksize)))
            md = cv2.dilate(m.astype(np.uint8), k, iterations=1)
            if int(md.sum()) >= min_area:
                cands.append(md.astype(np.uint8))

    if cands:
        rng.shuffle(cands)
    return _dedupe(cands, min_area=min_area)[: max_candidates]


def _synthetic_keep_core_and_candidates(
    *,
    gt: np.ndarray,
    labels: np.ndarray,
    region_classes: list[int],
    target_class: int,
    rng: np.random.Generator,
    min_area: int,
    wrong_side_negatives_per_sample: int = 0,
    wrong_side_area_ratio_min: float = 0.35,
    wrong_side_area_ratio_max: float = 2.8,
    wrong_side_dilate_ksize: int = 9,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    core = gt.astype(np.uint8)
    if int(core.sum()) < min_area:
        return None, []

    other_masks: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    for ridx, cid in enumerate(region_classes, start=1):
        m = (labels == ridx).astype(np.uint8)
        if int(m.sum()) < min_area:
            continue
        if int(cid) == int(target_class):
            target_masks.append(m)
        else:
            other_masks.append(m)

    negatives: list[np.ndarray] = []
    rng.shuffle(other_masks)
    for om in other_masks[: min(4, len(other_masks))]:
        negatives.append(np.logical_or(core > 0, om > 0).astype(np.uint8))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        negatives.append(cv2.dilate(np.logical_or(core > 0, om > 0).astype(np.uint8), k, iterations=1))

    if len(target_masks) >= 2:
        perm = rng.permutation(len(target_masks)).tolist()
        merged = core.copy()
        for idx in perm[: min(2, len(perm))]:
            # Boundary bleed across same-texture fragments with extra dilation creates tempting but unsafe candidates.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            merged = cv2.dilate(np.logical_or(merged > 0, target_masks[idx] > 0).astype(np.uint8), k, iterations=1)
            negatives.append(merged.astype(np.uint8))

    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dil = cv2.dilate(core.astype(np.uint8), k_big, iterations=1)
    if int(dil.sum()) >= min_area and not np.array_equal(dil, core):
        negatives.append(dil.astype(np.uint8))

    negatives.extend(
        _coherent_wrong_side_candidates(
            core=core.astype(np.uint8),
            other_masks=other_masks,
            rng=rng,
            min_area=min_area,
            max_candidates=int(max(0, wrong_side_negatives_per_sample)),
            area_ratio_min=float(wrong_side_area_ratio_min),
            area_ratio_max=float(wrong_side_area_ratio_max),
            dilate_ksize=int(wrong_side_dilate_ksize),
        )
    )

    cands = [c.astype(np.uint8) for c in negatives if int(c.sum()) >= min_area and not np.array_equal(c, core)]
    return core, _dedupe(cands, min_area=min_area)


def _build_classifier(args: argparse.Namespace):
    if str(args.cls_model) == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=280,
            l2_regularization=0.01,
            random_state=int(args.seed),
        )
    if str(args.cls_model) == "rf_balanced":
        return RandomForestClassifier(
            n_estimators=320,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=int(args.seed),
            n_jobs=-1,
        )
    if str(args.cls_model) == "logreg_balanced":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=int(args.seed),
            solver="liblinear",
        )
    if str(args.cls_model) == "torch_mlp":
        return TorchMLPBinaryClassifier(device=str(args.ptd_device), seed=int(args.seed))
    raise ValueError(f"Unsupported cls_model: {args.cls_model}")


def _positive_proba(estimator, X: np.ndarray) -> np.ndarray:
    try:
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32)
        if proba.ndim == 2 and proba.shape[1] == 1:
            cls = int(getattr(estimator, "classes_", np.array([0]))[0])
            fill = 1.0 if cls == 1 else 0.0
            return np.full((X.shape[0],), fill, dtype=np.float32)
    except Exception:
        pass
    pred = estimator.predict(X).astype(np.float32)
    return pred


def _scene_gate_feature(
    group_X: np.ndarray,
    pred_gain: np.ndarray,
    chosen_idx: int,
    pred_prob: np.ndarray | None = None,
) -> np.ndarray:
    if group_X.ndim != 2:
        raise ValueError("group_X must be 2D")
    if pred_gain.ndim != 1 or pred_gain.shape[0] != group_X.shape[0]:
        raise ValueError("pred_gain must align with group_X")
    top = group_X[chosen_idx].astype(np.float32)
    top_gain = float(pred_gain[chosen_idx])
    if pred_gain.shape[0] > 1:
        order = np.argsort(pred_gain)
        second_gain = float(pred_gain[int(order[-2])])
    else:
        second_gain = top_gain
    stats = [
        top_gain,
        second_gain,
        top_gain - second_gain,
        float(np.mean(pred_gain)),
        float(np.std(pred_gain)),
        float(pred_gain.shape[0] / 24.0),
    ]
    if pred_prob is not None:
        if pred_prob.ndim != 1 or pred_prob.shape[0] != group_X.shape[0]:
            raise ValueError("pred_prob must align with group_X")
        top_prob = float(pred_prob[chosen_idx])
        if pred_prob.shape[0] > 1:
            order_prob = np.argsort(pred_prob)
            second_prob = float(pred_prob[int(order_prob[-2])])
        else:
            second_prob = top_prob
        stats.extend(
            [
                top_prob,
                second_prob,
                top_prob - second_prob,
                float(np.mean(pred_prob)),
                float(np.std(pred_prob)),
            ]
        )
    extra = np.array(stats, dtype=np.float32)
    return np.concatenate([top, extra], axis=0).astype(np.float32)


def _triage_feature_from_option_block(
    *,
    option_block: np.ndarray,
    pred_gain: np.ndarray,
    pred_prob: np.ndarray,
    risk_prob: np.ndarray,
    support_raw: np.ndarray | None = None,
    support_norm: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Build a fixed-width scene vector for the 3-way policy:
    choose among {keep core, rank-1 dense, rank-2 dense}.
    """
    if option_block.ndim != 2:
        raise ValueError("option_block must be 2D")
    n_opt, feat_dim = option_block.shape
    if n_opt < 1:
        raise ValueError("option_block must include at least keep-core option")
    n_dense = max(0, n_opt - 1)
    core = option_block[0].astype(np.float32)
    zero = np.zeros((feat_dim,), dtype=np.float32)

    if n_dense <= 0 or pred_gain.size <= 0:
        stats = np.array(
            [
                0.0,  # n_dense
                -1.0,  # idx1
                -1.0,  # idx2
                1.0,  # top2_duplicate
                0.0, 0.0, 0.0,  # gain1 gain2 gap
                0.0, 0.0, 0.0,  # prob1 prob2 gap
                0.0, 0.0, 0.0,  # risk1 risk2 gap
                0.0, 0.0, 0.0,  # support1 support2 gap
                0.0, 0.0,  # support_norm1 support_norm2
                0.0, 0.0,  # mean/std gain
                0.0, 0.0,  # mean/std prob
                0.0, 0.0,  # mean/std risk
                0.0, 0.0,  # mean/std support
            ],
            dtype=np.float32,
        )
        feat = np.concatenate([core, zero, zero, stats], axis=0).astype(np.float32)
        return feat, (-1, -1)

    j1, j2 = _top2_indices(pred_gain)
    if j1 < 0:
        stats = np.zeros((26,), dtype=np.float32)
        feat = np.concatenate([core, zero, zero, stats], axis=0).astype(np.float32)
        return feat, (-1, -1)
    if j2 < 0:
        j2 = j1
    j1 = int(max(0, min(j1, n_dense - 1)))
    j2 = int(max(0, min(j2, n_dense - 1)))

    if pred_prob.shape[0] != n_dense:
        pred_prob = np.zeros((n_dense,), dtype=np.float32)
    if risk_prob.shape[0] != n_dense:
        risk_prob = np.zeros((n_dense,), dtype=np.float32)
    if support_raw is None or support_raw.shape[0] != n_dense:
        support_raw = np.zeros((n_dense,), dtype=np.float32)
    if support_norm is None or support_norm.shape[0] != n_dense:
        support_norm = np.zeros((n_dense,), dtype=np.float32)

    rank1 = option_block[j1 + 1].astype(np.float32)
    rank2 = option_block[j2 + 1].astype(np.float32)

    g1 = float(pred_gain[j1])
    g2 = float(pred_gain[j2])
    p1 = float(pred_prob[j1])
    p2 = float(pred_prob[j2])
    r1 = float(risk_prob[j1])
    r2 = float(risk_prob[j2])
    s1 = float(support_raw[j1])
    s2 = float(support_raw[j2])
    sn1 = float(support_norm[j1])
    sn2 = float(support_norm[j2])
    stats = np.array(
        [
            float(n_dense),
            float(j1),
            float(j2),
            float(j1 == j2),
            g1,
            g2,
            g1 - g2,
            p1,
            p2,
            p1 - p2,
            r1,
            r2,
            r1 - r2,
            s1,
            s2,
            s1 - s2,
            sn1,
            sn2,
            float(np.mean(pred_gain)),
            float(np.std(pred_gain)),
            float(np.mean(pred_prob)),
            float(np.std(pred_prob)),
            float(np.mean(risk_prob)),
            float(np.std(risk_prob)),
            float(np.mean(support_raw)),
            float(np.std(support_raw)),
        ],
        dtype=np.float32,
    )
    feat = np.concatenate([core, rank1, rank2, stats], axis=0).astype(np.float32)
    return feat, (j1, j2)


def _pick_core_candidate(
    *,
    image_rgb: np.ndarray,
    proposals: list[np.ndarray],
    descriptors: list[np.ndarray],
    feature_map: np.ndarray,
    scorer: PTDV8PartitionScorer,
    core_mode: str,
    v4_scorer: PTDV4SetScorer | None = None,
    v6_scorer: PTDV6CoverageScorer | None = None,
) -> tuple[np.ndarray | None, float]:
    proposal_union = _proposal_union(proposals)

    def _best_mask(s) -> tuple[np.ndarray | None, float]:
        comps, _, _ = s.merge_components(
            image_rgb=image_rgb,
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
        )
        if not comps:
            return None, 0.0
        scores = s.score_components(
            image_rgb=image_rgb,
            components=comps,
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
        )
        if not scores:
            return None, 0.0
        j = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        return (comps[j] > 0).astype(np.uint8), float(scores[j])

    if core_mode == "v9stack":
        if v4_scorer is None or v6_scorer is None:
            raise RuntimeError("v9stack core surrogate requires v4 and v6 scorers.")

        m4, s4 = _best_mask(v4_scorer)
        m6, s6 = _best_mask(v6_scorer)
        m8, s8 = _best_mask(scorer)
        if m4 is None or m6 is None or m8 is None:
            return None, 0.0

        c4, p4, a4 = _cov_prec_area(m4, proposal_union)
        c6, p6, a6 = _cov_prec_area(m6, proposal_union)
        area4 = float((m4 > 0).sum() / max(m4.size, 1))
        area6 = float((m6 > 0).sum() / max(m6.size, 1))
        use6 = False
        if c4 < 0.18 and a4 < 0.22 and (c6 - c4) > 0.15 and p6 > 0.55 and a6 < 1.40:
            use6 = True
        if area4 < 0.010 and area6 > 0.020 and area6 < 0.80 and p6 > 0.40:
            use6 = True
        m7 = m6 if use6 else m4
        s7 = s6 if use6 else s4

        c7, p7, a7 = _cov_prec_area(m7, proposal_union)
        c8, p8, a8 = _cov_prec_area(m8, proposal_union)
        area7 = float((m7 > 0).sum() / max(m7.size, 1))
        area8 = float((m8 > 0).sum() / max(m8.size, 1))
        use8 = False
        if c7 < 0.22 and (c8 - c7) > 0.12 and p8 > 0.45 and a8 < 1.60 and a8 > 0.05:
            use8 = True
        if area7 < 0.012 and area8 > 0.025 and area8 < 0.85 and p8 > 0.30:
            use8 = True
        return ((m8 if use8 else m7) > 0).astype(np.uint8), float(s8 if use8 else s7)

    comps, _, _ = scorer.merge_components(
        image_rgb=image_rgb,
        proposals=proposals,
        descriptors=descriptors,
        feature_map=feature_map,
    )
    if not comps:
        return None, 0.0
    scores = scorer.score_components(
        image_rgb=image_rgb,
        components=comps,
        proposals=proposals,
        descriptors=descriptors,
        feature_map=feature_map,
    )
    if not scores:
        return None, 0.0
    j = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    return (comps[j] > 0).astype(np.uint8), float(scores[j])


def _features_and_scores_for_masks(
    *,
    scorer: PTDV8PartitionScorer,
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    proposals: list[np.ndarray],
    descriptors: list[np.ndarray],
    feature_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    consensus = _proposal_consensus_masks(proposals, min_area=scorer.min_area)
    proposal_union = _proposal_union(proposals)
    feats = scorer._features_for_masks(
        image_rgb=image_rgb,
        masks=masks,
        atom_components=scorer._last_atom_components,
        atom_component_desc=scorer._last_atom_desc,
        proposals=proposals,
        descriptors=descriptors,
        feature_map=feature_map,
        consensus=consensus,
        proposal_union=proposal_union,
    )
    scores = np.asarray(
        scorer.score_components(
            image_rgb=image_rgb,
            components=masks,
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
        ),
        dtype=np.float32,
    )
    return feats.astype(np.float32), scores


def _load_or_train_bundle(args: argparse.Namespace) -> tuple[dict, dict[str, float | int] | None]:
    if args.bundle_path.exists() and args.bundle_metrics_json.exists() and not args.retrain:
        with args.bundle_path.open("rb") as f:
            payload = pickle.load(f)
        metrics = json.loads(args.bundle_metrics_json.read_text(encoding="utf-8"))
        return payload, metrics

    if not args.train_if_missing and not args.retrain:
        raise FileNotFoundError(
            f"Missing bundle/metrics: {args.bundle_path}, {args.bundle_metrics_json}. "
            "Pass --train-if-missing or --retrain."
        )
    print("[acute_rescue] training PTD-only acute rescue bundle...")
    metrics = _train_bundle(args)
    with args.bundle_path.open("rb") as f:
        payload = pickle.load(f)
    return payload, metrics


def _train_bundle(args: argparse.Namespace) -> dict[str, float | int]:
    rng = np.random.default_rng(args.seed)
    backend = PTDImageBackend(args.ptd_root)
    _, entries = load_ptd_entries(args.ptd_root)
    split = split_ptd_entries(entries, val_fraction=0.10, split_seed=args.seed, root=args.ptd_root)
    class_to_entries = group_entries_by_class(split.train)

    encoder = PTDTextureEncoder(PTDEncoderConfig(checkpoint=args.ptd_checkpoint, device=args.ptd_device))
    scorer = PTDV8PartitionScorer(args.ptd_v8_bundle)
    v4_scorer = PTDV4SetScorer(args.ptd_v4_bundle) if args.core_surrogate == "v9stack" else None
    v6_scorer = PTDV6CoverageScorer(args.ptd_v6_bundle) if args.core_surrogate == "v9stack" else None

    synth_cfg = PTDV8PartitionTrainConfig(
        ptd_root=args.ptd_root,
        ptd_encoder_ckpt=args.ptd_checkpoint,
        ptd_v3_bundle=Path("unused"),
        out_bundle=Path("unused"),
        out_metrics_json=Path("unused"),
        num_samples=int(args.synthetic_samples),
        val_fraction=float(args.val_fraction),
        image_size=int(args.synthetic_image_size),
        min_regions=int(args.min_regions),
        max_regions=int(args.max_regions),
        min_fg_frags=int(args.min_fg_frags),
        max_fg_frags=int(args.max_fg_frags),
        random_seed=int(args.seed),
        min_area=int(args.min_area),
        max_class_pool=int(args.max_class_pool),
        multi_target_prob=float(args.multi_target_prob),
        max_proposals_per_sample=int(args.max_proposals_per_sample),
    )

    requested_total = int(args.synthetic_samples)
    n_val = max(4, int(round(requested_total * float(args.val_fraction))))
    n_train = max(8, requested_total - n_val)
    target_total = n_train + n_val

    X_tr: list[np.ndarray] = []
    y_gain_tr: list[float] = []
    y_cls_tr: list[int] = []
    y_risk_tr: list[int] = []
    train_groups: list[_ValGroup] = []
    choice_X_tr: list[np.ndarray] = []
    choice_groups_tr: list[_ChoiceGroup] = []

    X_va: list[np.ndarray] = []
    y_gain_va: list[float] = []
    y_cls_va: list[int] = []
    y_risk_va: list[int] = []
    y_dense_iou_va: list[float] = []
    val_groups: list[_ValGroup] = []
    choice_X_va: list[np.ndarray] = []
    choice_groups_va: list[_ChoiceGroup] = []

    produced = 0
    attempts = 0
    while produced < target_total and attempts < target_total * 8:
        attempts += 1
        image, gt, labels, region_classes, target_class = _compose_synthetic_target_union(
            backend=backend,
            class_to_entries=class_to_entries,
            rng=rng,
            cfg=synth_cfg,
        )
        if int(gt.sum()) < int(args.min_area):
            continue

        proposals = _make_synthetic_proposals_union(
            gt_union=gt,
            labels=labels,
            region_classes=region_classes,
            target_class=target_class,
            rng=rng,
            cfg=synth_cfg,
        )
        if len(proposals) < 2:
            continue

        feat_map = compute_texture_feature_map(image)
        hand_desc = [region_descriptor(feat_map, p) for p in proposals]
        emb = encoder.encode_regions(image, proposals)
        desc = [np.concatenate([h, e], axis=0).astype(np.float32) for h, e in zip(hand_desc, emb)]

        natural_core_mask, natural_core_score = _pick_core_candidate(
            image_rgb=image,
            proposals=proposals,
            descriptors=desc,
            feature_map=feat_map,
            scorer=scorer,
            core_mode=args.core_surrogate,
            v4_scorer=v4_scorer,
            v6_scorer=v6_scorer,
        )
        repair_core_mask = None
        repair_cands: list[np.ndarray] = []
        keep_core_mask = None
        keep_cands: list[np.ndarray] = []
        if str(args.train_core_mode) == "repair_mix":
            if rng.random() < float(args.repair_core_prob):
                repair_core_mask, repair_cands = _synthetic_repair_core_and_candidates(
                    gt=gt,
                    labels=labels,
                    region_classes=region_classes,
                    target_class=target_class,
                    rng=rng,
                    cfg=synth_cfg,
                    min_area=int(args.min_area),
                    wrong_side_negatives_per_sample=int(args.wrong_side_negatives_per_sample),
                    wrong_side_area_ratio_min=float(args.wrong_side_area_ratio_min),
                    wrong_side_area_ratio_max=float(args.wrong_side_area_ratio_max),
                    wrong_side_dilate_ksize=int(args.wrong_side_dilate_ksize),
                )
            else:
                keep_core_mask, keep_cands = _synthetic_keep_core_and_candidates(
                    gt=gt,
                    labels=labels,
                    region_classes=region_classes,
                    target_class=target_class,
                    rng=rng,
                    min_area=int(args.min_area),
                    wrong_side_negatives_per_sample=int(args.wrong_side_negatives_per_sample),
                    wrong_side_area_ratio_min=float(args.wrong_side_area_ratio_min),
                    wrong_side_area_ratio_max=float(args.wrong_side_area_ratio_max),
                    wrong_side_dilate_ksize=int(args.wrong_side_dilate_ksize),
                )
        core_mask = repair_core_mask if repair_core_mask is not None else keep_core_mask if keep_core_mask is not None else natural_core_mask
        core_score = float(natural_core_score)
        if core_mask is None:
            continue

        dense_cands = _dense_candidate_bank(proposals, min_area=int(args.min_area))
        dense_cands.extend(repair_cands)
        dense_cands.extend(keep_cands)
        dense_cands = [m for m in dense_cands if not np.array_equal(m, core_mask)]
        dense_cands = _dedupe(dense_cands, min_area=int(args.min_area))
        if not dense_cands:
            continue
        proposal_union = _proposal_union(proposals)

        all_masks = [core_mask] + dense_cands
        feats_all, scores_all = _features_and_scores_for_masks(
            scorer=scorer,
            image_rgb=image,
            masks=all_masks,
            proposals=proposals,
            descriptors=desc,
            feature_map=feat_map,
        )
        core_feat = feats_all[0]
        core_v8_score = float(scores_all[0])
        core_met = rwtd_invariant_metrics(core_mask, gt)
        core_iou = float(core_met.iou)
        core_ari = float(core_met.ari)
        core_teacher_score, core_teacher_cov, _, _ = _teacher_score_candidate(core_mask, proposal_union, feat_map)

        sample_X: list[np.ndarray] = []
        sample_gain: list[float] = []
        sample_dense_iou: list[float] = []
        sample_dense_ari: list[float] = []
        sample_risk: list[int] = []
        sample_teacher_score: list[float] = []
        sample_teacher_cov: list[float] = []
        sample_teacher_prec: list[float] = []
        sample_teacher_area: list[float] = []
        sample_choice_X: list[np.ndarray] = [
            _choice_option_feature(
                core_mask=core_mask,
                option_mask=core_mask,
                core_feat=core_feat,
                option_feat=core_feat,
                core_score=core_v8_score,
                option_score=core_v8_score,
                proposal_count=len(proposals),
                proposal_union=proposal_union,
                feat_map=feat_map,
                is_keep=True,
            )
        ]
        for j, dm in enumerate(dense_cands, start=1):
            dense_met = rwtd_invariant_metrics(dm, gt)
            dense_iou = float(dense_met.iou)
            dense_ari = float(dense_met.ari)
            teacher_score, teacher_cov, teacher_prec, teacher_area = _teacher_score_candidate(dm, proposal_union, feat_map)
            gain = float(dense_iou - core_iou)
            pf = _pair_feature(
                core_mask=core_mask,
                dense_mask=dm,
                core_feat=core_feat,
                dense_feat=feats_all[j],
                core_score=core_v8_score,
                dense_score=float(scores_all[j]),
                proposal_count=len(proposals),
                proposal_union=proposal_union,
                feat_map=feat_map,
            )
            sample_X.append(pf)
            sample_choice_X.append(
                _choice_option_feature(
                    core_mask=core_mask,
                    option_mask=dm,
                    core_feat=core_feat,
                    option_feat=feats_all[j],
                    core_score=core_v8_score,
                    option_score=float(scores_all[j]),
                    proposal_count=len(proposals),
                    proposal_union=proposal_union,
                    feat_map=feat_map,
                    is_keep=False,
                )
            )
            sample_gain.append(gain)
            sample_dense_iou.append(dense_iou)
            sample_dense_ari.append(dense_ari)
            sample_risk.append(
                int(
                    (core_ari - dense_ari) > float(args.risk_margin_ari)
                    or (core_iou - dense_iou) > float(args.risk_margin_iou)
                )
            )
            sample_teacher_score.append(teacher_score)
            sample_teacher_cov.append(teacher_cov)
            sample_teacher_prec.append(teacher_prec)
            sample_teacher_area.append(teacher_area)

        if not sample_X:
            continue

        if str(args.supervision_source) == "v11_teacher":
            best_teacher_idx = int(np.argmax(np.asarray(sample_teacher_score, dtype=np.float32)))
            best_teacher_score = float(sample_teacher_score[best_teacher_idx])
            best_teacher_cov = float(sample_teacher_cov[best_teacher_idx])
            best_teacher_prec = float(sample_teacher_prec[best_teacher_idx])
            best_teacher_area = float(sample_teacher_area[best_teacher_idx])
            best_teacher_iou_gain = float(sample_dense_iou[best_teacher_idx] - core_iou)
            best_teacher_ari_gain = float(sample_dense_ari[best_teacher_idx] - core_ari)
            strong_teacher = (
                core_teacher_cov < float(args.teacher_cov9_low)
                and (best_teacher_cov - core_teacher_cov) > float(args.teacher_cov_gain_min)
                and best_teacher_prec > float(args.teacher_prec_best_min)
                and (best_teacher_score - core_teacher_score) > float(args.teacher_score_margin)
                and best_teacher_area > float(args.teacher_area_ratio_min)
                and best_teacher_area < float(args.teacher_area_ratio_max)
                and best_teacher_iou_gain > float(args.teacher_min_iou_gain)
                and best_teacher_ari_gain > float(args.teacher_min_ari_gain)
            )
            sample_cls = [0 for _ in sample_X]
            if strong_teacher:
                sample_cls[best_teacher_idx] = 1
            sample_gain = [float(v - core_teacher_score) for v in sample_teacher_score]
            choice_utils = [float(core_teacher_score)] + [float(v) for v in sample_teacher_score]
        else:
            if str(args.label_mode) == "oracle_safe_winner":
                safe_gain = [
                    float((di - core_iou) + float(args.ari_gain_weight) * (da - core_ari))
                    for di, da in zip(sample_dense_iou, sample_dense_ari)
                ]
                best_idx = int(np.argmax(np.asarray(safe_gain, dtype=np.float32)))
                best_iou_gain = float(sample_dense_iou[best_idx] - core_iou)
                best_ari_gain = float(sample_dense_ari[best_idx] - core_ari)
                sample_cls = [0 for _ in sample_X]
                if best_iou_gain > float(args.gain_positive_margin) and best_ari_gain > float(args.safe_min_ari_gain):
                    sample_cls[best_idx] = 1
                sample_gain = safe_gain
            else:
                sample_cls = _label_dense_candidates(
                    dense_ious=sample_dense_iou,
                    core_iou=core_iou,
                    gain_positive_margin=float(args.gain_positive_margin),
                    label_mode=str(args.label_mode),
                    oracle_slack=float(args.oracle_slack),
                )
            choice_utils = [
                _choice_utility(
                    cand_iou=core_iou,
                    cand_ari=core_ari,
                    core_iou=core_iou,
                    core_ari=core_ari,
                    ari_weight=float(args.ari_gain_weight),
                    risk_penalty_ari=float(args.choice_risk_penalty_ari),
                    risk_penalty_iou=float(args.choice_risk_penalty_iou),
                )
            ] + [
                _choice_utility(
                    cand_iou=di,
                    cand_ari=da,
                    core_iou=core_iou,
                    core_ari=core_ari,
                    ari_weight=float(args.ari_gain_weight),
                    risk_penalty_ari=float(args.choice_risk_penalty_ari),
                    risk_penalty_iou=float(args.choice_risk_penalty_iou),
                )
                for di, da in zip(sample_dense_iou, sample_dense_ari)
            ]

        choice_target = int(np.argmax(np.asarray(choice_utils, dtype=np.float32)))

        is_val = produced >= n_train
        if is_val:
            start = len(X_va)
            X_va.extend(sample_X)
            y_gain_va.extend(sample_gain)
            y_cls_va.extend(sample_cls)
            y_risk_va.extend(sample_risk)
            y_dense_iou_va.extend(sample_dense_iou)
            end = len(X_va)
            oracle_iou = max([core_iou] + sample_dense_iou)
            val_groups.append(_ValGroup(start=start, end=end, core_iou=core_iou, oracle_iou=float(oracle_iou)))
            c_start = len(choice_X_va)
            choice_X_va.extend(sample_choice_X)
            c_end = len(choice_X_va)
            choice_groups_va.append(
                _ChoiceGroup(
                    start=c_start,
                    end=c_end,
                    target=choice_target,
                    option_ious=tuple([core_iou] + sample_dense_iou),
                    option_utils=tuple(choice_utils),
                )
            )
        else:
            start = len(X_tr)
            X_tr.extend(sample_X)
            y_gain_tr.extend(sample_gain)
            y_cls_tr.extend(sample_cls)
            y_risk_tr.extend(sample_risk)
            end = len(X_tr)
            oracle_iou = max([core_iou] + sample_dense_iou)
            train_groups.append(_ValGroup(start=start, end=end, core_iou=core_iou, oracle_iou=float(oracle_iou)))
            c_start = len(choice_X_tr)
            choice_X_tr.extend(sample_choice_X)
            c_end = len(choice_X_tr)
            choice_groups_tr.append(
                _ChoiceGroup(
                    start=c_start,
                    end=c_end,
                    target=choice_target,
                    option_ious=tuple([core_iou] + sample_dense_iou),
                    option_utils=tuple(choice_utils),
                )
            )

        produced += 1
        if produced % 25 == 0:
            print(f"[acute_rescue] generated synthetic samples: {produced}/{target_total}")

    if produced < max(8, int(0.60 * target_total)):
        raise RuntimeError(
            f"Too few synthetic samples generated ({produced}/{target_total}). Check PTD root: {args.ptd_root}"
        )

    Xtr = np.stack(X_tr, axis=0).astype(np.float32)
    ytr_gain = np.asarray(y_gain_tr, dtype=np.float32)
    ytr_cls = np.asarray(y_cls_tr, dtype=np.int32)
    ytr_risk = np.asarray(y_risk_tr, dtype=np.int32)
    Xva = np.stack(X_va, axis=0).astype(np.float32)
    yva_gain = np.asarray(y_gain_va, dtype=np.float32)
    yva_cls = np.asarray(y_cls_va, dtype=np.int32)
    yva_risk = np.asarray(y_risk_va, dtype=np.int32)
    yva_dense_iou = np.asarray(y_dense_iou_va, dtype=np.float32)
    Xchoice_tr = np.stack(choice_X_tr, axis=0).astype(np.float32)
    Xchoice_va = np.stack(choice_X_va, axis=0).astype(np.float32)

    reg = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=260,
        l2_regularization=0.01,
        random_state=int(args.seed),
    )
    if len(np.unique(ytr_cls)) < 2:
        cls = DummyClassifier(strategy="constant", constant=int(ytr_cls[0]) if len(ytr_cls) else 0)
    else:
        cls = _build_classifier(args)
    if len(np.unique(ytr_risk)) < 2:
        risk_cls = DummyClassifier(strategy="constant", constant=int(ytr_risk[0]) if len(ytr_risk) else 0)
    else:
        risk_cls = _build_classifier(args)
    reg.fit(Xtr, ytr_gain)
    cls.fit(Xtr, ytr_cls)
    risk_cls.fit(Xtr, ytr_risk)

    reg_tr = reg.predict(Xtr).astype(np.float32)
    cls_tr = _positive_proba(cls, Xtr)
    risk_tr = _positive_proba(risk_cls, Xtr)
    cls_tr_label = cls.predict(Xtr).astype(np.int32)
    reg_va = reg.predict(Xva).astype(np.float32)
    cls_va = _positive_proba(cls, Xva)
    risk_va = _positive_proba(risk_cls, Xva)
    Xchoice_tr_aug = _augment_choice_features_with_pair_preds(
        choice_X=Xchoice_tr,
        choice_groups=choice_groups_tr,
        pair_groups=train_groups,
        pred_gain=reg_tr,
        pred_prob=cls_tr,
        risk_prob=risk_tr,
    )
    Xchoice_va_aug = _augment_choice_features_with_pair_preds(
        choice_X=Xchoice_va,
        choice_groups=choice_groups_va,
        pair_groups=val_groups,
        pred_gain=reg_va,
        pred_prob=cls_va,
        risk_prob=risk_va,
    )
    triage_model = None
    triage_val_acc = 0.0
    triage_val_sel_util = 0.0
    triage_val_sel_iou = 0.0
    triage_val_oracle_util = 0.0
    triage_val_oracle_iou = 0.0
    triage_val_switch_rate = 0.0
    triage_pred_va = np.zeros((0,), dtype=np.int32)
    triage_sel_iou_va: list[float] = []
    triage_oracle_iou_va: list[float] = []
    triage_X_tr: list[np.ndarray] = []
    triage_y_tr: list[int] = []
    triage_X_va: list[np.ndarray] = []
    triage_y_va: list[int] = []
    triage_va_utils: list[tuple[float, float, float]] = []
    triage_va_ious: list[tuple[float, float, float]] = []
    triage_va_oracle_utils: list[float] = []
    triage_va_oracle_ious: list[float] = []
    for cg, pg in zip(choice_groups_tr, train_groups):
        local_block = Xchoice_tr_aug[cg.start:cg.end]
        local_gain = reg_tr[pg.start:pg.end]
        local_prob = cls_tr[pg.start:pg.end]
        local_risk = risk_tr[pg.start:pg.end]
        tri_feat, (j1, j2) = _triage_feature_from_option_block(
            option_block=local_block,
            pred_gain=local_gain,
            pred_prob=local_prob,
            risk_prob=local_risk,
        )
        u0 = float(cg.option_utils[0])
        u1 = float(cg.option_utils[j1 + 1]) if j1 >= 0 and (j1 + 1) < len(cg.option_utils) else u0
        u2 = float(cg.option_utils[j2 + 1]) if j2 >= 0 and (j2 + 1) < len(cg.option_utils) else u0
        target = int(np.argmax(np.asarray([u0, u1, u2], dtype=np.float32)))
        triage_X_tr.append(tri_feat)
        triage_y_tr.append(target)
    for cg, pg in zip(choice_groups_va, val_groups):
        local_block = Xchoice_va_aug[cg.start:cg.end]
        local_gain = reg_va[pg.start:pg.end]
        local_prob = cls_va[pg.start:pg.end]
        local_risk = risk_va[pg.start:pg.end]
        tri_feat, (j1, j2) = _triage_feature_from_option_block(
            option_block=local_block,
            pred_gain=local_gain,
            pred_prob=local_prob,
            risk_prob=local_risk,
        )
        u0 = float(cg.option_utils[0])
        u1 = float(cg.option_utils[j1 + 1]) if j1 >= 0 and (j1 + 1) < len(cg.option_utils) else u0
        u2 = float(cg.option_utils[j2 + 1]) if j2 >= 0 and (j2 + 1) < len(cg.option_utils) else u0
        i0 = float(cg.option_ious[0])
        i1 = float(cg.option_ious[j1 + 1]) if j1 >= 0 and (j1 + 1) < len(cg.option_ious) else i0
        i2 = float(cg.option_ious[j2 + 1]) if j2 >= 0 and (j2 + 1) < len(cg.option_ious) else i0
        target = int(np.argmax(np.asarray([u0, u1, u2], dtype=np.float32)))
        triage_X_va.append(tri_feat)
        triage_y_va.append(target)
        triage_va_utils.append((u0, u1, u2))
        triage_va_ious.append((i0, i1, i2))
        triage_va_oracle_utils.append(float(max(cg.option_utils)))
        triage_va_oracle_ious.append(float(max(cg.option_ious)))
    if triage_X_tr and triage_X_va:
        Xt = np.stack(triage_X_tr, axis=0).astype(np.float32)
        yt = np.asarray(triage_y_tr, dtype=np.int32)
        Xv = np.stack(triage_X_va, axis=0).astype(np.float32)
        yv = np.asarray(triage_y_va, dtype=np.int32)
        if len(np.unique(yt)) < 2:
            triage_model = DummyClassifier(strategy="most_frequent")
            triage_model.fit(Xt, yt)
        else:
            triage_model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=320,
                l2_regularization=0.02,
                random_state=int(args.seed),
            )
            counts = np.bincount(yt, minlength=3).astype(np.float32)
            inv = np.zeros_like(counts)
            inv[counts > 0] = 1.0 / counts[counts > 0]
            sw = inv[yt]
            sw = sw * (len(sw) / max(float(np.sum(sw)), 1e-6))
            triage_model.fit(Xt, yt, sample_weight=sw)
        triage_pred_va = triage_model.predict(Xv).astype(np.int32)
        triage_val_acc = float(np.mean(triage_pred_va == yv)) if len(yv) else 0.0
        triage_val_switch_rate = float(np.mean(triage_pred_va > 0)) if len(triage_pred_va) else 0.0
        for idx, pred_cls in enumerate(triage_pred_va.tolist()):
            p = int(max(0, min(int(pred_cls), 2)))
            triage_val_sel_util += float(triage_va_utils[idx][p])
            triage_val_sel_iou += float(triage_va_ious[idx][p])
            triage_sel_iou_va.append(float(triage_va_ious[idx][p]))
            triage_oracle_iou_va.append(float(triage_va_oracle_ious[idx]))
            triage_val_oracle_util += float(triage_va_oracle_utils[idx])
            triage_val_oracle_iou += float(triage_va_oracle_ious[idx])
        ntri = max(len(triage_pred_va), 1)
        triage_val_sel_util /= float(ntri)
        triage_val_sel_iou /= float(ntri)
        triage_val_oracle_util /= float(ntri)
        triage_val_oracle_iou /= float(ntri)

    mae = float(mean_absolute_error(yva_gain, reg_va))
    try:
        auc = float(roc_auc_score(yva_cls, cls_va))
    except Exception:
        auc = 0.0
    try:
        risk_auc = float(roc_auc_score(yva_risk, risk_va))
    except Exception:
        risk_auc = 0.0
    f1 = float(f1_score(yva_cls, (cls_va >= 0.5).astype(np.int32)))

    cls_va_label = cls.predict(Xva).astype(np.int32)
    chosen: list[float] = []
    oracle: list[float] = []
    switches = 0
    gate = None
    choice_model = None
    gate_auc = 0.0
    gate_f1 = 0.0
    gate_pos_rate_tr = 0.0
    gate_pos_rate_va = 0.0
    choice_val_acc = 0.0
    choice_val_sel_util = 0.0
    choice_val_sel_iou = 0.0
    choice_val_oracle_util = 0.0
    decision_architecture = str(args.decision_architecture)
    if decision_architecture == "group_argmax":
        choice_model = TorchMLPGroupChooser(device=str(args.ptd_device), seed=int(args.seed))
        choice_metrics = choice_model.fit(Xchoice_tr_aug, choice_groups_tr, Xchoice_va_aug, choice_groups_va)
        choice_val_acc = float(choice_metrics.get("choice_val_acc", 0.0))
        choice_val_sel_util = float(choice_metrics.get("choice_val_selected_utility_mean", 0.0))
        choice_val_sel_iou = float(choice_metrics.get("choice_val_selected_iou_mean", 0.0))
        choice_val_oracle_util = float(choice_metrics.get("choice_val_oracle_utility_mean", 0.0))
        val_scores = choice_model.predict_scores(Xchoice_va_aug)
        for g in choice_groups_va:
            local_scores = val_scores[g.start:g.end]
            j_local = int(np.argmax(local_scores))
            chosen.append(float(g.option_ious[j_local]))
            oracle.append(float(max(g.option_ious)))
            switches += int(j_local > 0)
        gate_pos_rate_va = float(choice_metrics.get("choice_val_switch_rate", 0.0))
    elif decision_architecture == "triage_top2":
        if triage_model is None:
            raise RuntimeError("triage_top2 decision architecture requires a trained triage estimator.")
        chosen = [float(v) for v in triage_sel_iou_va]
        oracle = [float(v) for v in triage_oracle_iou_va]
        switches = int(np.sum(triage_pred_va > 0)) if triage_pred_va.size else 0
        gate_pos_rate_va = float(triage_val_switch_rate)
    elif decision_architecture == "scene_gate":
        Xg_tr: list[np.ndarray] = []
        yg_tr: list[int] = []
        Xg_va: list[np.ndarray] = []
        yg_va: list[int] = []
        dense_iou_sel_va: list[float] = []

        for g in train_groups:
            local_reg = reg_tr[g.start:g.end]
            local_prob = cls_tr[g.start:g.end]
            local_gain = ytr_gain[g.start:g.end]
            local_X = Xtr[g.start:g.end]
            j_local = int(np.argmax(local_reg))
            Xg_tr.append(_scene_gate_feature(local_X, local_reg, j_local, local_prob))
            yg_tr.append(int(float(local_gain[j_local]) > float(args.gain_positive_margin)))

        for g in val_groups:
            local_reg = reg_va[g.start:g.end]
            local_prob = cls_va[g.start:g.end]
            local_gain = yva_gain[g.start:g.end]
            local_X = Xva[g.start:g.end]
            j_local = int(np.argmax(local_reg))
            Xg_va.append(_scene_gate_feature(local_X, local_reg, j_local, local_prob))
            yg_va.append(int(float(local_gain[j_local]) > float(args.gain_positive_margin)))
            dense_iou_sel_va.append(float(g.core_iou + float(local_gain[j_local])))

        Xgtr = np.stack(Xg_tr, axis=0).astype(np.float32)
        ygtr = np.asarray(yg_tr, dtype=np.int32)
        Xgva = np.stack(Xg_va, axis=0).astype(np.float32)
        ygva = np.asarray(yg_va, dtype=np.int32)
        gate_pos_rate_tr = float(np.mean(ygtr)) if len(ygtr) else 0.0
        gate_pos_rate_va = float(np.mean(ygva)) if len(ygva) else 0.0

        if len(np.unique(ygtr)) < 2:
            gate = DummyClassifier(strategy="constant", constant=int(ygtr[0]) if len(ygtr) else 0)
        else:
            gate = _build_classifier(args)
        gate.fit(Xgtr, ygtr)

        try:
            gate_va = gate.predict_proba(Xgva)[:, 1].astype(np.float32)
        except Exception:
            gate_va = gate.predict(Xgva).astype(np.float32)
        gate_va_label = gate.predict(Xgva).astype(np.int32)
        try:
            gate_auc = float(roc_auc_score(ygva, gate_va))
        except Exception:
            gate_auc = 0.0
        try:
            gate_f1 = float(f1_score(ygva, gate_va_label))
        except Exception:
            gate_f1 = 0.0

        for idx, g in enumerate(val_groups):
            if int(gate_va_label[idx]) > 0:
                chosen.append(float(dense_iou_sel_va[idx]))
                switches += 1
            else:
                chosen.append(float(g.core_iou))
            oracle.append(float(g.oracle_iou))
    elif decision_architecture == "candidate_veto_gate":
        Xg_tr: list[np.ndarray] = []
        yg_tr: list[int] = []
        Xg_va: list[np.ndarray] = []
        yg_va: list[int] = []
        dense_iou_sel_va: list[float] = []
        val_has_pos: list[bool] = []

        for g in train_groups:
            local_reg = reg_tr[g.start:g.end]
            local_prob = cls_tr[g.start:g.end]
            local_pred = cls_tr_label[g.start:g.end]
            local_true = ytr_cls[g.start:g.end]
            local_X = Xtr[g.start:g.end]
            pos = np.where(local_pred > 0)[0]
            if pos.size == 0:
                continue
            j_local = int(pos[np.argmax(local_reg[pos])])
            Xg_tr.append(_scene_gate_feature(local_X, local_reg, j_local, local_prob))
            yg_tr.append(int(local_true[j_local] > 0))

        for g in val_groups:
            local_reg = reg_va[g.start:g.end]
            local_prob = cls_va[g.start:g.end]
            local_pred = cls_va_label[g.start:g.end]
            local_true = yva_cls[g.start:g.end]
            local_iou = yva_dense_iou[g.start:g.end]
            local_X = Xva[g.start:g.end]
            pos = np.where(local_pred > 0)[0]
            if pos.size == 0:
                val_has_pos.append(False)
                dense_iou_sel_va.append(float(g.core_iou))
                continue
            j_local = int(pos[np.argmax(local_reg[pos])])
            val_has_pos.append(True)
            Xg_va.append(_scene_gate_feature(local_X, local_reg, j_local, local_prob))
            yg_va.append(int(local_true[j_local] > 0))
            dense_iou_sel_va.append(float(local_iou[j_local]))

        if Xg_tr:
            Xgtr = np.stack(Xg_tr, axis=0).astype(np.float32)
            ygtr = np.asarray(yg_tr, dtype=np.int32)
            gate_pos_rate_tr = float(np.mean(ygtr)) if len(ygtr) else 0.0
            if len(np.unique(ygtr)) < 2:
                gate = DummyClassifier(strategy="constant", constant=int(ygtr[0]) if len(ygtr) else 0)
            else:
                gate = _build_classifier(args)
            gate.fit(Xgtr, ygtr)
        else:
            gate = DummyClassifier(strategy="constant", constant=0)
            gate.fit(np.zeros((1, Xtr.shape[1] + 11), dtype=np.float32), np.zeros((1,), dtype=np.int32))
            gate_pos_rate_tr = 0.0

        if Xg_va:
            Xgva = np.stack(Xg_va, axis=0).astype(np.float32)
            ygva = np.asarray(yg_va, dtype=np.int32)
            gate_pos_rate_va = float(np.mean(ygva)) if len(ygva) else 0.0
            gate_va = _positive_proba(gate, Xgva)
            gate_va_label = gate.predict(Xgva).astype(np.int32)
            try:
                gate_auc = float(roc_auc_score(ygva, gate_va))
            except Exception:
                gate_auc = 0.0
            try:
                gate_f1 = float(f1_score(ygva, gate_va_label))
            except Exception:
                gate_f1 = 0.0
        else:
            gate_pos_rate_va = 0.0
            gate_va_label = np.zeros((0,), dtype=np.int32)
            gate_auc = 0.0
            gate_f1 = 0.0

        gate_ptr = 0
        for idx, g in enumerate(val_groups):
            if not val_has_pos[idx]:
                chosen.append(float(g.core_iou))
                oracle.append(float(g.oracle_iou))
                continue
            if int(gate_va_label[gate_ptr]) > 0:
                chosen.append(float(dense_iou_sel_va[idx]))
                switches += 1
            else:
                chosen.append(float(g.core_iou))
            oracle.append(float(g.oracle_iou))
            gate_ptr += 1
    else:
        for g in val_groups:
            local_reg = reg_va[g.start:g.end]
            local_cls = cls_va_label[g.start:g.end]
            local_iou = yva_dense_iou[g.start:g.end]
            pos = np.where(local_cls > 0)[0]
            if pos.size > 0:
                j_local = int(pos[np.argmax(local_reg[pos])])
                chosen.append(float(local_iou[j_local]))
                switches += 1
            else:
                chosen.append(float(g.core_iou))
            oracle.append(float(g.oracle_iou))
    best_sel = float(np.mean(chosen)) if chosen else 0.0
    best_oracle = float(np.mean(oracle)) if oracle else 0.0
    best_switch = float(switches / max(len(val_groups), 1))

    if decision_architecture == "group_argmax":
        feature_version = "ptd_acute_rescue_v2_choice"
    elif decision_architecture == "triage_top2":
        feature_version = "ptd_acute_rescue_v3_triage"
    else:
        feature_version = "ptd_acute_rescue_v1"

    payload = {
        "feature_version": feature_version,
        "reg_model": reg,
        "cls_estimator": cls,
        "risk_estimator": risk_cls,
        "gate_estimator": gate,
        "choice_estimator": choice_model,
        "triage_estimator": triage_model,
        "gain_positive_margin": float(args.gain_positive_margin),
        "label_mode": str(args.label_mode),
        "oracle_slack": float(args.oracle_slack),
        "safe_min_ari_gain": float(args.safe_min_ari_gain),
        "ari_gain_weight": float(args.ari_gain_weight),
        "choice_risk_penalty_ari": float(args.choice_risk_penalty_ari),
        "choice_risk_penalty_iou": float(args.choice_risk_penalty_iou),
        "risk_margin_ari": float(args.risk_margin_ari),
        "risk_margin_iou": float(args.risk_margin_iou),
        "cls_model_name": str(args.cls_model),
        "decision_architecture": decision_architecture,
        "supervision_source": str(args.supervision_source),
        "train_core_mode": str(args.train_core_mode),
        "repair_core_prob": float(args.repair_core_prob),
        "wrong_side_negatives_per_sample": int(args.wrong_side_negatives_per_sample),
        "wrong_side_area_ratio_min": float(args.wrong_side_area_ratio_min),
        "wrong_side_area_ratio_max": float(args.wrong_side_area_ratio_max),
        "wrong_side_dilate_ksize": int(args.wrong_side_dilate_ksize),
        "decision_margin_min": float(args.decision_margin_min),
        "abstain_margin": float(args.abstain_margin),
        "plausible_gap": float(args.plausible_gap),
        "support_prior_weight": float(args.support_prior_weight),
        "boundary_tolerance": int(args.boundary_tolerance),
        "min_area": int(args.min_area),
        "core_surrogate": str(args.core_surrogate),
        "ptd_v4_bundle": str(Path(args.ptd_v4_bundle).resolve()),
        "ptd_v6_bundle": str(Path(args.ptd_v6_bundle).resolve()),
        "ptd_v8_bundle": str(Path(args.ptd_v8_bundle).resolve()),
        "ptd_checkpoint": str(Path(args.ptd_checkpoint).resolve()),
    }
    args.bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with args.bundle_path.open("wb") as f:
        pickle.dump(payload, f)

    metrics = {
        "synthetic_samples_target": int(target_total),
        "synthetic_samples_generated": int(produced),
        "train_pairs": int(Xtr.shape[0]),
        "val_pairs": int(Xva.shape[0]),
        "val_groups": int(len(val_groups)),
        "reg_val_mae_gain": float(mae),
        "cls_val_auc": float(auc),
        "risk_val_auc": float(risk_auc),
        "cls_val_f1@0.5": float(f1),
        "gate_val_auc": float(gate_auc),
        "gate_val_f1@0.5": float(gate_f1),
        "gate_positive_rate_train": float(gate_pos_rate_tr),
        "gate_positive_rate_val": float(gate_pos_rate_va),
        "val_selected_iou_mean": float(best_sel),
        "val_oracle_iou_mean": float(best_oracle),
        "val_switch_rate": float(best_switch),
        "choice_val_acc": float(choice_val_acc),
        "choice_val_selected_utility_mean": float(choice_val_sel_util),
        "choice_val_selected_iou_mean": float(choice_val_sel_iou),
        "choice_val_oracle_utility_mean": float(choice_val_oracle_util),
        "triage_val_acc": float(triage_val_acc),
        "triage_val_selected_utility_mean": float(triage_val_sel_util),
        "triage_val_selected_iou_mean": float(triage_val_sel_iou),
        "triage_val_oracle_utility_mean": float(triage_val_oracle_util),
        "triage_val_oracle_iou_mean": float(triage_val_oracle_iou),
        "triage_val_switch_rate": float(triage_val_switch_rate),
        "core_surrogate": str(args.core_surrogate),
        "label_mode": str(args.label_mode),
        "oracle_slack": float(args.oracle_slack),
        "safe_min_ari_gain": float(args.safe_min_ari_gain),
        "ari_gain_weight": float(args.ari_gain_weight),
        "choice_risk_penalty_ari": float(args.choice_risk_penalty_ari),
        "choice_risk_penalty_iou": float(args.choice_risk_penalty_iou),
        "risk_margin_ari": float(args.risk_margin_ari),
        "risk_margin_iou": float(args.risk_margin_iou),
        "cls_model": str(args.cls_model),
        "decision_architecture": decision_architecture,
        "supervision_source": str(args.supervision_source),
        "train_core_mode": str(args.train_core_mode),
        "repair_core_prob": float(args.repair_core_prob),
        "wrong_side_negatives_per_sample": int(args.wrong_side_negatives_per_sample),
        "wrong_side_area_ratio_min": float(args.wrong_side_area_ratio_min),
        "wrong_side_area_ratio_max": float(args.wrong_side_area_ratio_max),
        "wrong_side_dilate_ksize": int(args.wrong_side_dilate_ksize),
        "decision_margin_min": float(args.decision_margin_min),
        "abstain_margin": float(args.abstain_margin),
        "plausible_gap": float(args.plausible_gap),
        "support_prior_weight": float(args.support_prior_weight),
        "boundary_tolerance": int(args.boundary_tolerance),
        "positive_rate_train": float(np.mean(ytr_cls)) if len(ytr_cls) else 0.0,
        "positive_rate_val": float(np.mean(yva_cls)) if len(yva_cls) else 0.0,
    }
    args.bundle_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def run_rwtd(args: argparse.Namespace, payload: dict, train_metrics: dict[str, float | int] | None) -> dict:
    reg = payload["reg_model"]
    decision_architecture = str(payload.get("decision_architecture", args.decision_architecture))
    cls = payload.get("cls_estimator", payload.get("cls_model"))
    risk_cls = payload.get("risk_estimator")
    if decision_architecture != "group_argmax" and (cls is None or isinstance(cls, str)):
        raise RuntimeError("Acute rescue bundle is missing a valid classifier estimator.")
    gate = payload.get("gate_estimator")
    choice_model = payload.get("choice_estimator")
    triage_model = payload.get("triage_estimator")
    min_area = int(payload.get("min_area", args.min_area))

    encoder = PTDTextureEncoder(PTDEncoderConfig(checkpoint=args.ptd_checkpoint, device=args.ptd_device))
    scorer = PTDV8PartitionScorer(args.ptd_v8_bundle)
    store = PromptMaskProposalStore(args.dense_prompt_masks_root, ProposalLoadConfig(min_area=min_area))

    image_dir, label_dir = infer_rwtd_dirs(args.rwtd_root)
    images = list_rwtd_images(image_dir)
    keep_ids = _load_image_id_filter(args.image_ids_file)
    if keep_ids is not None:
        images = [p for p in images if p.stem in keep_ids]
    if args.max_images is not None:
        images = images[: int(args.max_images)]

    out_dir = args.out_root / "texturesam2_acute_learned_rescue"
    out_masks = out_dir / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)
    diag_root = out_dir / "diagnostics"
    diag_json_dir = diag_root / "per_image"
    diag_topk_dir = diag_root / "topk"
    audit_case_dir = diag_root / "audit_cases"
    if args.save_diagnostics:
        diag_json_dir.mkdir(parents=True, exist_ok=True)
    if int(args.topk_output) > 0:
        diag_topk_dir.mkdir(parents=True, exist_ok=True)
    if args.save_audit_artifacts:
        audit_case_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    switched = 0
    ambiguous_count = 0
    abstained_count = 0
    for i, image_path in enumerate(images, start=1):
        image_id = image_path.stem
        id_num = int(image_id)
        image = read_image_rgb(image_path)
        gt = ensure_binary_gt(read_mask_raw(label_dir / f"{image_id}.png"), source_name=str(label_dir / f"{image_id}.png"))
        core = _read_mask(args.v9_masks_root / f"{image_id}.png", image.shape[:2])
        proposals = store.load(id_num, expected_shape=image.shape[:2])

        pred = core.copy()
        source = "v9_keep"
        dense_count = 0
        best_score = float("-inf")
        best_gain = 0.0
        best_prob = 0.0
        decision_margin = 0.0
        top1_gain = 0.0
        top2_gain = 0.0
        top1_prob = 0.0
        top2_prob = 0.0
        probable_count = 0
        support_concentration = 0.0
        selected_support = 0.0
        selected_dense_rank = 0
        ambiguous_flag = 0
        abstained_flag = 0
        topk_oracle_recoverable = 0
        topk_oracle_gain_iou = 0.0
        core_union_iou = 0.0
        final_union_iou = 0.0
        decision_top1_score = 0.0
        decision_top2_score = 0.0
        selected_option_score = 0.0
        selected_dense_idx = -1
        dense_cands: list[np.ndarray] = []
        dense_order_list: list[int] = []
        support_freq_map: np.ndarray | None = None
        selected_mask_iou = 0.0
        candidate_records: list[dict[str, float | int | bool]] = []

        core_met = rwtd_invariant_metrics(core, gt)
        selected_mask_iou = float(core_met.iou)
        core_bf1, core_biou, _ = boundary_metrics(core, gt, tol=int(args.boundary_tolerance))

        if proposals:
            feat_map = compute_texture_feature_map(image)
            hand_desc = [region_descriptor(feat_map, p) for p in proposals]
            emb = encoder.encode_regions(image, proposals)
            desc = [np.concatenate([h, e], axis=0).astype(np.float32) for h, e in zip(hand_desc, emb)]

            scorer.merge_components(
                image_rgb=image,
                proposals=proposals,
                descriptors=desc,
                feature_map=feat_map,
            )
            dense_cands = _dense_candidate_bank(proposals, min_area=min_area)
            dense_cands = [m for m in dense_cands if not np.array_equal(m, core)]
            dense_count = len(dense_cands)
            if dense_cands:
                proposal_union = _proposal_union(proposals)
                core_union_iou = _iou(core, proposal_union)
                feats_all, scores_all = _features_and_scores_for_masks(
                    scorer=scorer,
                    image_rgb=image,
                    masks=[core] + dense_cands,
                    proposals=proposals,
                    descriptors=desc,
                    feature_map=feat_map,
                )
                core_feat = feats_all[0]
                core_score = float(scores_all[0])

                X = np.stack(
                    [
                        _pair_feature(
                            core_mask=core,
                            dense_mask=dm,
                            core_feat=core_feat,
                            dense_feat=feats_all[j + 1],
                            core_score=core_score,
                            dense_score=float(scores_all[j + 1]),
                            proposal_count=len(proposals),
                            proposal_union=proposal_union,
                            feat_map=feat_map,
                        )
                        for j, dm in enumerate(dense_cands)
                    ],
                    axis=0,
                ).astype(np.float32)
                pred_gain = reg.predict(X).astype(np.float32)
                pred_prob = np.zeros_like(pred_gain, dtype=np.float32)
                risk_prob = np.zeros_like(pred_gain, dtype=np.float32)

                stack = np.stack([(p > 0).astype(np.float32) for p in proposals], axis=0)
                freq = stack.mean(axis=0)
                support_freq_map = freq
                support_raw = np.asarray(
                    [float(freq[dm > 0].mean()) if int((dm > 0).sum()) > 0 else 0.0 for dm in dense_cands],
                    dtype=np.float32,
                )
                support_norm = _normalized_support_prior(proposals, dense_cands)
                adjusted_gain = pred_gain + float(args.support_prior_weight) * support_norm

                j_dense_top1, j_dense_top2 = _top2_indices(adjusted_gain)
                if j_dense_top1 >= 0:
                    top1_gain = float(adjusted_gain[j_dense_top1])
                    top2_gain = float(adjusted_gain[j_dense_top2])
                    best_gain = top1_gain
                    probable_count = int(np.sum(adjusted_gain >= (top1_gain - float(args.plausible_gap))))
                if support_raw.size > 0:
                    support_sum = float(np.sum(support_raw))
                    support_concentration = float(np.max(support_raw) / max(support_sum, EPS))
                    if j_dense_top1 >= 0:
                        selected_support = float(support_raw[j_dense_top1])

                dense_iou = np.zeros((dense_count,), dtype=np.float32)
                dense_ari = np.zeros((dense_count,), dtype=np.float32)
                dense_bf1 = np.zeros((dense_count,), dtype=np.float32)
                for j, dm in enumerate(dense_cands):
                    met_dm = rwtd_invariant_metrics(dm, gt)
                    dense_iou[j] = float(met_dm.iou)
                    dense_ari[j] = float(met_dm.ari)
                    bf1_dm, _, _ = boundary_metrics(dm, gt, tol=int(args.boundary_tolerance))
                    dense_bf1[j] = float(bf1_dm)

                selected_dense_idx = -1
                selected_option_idx = 0
                decision_scores = np.concatenate([np.array([0.0], dtype=np.float32), adjusted_gain], axis=0).astype(np.float32)
                decision_probs = np.zeros_like(decision_scores)
                option_X_base = None
                option_X = None

                if decision_architecture in {"group_argmax", "triage_top2"}:
                    if cls is None or isinstance(cls, str) or risk_cls is None or isinstance(risk_cls, str):
                        raise RuntimeError(
                            f"{decision_architecture} requires valid cls and risk estimators for meta-choice features."
                        )
                    pred_prob = _positive_proba(cls, X)
                    risk_prob = _positive_proba(risk_cls, X)
                    option_X_base = np.stack(
                        [
                            _choice_option_feature(
                                core_mask=core,
                                option_mask=core,
                                core_feat=core_feat,
                                option_feat=core_feat,
                                core_score=core_score,
                                option_score=core_score,
                                proposal_count=len(proposals),
                                proposal_union=proposal_union,
                                feat_map=feat_map,
                                is_keep=True,
                            )
                        ]
                        + [
                            _choice_option_feature(
                                core_mask=core,
                                option_mask=dm,
                                core_feat=core_feat,
                                option_feat=feats_all[k + 1],
                                core_score=core_score,
                                option_score=float(scores_all[k + 1]),
                                proposal_count=len(proposals),
                                proposal_union=proposal_union,
                                feat_map=feat_map,
                                is_keep=False,
                            )
                            for k, dm in enumerate(dense_cands)
                        ],
                        axis=0,
                    ).astype(np.float32)
                    option_X = _augment_choice_option_block(option_X_base, adjusted_gain, pred_prob, risk_prob)

                if decision_architecture == "group_argmax":
                    if choice_model is None:
                        raise RuntimeError("group_argmax requires a choice_estimator in the bundle.")
                    if option_X is None:
                        raise RuntimeError("group_argmax internal error: missing option features.")
                    choice_scores = choice_model.predict_scores(option_X).astype(np.float32)
                    j_choice = int(np.argmax(choice_scores)) if choice_scores.size else 0
                    choice_prob = torch.softmax(torch.from_numpy(choice_scores), dim=0).numpy().astype(np.float32)
                    decision_scores = choice_scores
                    decision_probs = choice_prob
                    selected_option_idx = int(j_choice)
                    if j_choice > 0:
                        selected_dense_idx = int(j_choice - 1)
                    best_prob = float(choice_prob[j_choice]) if choice_prob.size else 0.0
                elif decision_architecture == "triage_top2":
                    if triage_model is None:
                        raise RuntimeError("triage_top2 requires a triage_estimator in the bundle.")
                    if option_X is None:
                        raise RuntimeError("triage_top2 internal error: missing option features.")
                    tri_feat, (j1, j2) = _triage_feature_from_option_block(
                        option_block=option_X,
                        pred_gain=adjusted_gain,
                        pred_prob=pred_prob,
                        risk_prob=risk_prob,
                        support_raw=support_raw,
                        support_norm=support_norm,
                    )
                    tri_feat = tri_feat[None, :].astype(np.float32)
                    try:
                        tri_prob = triage_model.predict_proba(tri_feat)[0].astype(np.float32)
                    except Exception:
                        tri_pred = int(triage_model.predict(tri_feat)[0])
                        tri_prob = np.zeros((3,), dtype=np.float32)
                        tri_prob[max(0, min(tri_pred, 2))] = 1.0
                    tri_pred = int(triage_model.predict(tri_feat)[0])
                    tri_pred = int(max(0, min(tri_pred, 2)))
                    decision_scores = tri_prob
                    decision_probs = tri_prob
                    selected_option_idx = tri_pred
                    if tri_pred == 1 and j1 >= 0:
                        selected_dense_idx = int(j1)
                    elif tri_pred == 2 and j2 >= 0:
                        selected_dense_idx = int(j2)
                    best_prob = float(tri_prob[tri_pred]) if tri_prob.size else 0.0
                elif decision_architecture == "scene_gate":
                    if gate is None:
                        raise RuntimeError("Scene-gate decision architecture requires a gate_estimator in the bundle.")
                    pred_prob = _positive_proba(cls, X)
                    j = int(j_dense_top1) if j_dense_top1 >= 0 else 0
                    gate_X = _scene_gate_feature(X, adjusted_gain, j, pred_prob)[None, :].astype(np.float32)
                    try:
                        gate_prob = gate.predict_proba(gate_X)[:, 1].astype(np.float32)
                    except Exception:
                        gate_prob = gate.predict(gate_X).astype(np.float32)
                    gate_cls = gate.predict(gate_X).astype(np.int32)
                    best_prob = float(gate_prob[0]) if gate_prob.size else 0.0
                    if gate_cls.size > 0 and int(gate_cls[0]) > 0:
                        selected_dense_idx = j
                        selected_option_idx = int(j + 1)
                elif decision_architecture == "candidate_veto_gate":
                    if gate is None:
                        raise RuntimeError("candidate_veto_gate requires a gate_estimator in the bundle.")
                    pred_prob = _positive_proba(cls, X)
                    pred_cls = cls.predict(X).astype(np.int32)
                    pos = np.where(pred_cls > 0)[0]
                    if pos.size > 0:
                        j = int(pos[np.argmax(adjusted_gain[pos])])
                        gate_X = _scene_gate_feature(X, adjusted_gain, j, pred_prob)[None, :].astype(np.float32)
                        gate_prob = _positive_proba(gate, gate_X)
                        gate_cls = gate.predict(gate_X).astype(np.int32)
                        best_prob = float(gate_prob[0]) if gate_prob.size else 0.0
                        if gate_cls.size > 0 and int(gate_cls[0]) > 0:
                            selected_dense_idx = j
                            selected_option_idx = int(j + 1)
                    elif j_dense_top1 >= 0 and pred_prob.size:
                        best_prob = float(pred_prob[int(j_dense_top1)])
                else:
                    pred_prob = _positive_proba(cls, X)
                    pred_cls = cls.predict(X).astype(np.int32)
                    pos = np.where(pred_cls > 0)[0]
                    if pos.size > 0:
                        j = int(pos[np.argmax(adjusted_gain[pos])])
                        selected_dense_idx = j
                        selected_option_idx = int(j + 1)
                        best_prob = float(pred_prob[j])
                    elif j_dense_top1 >= 0 and pred_prob.size:
                        best_prob = float(pred_prob[int(j_dense_top1)])

                if decision_architecture == "group_argmax":
                    j_opt1, j_opt2 = _top2_indices(decision_scores)
                    decision_top1_score = float(decision_scores[j_opt1]) if j_opt1 >= 0 else 0.0
                    decision_top2_score = float(decision_scores[j_opt2]) if j_opt2 >= 0 else decision_top1_score
                    top1_prob = float(decision_probs[j_opt1]) if decision_probs.size and j_opt1 >= 0 else 0.0
                    top2_prob = float(decision_probs[j_opt2]) if decision_probs.size and j_opt2 >= 0 else top1_prob
                elif decision_architecture == "triage_top2":
                    j_opt1, j_opt2 = _top2_indices(decision_scores)
                    decision_top1_score = float(decision_scores[j_opt1]) if j_opt1 >= 0 else 0.0
                    decision_top2_score = float(decision_scores[j_opt2]) if j_opt2 >= 0 else decision_top1_score
                    j_prob1, j_prob2 = _top2_indices(pred_prob)
                    top1_prob = float(pred_prob[j_prob1]) if j_prob1 >= 0 else 0.0
                    top2_prob = float(pred_prob[j_prob2]) if j_prob2 >= 0 else top1_prob
                else:
                    j_opt1, j_opt2 = _top2_indices(decision_scores)
                    decision_top1_score = float(decision_scores[j_opt1]) if j_opt1 >= 0 else 0.0
                    decision_top2_score = float(decision_scores[j_opt2]) if j_opt2 >= 0 else decision_top1_score
                    j_prob1, j_prob2 = _top2_indices(pred_prob)
                    top1_prob = float(pred_prob[j_prob1]) if j_prob1 >= 0 else 0.0
                    top2_prob = float(pred_prob[j_prob2]) if j_prob2 >= 0 else top1_prob

                if selected_option_idx < decision_scores.size:
                    selected_option_score = float(decision_scores[selected_option_idx])
                else:
                    selected_option_score = 0.0
                if decision_scores.size <= 1:
                    best_other = selected_option_score
                else:
                    keep = np.ones((decision_scores.size,), dtype=bool)
                    keep[max(0, min(selected_option_idx, decision_scores.size - 1))] = False
                    others = decision_scores[keep]
                    best_other = float(np.max(others)) if others.size else selected_option_score
                decision_margin = _safe_margin(selected_option_score, best_other)

                if selected_dense_idx >= 0:
                    selected_dense_rank = int(np.where(np.argsort(adjusted_gain)[::-1] == selected_dense_idx)[0][0] + 1)
                else:
                    selected_dense_rank = 0

                if selected_dense_idx >= 0 and (
                    decision_margin < float(args.decision_margin_min)
                    or (float(args.abstain_margin) >= 0.0 and decision_margin < float(args.abstain_margin))
                ):
                    if float(args.abstain_margin) >= 0.0 and decision_margin < float(args.abstain_margin):
                        source = "abstain_keep_core"
                        ambiguous_flag = 1
                        abstained_flag = 1
                        ambiguous_count += 1
                        abstained_count += 1
                    else:
                        source = "margin_keep_core"
                    pred = core.copy()
                    selected_dense_idx = -1
                    selected_option_idx = 0
                    selected_option_score = float(decision_scores[0]) if decision_scores.size else 0.0
                else:
                    if float(args.abstain_margin) >= 0.0 and decision_margin < float(args.abstain_margin):
                        ambiguous_flag = 1
                        ambiguous_count += 1
                        if source == "v9_keep":
                            source = "ambiguous_keep_core"
                    if selected_dense_idx >= 0:
                        pred = dense_cands[selected_dense_idx]
                        source = "acute_learned_rescue"
                        switched += 1
                    else:
                        pred = core.copy()
                        if source == "v9_keep" and ambiguous_flag:
                            source = "ambiguous_keep_core"

                if selected_dense_idx >= 0:
                    best_score = float(adjusted_gain[selected_dense_idx])
                    best_gain = float(adjusted_gain[selected_dense_idx])
                    selected_support = float(support_raw[selected_dense_idx]) if support_raw.size else selected_support
                else:
                    best_score = float(selected_option_score)
                    best_gain = 0.0
                if selected_dense_idx < 0 and best_prob == 0.0:
                    best_prob = float(top1_prob)

                selected_iou = float(core_met.iou if selected_dense_idx < 0 else dense_iou[selected_dense_idx])
                selected_mask_iou = float(selected_iou)
                best_dense_iou = float(np.max(dense_iou)) if dense_iou.size else selected_iou
                topk_oracle_gain_iou = float(best_dense_iou - selected_iou)
                topk_oracle_recoverable = int(topk_oracle_gain_iou > 1e-6)
                dense_order = np.argsort(adjusted_gain)[::-1]
                dense_order_list = [int(v) for v in dense_order.tolist()]

                if int(args.topk_output) > 0:
                    topk_dir = diag_topk_dir / image_id
                    topk_dir.mkdir(parents=True, exist_ok=True)
                    write_binary_mask(topk_dir / "rank0_core.png", core)
                    for rank, idx_dense in enumerate(dense_order[: int(args.topk_output)], start=1):
                        write_binary_mask(topk_dir / f"rank{rank}_dense_{int(idx_dense)}.png", dense_cands[int(idx_dense)])

                rank_lookup = {int(idx): int(rank + 1) for rank, idx in enumerate(dense_order.tolist())}
                for j, dm in enumerate(dense_cands):
                    rec = {
                        "idx_dense": int(j),
                        "rank_adjusted_gain": int(rank_lookup[int(j)]),
                        "pred_gain": float(pred_gain[j]),
                        "adjusted_gain": float(adjusted_gain[j]),
                        "pred_prob": float(pred_prob[j]) if pred_prob.size else 0.0,
                        "risk_prob": float(risk_prob[j]) if risk_prob.size else 0.0,
                        "support_raw": float(support_raw[j]) if support_raw.size else 0.0,
                        "support_norm": float(support_norm[j]) if support_norm.size else 0.0,
                        "dense_iou": float(dense_iou[j]),
                        "dense_ari": float(dense_ari[j]),
                        "dense_boundary_f1": float(dense_bf1[j]),
                        "delta_iou_vs_core": float(dense_iou[j] - float(core_met.iou)),
                        "delta_ari_vs_core": float(dense_ari[j] - float(core_met.ari)),
                        "is_selected": bool(j == selected_dense_idx),
                    }
                    candidate_records.append(rec)

                if args.save_diagnostics:
                    diag_payload = {
                        "image_id": int(image_id),
                        "decision_architecture": decision_architecture,
                        "source": source,
                        "switched": bool(selected_dense_idx >= 0),
                        "ambiguous_flag": bool(ambiguous_flag),
                        "abstained_flag": bool(abstained_flag),
                        "selected_dense_idx": int(selected_dense_idx),
                        "selected_dense_rank": int(selected_dense_rank),
                        "dense_count": int(dense_count),
                        "proposal_count": int(len(proposals)),
                        "decision_margin": float(decision_margin),
                        "decision_top1_score": float(decision_top1_score),
                        "decision_top2_score": float(decision_top2_score),
                        "top1_gain_dense": float(top1_gain),
                        "top2_gain_dense": float(top2_gain),
                        "top1_prob_dense": float(top1_prob),
                        "top2_prob_dense": float(top2_prob),
                        "plausible_count": int(probable_count),
                        "support_concentration": float(support_concentration),
                        "selected_support": float(selected_support),
                        "core_union_iou": float(core_union_iou),
                        "topk_oracle_recoverable": int(topk_oracle_recoverable),
                        "topk_oracle_gain_iou": float(topk_oracle_gain_iou),
                        "candidates": candidate_records,
                    }
                    (diag_json_dir / f"{image_id}.json").write_text(
                        json.dumps(diag_payload, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )

        write_binary_mask(out_masks / f"{image_id}.png", pred)
        met = rwtd_invariant_metrics(pred, gt)
        final_bf1, final_biou, _ = boundary_metrics(pred, gt, tol=int(args.boundary_tolerance))
        proposal_union_mask = _proposal_union(proposals) if proposals else np.zeros_like(gt, dtype=np.uint8)
        final_union_iou = _iou(pred, proposal_union_mask) if proposals else 0.0

        if args.save_audit_artifacts:
            case_root = audit_case_dir / image_id
            cand_root = case_root / "candidates"
            case_root.mkdir(parents=True, exist_ok=True)
            cand_root.mkdir(parents=True, exist_ok=True)

            _write_rgb(case_root / "image_rgb.png", image)
            write_binary_mask(case_root / "gt_partition_mask.png", gt)
            write_binary_mask(case_root / "baseline_scored_mask.png", core)
            write_binary_mask(case_root / "core_mask.png", core)
            write_binary_mask(case_root / "final_selected_mask.png", pred)
            write_binary_mask(case_root / "proposal_union_mask.png", proposal_union_mask)

            if dense_cands:
                dense_union = np.zeros_like(gt, dtype=np.uint8)
                for dm in dense_cands:
                    dense_union = np.logical_or(dense_union > 0, dm > 0).astype(np.uint8)
                write_binary_mask(case_root / "candidate_overlap_union.png", dense_union)
            else:
                write_binary_mask(case_root / "candidate_overlap_union.png", np.zeros_like(gt, dtype=np.uint8))
            cand_density = _candidate_stack_density(dense_cands, gt.shape)
            _write_gray_float(case_root / "candidate_stack_density.png", cand_density)
            _write_candidate_stack_overlay(case_root / "candidate_stack_overlay.png", image, cand_density)

            if support_freq_map is not None:
                _write_gray_float(case_root / "proposal_support_map.png", support_freq_map)
                _write_support_heatmap(case_root / "proposal_support_heatmap.png", support_freq_map)

            if selected_dense_idx >= 0 and selected_dense_idx < len(dense_cands):
                write_binary_mask(case_root / "selected_rescue_candidate_mask.png", dense_cands[selected_dense_idx])

            if dense_order_list:
                best_dense_idx = int(dense_order_list[0])
                if 0 <= best_dense_idx < len(dense_cands):
                    write_binary_mask(case_root / "best_ranked_candidate_mask.png", dense_cands[best_dense_idx])

            export_order = dense_order_list if dense_order_list else [int(j) for j in range(len(dense_cands))]
            if int(args.audit_max_candidates) > 0:
                export_order = export_order[: int(args.audit_max_candidates)]
            for rank, idx_dense in enumerate(export_order, start=1):
                if 0 <= int(idx_dense) < len(dense_cands):
                    write_binary_mask(cand_root / f"rank{rank:02d}_idx{int(idx_dense):03d}.png", dense_cands[int(idx_dense)])

            _write_binary_diff_overlay(case_root / "diff_baseline_vs_final.png", core, pred)
            _write_binary_diff_overlay(case_root / "diff_core_vs_final.png", core, pred)
            _write_binary_diff_overlay(case_root / "diff_final_vs_gt.png", gt, pred)
            _write_binary_diff_overlay(case_root / "diff_baseline_vs_gt.png", gt, core)

            case_manifest = {
                "image_id": int(id_num),
                "source": source,
                "switched": bool(source == "acute_learned_rescue"),
                "ambiguous_flag": bool(ambiguous_flag),
                "abstained_flag": bool(abstained_flag),
                "selected_dense_idx": int(selected_dense_idx),
                "selected_dense_rank": int(selected_dense_rank),
                "selected_mask_iou": float(selected_mask_iou),
                "final_iou": float(met.iou),
                "final_ari": float(met.ari),
                "core_iou": float(core_met.iou),
                "core_ari": float(core_met.ari),
                "decision_margin": float(decision_margin),
                "topk_oracle_recoverable": int(topk_oracle_recoverable),
                "topk_oracle_gain_iou": float(topk_oracle_gain_iou),
                "dense_candidate_count": int(len(dense_cands)),
                "exported_candidate_count": int(len(export_order)),
                "candidate_ranking_order": [int(v) for v in export_order],
                "candidate_records": candidate_records,
            }
            (case_root / "audit_manifest.json").write_text(
                json.dumps(case_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        rows.append(
            {
                "image_id": image_id,
                "acute_iou": float(met.iou),
                "acute_ari": float(met.ari),
                "acute_boundary_f1": float(final_bf1),
                "acute_boundary_iou": float(final_biou),
                "v9_iou": float(core_met.iou),
                "v9_ari": float(core_met.ari),
                "v9_boundary_f1": float(core_bf1),
                "v9_boundary_iou": float(core_biou),
                "delta_iou_vs_v9": float(met.iou - core_met.iou),
                "delta_ari_vs_v9": float(met.ari - core_met.ari),
                "delta_boundary_f1_vs_v9": float(final_bf1 - core_bf1),
                "delta_boundary_iou_vs_v9": float(final_biou - core_biou),
                "proposal_count_dense": int(len(proposals)),
                "dense_candidate_count": int(dense_count),
                "source": source,
                "switched": int(source == "acute_learned_rescue"),
                "ambiguous_flag": int(ambiguous_flag),
                "abstained_flag": int(abstained_flag),
                "decision_margin": float(decision_margin),
                "decision_top1_score": float(decision_top1_score),
                "decision_top2_score": float(decision_top2_score),
                "top1_gain_dense": float(top1_gain),
                "top2_gain_dense": float(top2_gain),
                "gain_margin_dense": float(top1_gain - top2_gain),
                "top1_prob_dense": float(top1_prob),
                "top2_prob_dense": float(top2_prob),
                "prob_margin_dense": float(top1_prob - top2_prob),
                "plausible_candidate_count": int(probable_count),
                "support_concentration": float(support_concentration),
                "selected_support": float(selected_support),
                "core_union_iou": float(core_union_iou),
                "final_union_iou": float(final_union_iou),
                "selected_dense_rank": int(selected_dense_rank),
                "topk_oracle_recoverable": int(topk_oracle_recoverable),
                "topk_oracle_gain_iou": float(topk_oracle_gain_iou),
                "mask_change_core_to_final": float(_mask_change_ratio(core, pred)),
                "best_pred_score": float(best_score),
                "best_pred_gain": float(best_gain),
                "best_pred_prob": float(best_prob),
                "audit_case_dir": str((audit_case_dir / image_id).resolve()) if args.save_audit_artifacts else "",
            }
        )
        if i % 32 == 0:
            print(f"[acute_rescue] processed {i}/{len(images)} images")

    def _mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0

    summary = {
        "dataset": "rwtd_kaust256",
        "num_images": int(len(rows)),
        "protocol": {
            "name": "texturesam2_acute_learned_rescue",
            "uses_rwtd_labels_for_training": False,
            "uses_rwtd_labels_for_hyperparameter_search": False,
            "uses_rwtd_labels_for_final_metric_reporting": True,
            "external_training_data": "PTD only",
            "frozen_after_ptd_validation_only": True,
        },
        "paths": {
            "rwtd_root": str(args.rwtd_root),
            "dense_prompt_masks_root": str(args.dense_prompt_masks_root),
            "v9_masks_root": str(args.v9_masks_root),
            "bundle_path": str(args.bundle_path),
            "ptd_v8_bundle": str(args.ptd_v8_bundle),
            "ptd_checkpoint": str(args.ptd_checkpoint),
        },
        "metrics": {
            "acute_miou": _mean("acute_iou"),
            "acute_ari": _mean("acute_ari"),
            "acute_boundary_f1": _mean("acute_boundary_f1"),
            "acute_boundary_iou": _mean("acute_boundary_iou"),
            "v9_miou": _mean("v9_iou"),
            "v9_ari": _mean("v9_ari"),
            "v9_boundary_f1": _mean("v9_boundary_f1"),
            "v9_boundary_iou": _mean("v9_boundary_iou"),
            "delta_miou_vs_v9": _mean("delta_iou_vs_v9"),
            "delta_ari_vs_v9": _mean("delta_ari_vs_v9"),
            "delta_boundary_f1_vs_v9": _mean("delta_boundary_f1_vs_v9"),
            "delta_boundary_iou_vs_v9": _mean("delta_boundary_iou_vs_v9"),
        },
        "switching": {
            "switched_count": int(switched),
            "switch_rate": float(switched / max(len(rows), 1)),
            "ambiguous_count": int(ambiguous_count),
            "ambiguous_rate": float(ambiguous_count / max(len(rows), 1)),
            "abstained_count": int(abstained_count),
            "abstained_rate": float(abstained_count / max(len(rows), 1)),
        },
        "bundle": {
            "gain_positive_margin": float(payload.get("gain_positive_margin", args.gain_positive_margin)),
            "core_surrogate": str(payload.get("core_surrogate", "unknown")),
            "label_mode": str(payload.get("label_mode", args.label_mode)),
            "oracle_slack": float(payload.get("oracle_slack", args.oracle_slack)),
            "safe_min_ari_gain": float(payload.get("safe_min_ari_gain", args.safe_min_ari_gain)),
            "ari_gain_weight": float(payload.get("ari_gain_weight", args.ari_gain_weight)),
            "choice_risk_penalty_ari": float(payload.get("choice_risk_penalty_ari", args.choice_risk_penalty_ari)),
            "choice_risk_penalty_iou": float(payload.get("choice_risk_penalty_iou", args.choice_risk_penalty_iou)),
            "risk_margin_ari": float(payload.get("risk_margin_ari", args.risk_margin_ari)),
            "risk_margin_iou": float(payload.get("risk_margin_iou", args.risk_margin_iou)),
            "cls_model": str(payload.get("cls_model_name", args.cls_model)),
            "supervision_source": str(payload.get("supervision_source", args.supervision_source)),
            "train_core_mode": str(payload.get("train_core_mode", args.train_core_mode)),
            "repair_core_prob": float(payload.get("repair_core_prob", args.repair_core_prob)),
            "wrong_side_negatives_per_sample": int(
                payload.get("wrong_side_negatives_per_sample", args.wrong_side_negatives_per_sample)
            ),
            "wrong_side_area_ratio_min": float(payload.get("wrong_side_area_ratio_min", args.wrong_side_area_ratio_min)),
            "wrong_side_area_ratio_max": float(payload.get("wrong_side_area_ratio_max", args.wrong_side_area_ratio_max)),
            "wrong_side_dilate_ksize": int(payload.get("wrong_side_dilate_ksize", args.wrong_side_dilate_ksize)),
            "decision_mode": str(payload.get("decision_architecture", args.decision_architecture)),
            "decision_margin_min": float(args.decision_margin_min),
            "abstain_margin": float(args.abstain_margin),
            "plausible_gap": float(args.plausible_gap),
            "support_prior_weight": float(args.support_prior_weight),
            "boundary_tolerance": int(args.boundary_tolerance),
            "save_audit_artifacts": bool(args.save_audit_artifacts),
            "audit_max_candidates": int(args.audit_max_candidates),
        },
        "training": train_metrics,
        "artifacts": {
            "summary_json": str((out_dir / "summary.json").resolve()),
            "per_image_csv": str((out_dir / "per_image.csv").resolve()),
            "masks_dir": str(out_masks.resolve()),
            "diagnostics_dir": str(diag_root.resolve()),
            "audit_cases_dir": str(audit_case_dir.resolve()) if args.save_audit_artifacts else "",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    payload, train_metrics = _load_or_train_bundle(args)
    run_rwtd(args, payload, train_metrics)


if __name__ == "__main__":
    main()
